import torch


def proximal_top_r(v, r):
    """
    Hard top-r projection.

    Args:
        v: Tensor [K]
        r: int

    Returns:
        projected vector
    """
    if r >= v.numel():
        return v

    values, indices = torch.topk(v, r)
    out = torch.zeros_like(v)
    out[indices] = values
    return out


def accelerated_proximal_gradient(
    logits,
    M,
    theta,
    r,
    step_size=1e-2,
    n_iters=50
):
    """
    Solve proximal objective:

        min_a || M^T a - theta ||^2
        s.t. ||a||_0 <= r

    Args:
        logits: Tensor [K]
        M: Tensor [K, d]
        theta: Tensor [d]

    Returns:
        a_hat: Tensor [K]
    """
    a = logits.clone()
    y = a.clone()
    t = 1.0

    for _ in range(n_iters):
        grad = 2 * M @ (M.T @ y - theta)
        a_next = proximal_top_r(y - step_size * grad, r)

        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = a_next + ((t - 1) / t_next) * (a_next - a)

        a = a_next
        t = t_next

    return a
