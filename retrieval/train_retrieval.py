import torch
from retrieval.proximal_solver import accelerated_proximal_gradient
from retrieval.outer_objective import outer_loss


def train_retrieval(
    retrieval_net,
    descriptors,
    thetas,
    M,
    r,
    optimizer,
    lambda_entropy,
    lambda_sparse,
    epochs=50
):
    """
    Phase-2 training loop for memory retrieval.

    Args:
        descriptors: Tensor [T, D]
        thetas: Tensor [T, d]
        M: Tensor [K, d]
    """
    retrieval_net.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for z_t, theta_t in zip(descriptors, thetas):
            z_t = z_t.unsqueeze(0)
            logits = retrieval_net(z_t).squeeze(0)

            a_hat = accelerated_proximal_gradient(
                logits,
                M,
                theta_t,
                r
            )

            theta_hat = M.T @ a_hat

            loss = outer_loss(
                theta_hat,
                theta_t,
                a_hat,
                lambda_entropy,
                lambda_sparse
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Loss = {total_loss:.4f}")
