import matplotlib.pyplot as plt


def plot_training_curves(
    history,
    keys=("train_loss", "val_loss"),
    save_path=None
):
    """
    Plot training and validation curves.

    Args:
        history: dict with metric lists
    """
    plt.figure(figsize=(6, 4))
    for k in keys:
        if k in history:
            plt.plot(history[k], label=k)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curve(confidence, accuracy, save_path=None):
    """
    Reliability diagram.
    """
    plt.figure(figsize=(4, 4))
    plt.plot(confidence, accuracy, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
