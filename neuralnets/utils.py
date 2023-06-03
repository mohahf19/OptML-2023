import pickle

from matplotlib import pyplot as plt


def get_losses_indices(
    losses: list[tuple[int, float]]
) -> tuple[list[float], list[int]]:
    loss_values = [loss for _, loss in losses]
    loss_indices = [idx for idx, _ in losses]
    return loss_values, loss_indices


def visualize_losses(
    output_path: str,
    tr_losses: list[tuple[int, float]],
    val_losses: list[tuple[int, float]] | None = None,
    test_losses: list[tuple[int, float]] | None = None,
    snapshots: list[tuple[int, bool]] | None = None,
    title: str = "Losses",
):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    tr_loss_values, tr_loss_indices = get_losses_indices(tr_losses)
    ymin, ymax = -0.5, round(5 * tr_loss_values[0])
    ax.plot(tr_loss_indices, tr_loss_values, label="Train Loss", linewidth=1, alpha=0.6)
    if val_losses is not None:
        val_loss_values, val_loss_indices = get_losses_indices(val_losses)
        ax.plot(val_loss_indices, val_loss_values, label="Validation Loss", linewidth=5)

    if test_losses is not None:
        test_loss_values, test_loss_indices = get_losses_indices(test_losses)
        ax.plot(test_loss_indices, test_loss_values, label="Test Loss")

    if snapshots is not None:
        snapshot_indices = [idx for idx, snapshot_taken in snapshots if snapshot_taken]
        # add a vertical dashed line at snapshot indices
        ax.vlines(
            snapshot_indices,
            ymin=ymin,
            ymax=ymax,
            label="Snapshot",
            linestyles="dashed",
            color="black",
            linewidth=1,
            alpha=0.5,
        )

    ax.legend()
    # ax.yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    ax.set_ylim([ymin, ymax])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def visualize_losses_from_file(filepath: str, output_path: str, title: str = "Losses"):
    with open(filepath, "rb") as f:
        losses = pickle.load(f)
    if "took_snapshots" in losses:
        visualize_losses(
            output_path,
            losses["train"],
            losses["val"],
            snapshots=losses["took_snapshots"],
            title=title,
        )
    else:
        visualize_losses(output_path, losses["train"], losses["val"], title=title)
