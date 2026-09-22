"""Plots for training curves, image grids and confusion matrices."""
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def find_misclassified(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(preds != targets).flatten()


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, path: str | Path, title: str = "") -> None:
    """Loss and accuracy per epoch, train vs validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        ax_loss.plot(epochs, history[f"{split}_loss"], label=split)
        ax_acc.plot(epochs, [100 * a for a in history[f"{split}_acc"]], label=split)
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_comparison(histories: dict[str, dict], path: str | Path) -> None:
    """Validation loss and accuracy of several runs on shared axes."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        ax_loss.plot(epochs, history["val_loss"], label=name)
        ax_acc.plot(epochs, [100 * a for a in history["val_acc"]], label=name)
    ax_loss.set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
    ax_acc.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)


def plot_image_grid(images: torch.Tensor, captions: Sequence[str], path: str | Path,
                    ncols: int = 8) -> None:
    """Show (N,3,H,W) images in [0,1] with one caption each."""
    n = len(images)
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 2.1 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, caption in zip(axes.flat, images, captions):
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(caption, fontsize=8)
    _save(fig, path)


def plot_confusion_matrix(cm: torch.Tensor, classes: Sequence[str], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true")
    threshold = cm.max().item() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = cm[i, j].item()
            ax.text(j, i, value, ha="center", va="center", fontsize=7,
                    color="white" if value > threshold else "black")
    fig.colorbar(im, ax=ax)
    _save(fig, path)
