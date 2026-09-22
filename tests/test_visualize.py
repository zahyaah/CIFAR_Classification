import torch

from cifar10.visualize import (
    find_misclassified,
    plot_comparison,
    plot_confusion_matrix,
    plot_history,
    plot_image_grid,
)

HISTORY = {"train_loss": [2.0, 1.5], "val_loss": [1.9, 1.6],
           "train_acc": [0.3, 0.5], "val_acc": [0.35, 0.45], "lr": [0.1, 0.05]}


def test_find_misclassified_returns_indices_of_wrong_predictions():
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 2, 2, 1])

    assert find_misclassified(preds, targets).tolist() == [1, 3]


def test_plot_history_writes_image(tmp_path):
    path = tmp_path / "curves.png"
    plot_history(HISTORY, path)
    assert path.stat().st_size > 0


def test_plot_comparison_writes_image(tmp_path):
    path = tmp_path / "compare.png"
    plot_comparison({"baseline": HISTORY, "resnet": HISTORY}, path)
    assert path.stat().st_size > 0


def test_plot_image_grid_writes_image(tmp_path):
    path = tmp_path / "grid.png"
    images = torch.rand(6, 3, 32, 32)
    plot_image_grid(images, ["a"] * 6, path, ncols=3)
    assert path.stat().st_size > 0


def test_plot_confusion_matrix_writes_image(tmp_path):
    path = tmp_path / "cm.png"
    plot_confusion_matrix(torch.eye(3, dtype=torch.long), ["a", "b", "c"], path)
    assert path.stat().st_size > 0
