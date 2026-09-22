"""Evaluate a finished run on the CIFAR-10 test set.

Loads best.pt (highest validation accuracy), then writes test_metrics.json,
training curves, a confusion matrix and a grid of misclassified test images.

Usage: python -m cifar10.evaluate --run-dir outputs/baseline
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar10.checkpoint import load_checkpoint
from cifar10.config import TrainConfig
from cifar10.data import CLASSES, build_dataloaders, denormalize
from cifar10.engine import predict
from cifar10.metrics import confusion_matrix, per_class_accuracy
from cifar10.models import build_model, count_parameters
from cifar10.utils import get_device
from cifar10.visualize import (find_misclassified, plot_confusion_matrix, plot_history,
                               plot_image_grid)


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device,
                   out_dir: str | Path, num_misclassified: int = 32) -> dict:
    """Compute test metrics and write them plus figures to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits, targets = predict(model.to(device), test_loader, device)
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, targets, num_classes=len(CLASSES))

    metrics = {
        "test_acc": (preds == targets).float().mean().item(),
        "test_loss": nn.functional.cross_entropy(logits, targets).item(),
        "num_parameters": count_parameters(model),
        "per_class_acc": dict(zip(CLASSES, per_class_accuracy(cm).tolist())),
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(cm, CLASSES, out_dir / "confusion_matrix.png")
    wrong = find_misclassified(preds, targets)[:num_misclassified]
    # Rebuild images from the loader in order rather than holding the whole test set in memory.
    images = _gather(test_loader, wrong)
    confidence = logits[wrong].softmax(dim=1).max(dim=1).values
    captions = [f"true: {CLASSES[targets[i]]}\npred: {CLASSES[preds[i]]} ({c:.0%})"
                for i, c in zip(wrong.tolist(), confidence.tolist())]
    plot_image_grid(denormalize(images), captions, out_dir / "misclassified.png")
    return metrics


def _gather(loader: DataLoader, indices: torch.Tensor) -> torch.Tensor:
    wanted = set(indices.tolist())
    found: dict[int, torch.Tensor] = {}
    offset = 0
    for images, _ in loader:
        for j in range(len(images)):
            if offset + j in wanted:
                found[offset + j] = images[j]
        offset += len(images)
    return torch.stack([found[i] for i in indices.tolist()]) if found else torch.empty(0, 3, 32, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run on the CIFAR-10 test set.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default=None, help="defaults to the run's data_dir")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    state = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = TrainConfig.from_dict(state["config"])
    device = get_device(args.device)
    model = build_model(config.model, dropout=config.dropout)
    load_checkpoint(args.run_dir / "best.pt", model)

    _, _, test_loader = build_dataloaders(args.data_dir or config.data_dir, config.batch_size,
                                          config.val_size, config.seed, config.num_workers)
    metrics = evaluate_model(model, test_loader, device, args.run_dir)
    history = json.loads((args.run_dir / "history.json").read_text())
    plot_history(history, args.run_dir / "curves.png", title=f"{args.run_dir.name} ({config.model})")

    print(f"best.pt from epoch {state['epoch']} (val acc {state['best_val_acc']:.2%})")
    print(f"test accuracy {metrics['test_acc']:.2%}, test loss {metrics['test_loss']:.4f}")
    for name, acc in metrics["per_class_acc"].items():
        print(f"  {name:<10} {acc:.2%}")


if __name__ == "__main__":
    main()
