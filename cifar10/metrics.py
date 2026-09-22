"""Classification metrics on tensors."""
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of rows whose argmax equals the target."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Counts matrix: row = true class, column = predicted class."""
    flat = targets.long() * num_classes + preds.long()
    return torch.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """Recall per class. Classes with no samples get 0."""
    totals = cm.sum(dim=1)
    return torch.where(totals > 0, cm.diag() / totals.clamp(min=1), torch.zeros_like(totals, dtype=torch.float))
