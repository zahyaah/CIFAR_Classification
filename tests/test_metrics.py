import torch

from cifar10.metrics import accuracy, confusion_matrix, per_class_accuracy


def test_accuracy_counts_argmax_matches():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 1])

    assert accuracy(logits, targets) == 0.75


def test_confusion_matrix_rows_are_true_labels():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 0, 1, 2])

    cm = confusion_matrix(preds, targets, num_classes=3)

    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_per_class_accuracy_divides_diagonal_by_row_sum():
    cm = torch.tensor([[3, 1], [0, 2]])

    assert per_class_accuracy(cm).tolist() == [0.75, 1.0]


def test_per_class_accuracy_handles_absent_class():
    cm = torch.tensor([[2, 0], [0, 0]])

    assert per_class_accuracy(cm).tolist() == [1.0, 0.0]
