import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from cifar10.evaluate import evaluate_model
from cifar10.models import build_model


def test_evaluate_model_writes_metrics_and_figures(tmp_path):
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(40, 3, 32, 32, generator=g),
                                      torch.randint(0, 10, (40,), generator=g)), batch_size=16)

    metrics = evaluate_model(build_model("baseline"), loader, torch.device("cpu"), tmp_path)

    assert 0.0 <= metrics["test_acc"] <= 1.0
    assert len(metrics["per_class_acc"]) == 10
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 40
    assert json.loads((tmp_path / "test_metrics.json").read_text()) == metrics
    for name in ("misclassified.png", "confusion_matrix.png"):
        assert (tmp_path / name).stat().st_size > 0
