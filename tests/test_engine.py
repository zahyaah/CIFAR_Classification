import json

import pytest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10.config import TrainConfig
from cifar10.engine import evaluate, fit, predict, train_one_epoch
from cifar10.models import build_model


def tiny_loader(n=32, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    labels = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_train_one_epoch_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    loader = tiny_loader(n=16, batch_size=16)
    model = build_model("baseline", dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    first_loss, _ = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    for _ in range(20):
        last_loss, last_acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert last_loss < first_loss / 2
    assert last_acc > 0.9


def test_train_one_epoch_steps_scheduler_once_per_batch():
    loader = tiny_loader(n=32, batch_size=8)
    model = build_model("baseline")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))

    train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), scheduler=scheduler)

    assert scheduler.last_epoch == 4


def test_evaluate_does_not_change_weights_and_returns_valid_accuracy():
    model = build_model("resnet")
    before = {k: v.clone() for k, v in model.state_dict().items()}

    loss, acc = evaluate(model, tiny_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))

    assert loss > 0
    assert 0.0 <= acc <= 1.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_predict_returns_logits_and_targets_in_loader_order():
    loader = tiny_loader(n=20, batch_size=8)
    logits, targets = predict(build_model("baseline"), loader, torch.device("cpu"))

    assert logits.shape == (20, 10)
    assert torch.equal(targets, loader.dataset.tensors[1])


def small_config(tmp_path, **overrides):
    values = dict(model="baseline", epochs=2, batch_size=8, max_lr=0.05,
                  num_workers=0, output_dir=str(tmp_path), run_name="test-run")
    values.update(overrides)
    return TrainConfig(**values)


def test_fit_writes_history_config_and_checkpoints(tmp_path):
    config = small_config(tmp_path)
    model = build_model(config.model)

    history = fit(model, tiny_loader(), tiny_loader(seed=1), config, torch.device("cpu"))

    run_dir = tmp_path / "test-run"
    for name in ("last.pt", "best.pt", "history.json", "config.json"):
        assert (run_dir / name).exists(), name
    assert set(history) >= {"train_loss", "train_acc", "val_loss", "val_acc", "lr"}
    assert all(len(values) == 2 for values in history.values())
    assert json.loads((run_dir / "history.json").read_text()) == history
    assert json.loads((run_dir / "config.json").read_text())["model"] == "baseline"


def test_fit_best_checkpoint_holds_highest_validation_accuracy(tmp_path):
    config = small_config(tmp_path, epochs=3)
    history = fit(build_model("baseline"), tiny_loader(), tiny_loader(seed=1), config, torch.device("cpu"))

    best = torch.load(tmp_path / "test-run" / "best.pt", weights_only=True)

    assert best["best_val_acc"] == max(history["val_acc"])


class FailOnSecondEpoch:
    """Loader wrapper that raises when iterated a second time, simulating a crash."""

    def __init__(self, loader):
        self.loader, self.passes = loader, 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.passes += 1
        if self.passes > 1:
            raise RuntimeError("simulated crash")
        return iter(self.loader)


def test_fit_resume_continues_from_last_checkpoint(tmp_path):
    config = small_config(tmp_path, epochs=2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        fit(build_model("baseline"), FailOnSecondEpoch(tiny_loader()), tiny_loader(seed=1),
            config, torch.device("cpu"))
    assert torch.load(tmp_path / "test-run" / "last.pt", weights_only=True)["epoch"] == 1

    history = fit(build_model("baseline"), tiny_loader(), tiny_loader(seed=1),
                  config, torch.device("cpu"), resume=True)

    assert len(history["val_acc"]) == 2
    assert torch.load(tmp_path / "test-run" / "last.pt", weights_only=True)["epoch"] == 2


def test_fit_resume_rejects_a_different_epoch_budget(tmp_path):
    fit(build_model("baseline"), tiny_loader(), tiny_loader(seed=1),
        small_config(tmp_path, epochs=1), torch.device("cpu"))

    with pytest.raises(ValueError, match="epochs"):
        fit(build_model("baseline"), tiny_loader(), tiny_loader(seed=1),
            small_config(tmp_path, epochs=3), torch.device("cpu"), resume=True)
