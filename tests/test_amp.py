from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10.engine import autocast_context, train_one_epoch
from cifar10.models import build_model


def test_autocast_is_disabled_when_amp_is_off():
    assert isinstance(autocast_context(torch.device("cpu"), enabled=False), nullcontext)


def test_autocast_is_disabled_on_mps_because_torch_does_not_document_it():
    assert isinstance(autocast_context(torch.device("mps"), enabled=True), nullcontext)


def test_autocast_on_cpu_uses_bfloat16():
    context = autocast_context(torch.device("cpu"), enabled=True)

    assert isinstance(context, torch.autocast)
    with context:
        assert (torch.randn(2, 4) @ torch.randn(4, 2)).dtype == torch.bfloat16


def test_training_step_under_amp_updates_weights_and_keeps_loss_finite():
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(16, 3, 32, 32, generator=g),
                                      torch.randint(0, 10, (16,), generator=g)), batch_size=8)
    model = build_model("baseline")
    before = model.head[-1].weight.clone()

    loss, acc = train_one_epoch(model, loader, nn.CrossEntropyLoss(),
                                torch.optim.SGD(model.parameters(), lr=0.1),
                                torch.device("cpu"), amp=True)

    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 1.0
    assert not torch.equal(before, model.head[-1].weight)
