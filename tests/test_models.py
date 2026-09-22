import pytest
import torch
from torch import nn

from cifar10.models import (
    BaselineCNN,
    ResidualBlock,
    ResidualCNN,
    build_model,
    count_parameters,
)


@pytest.mark.parametrize("name", ["baseline", "resnet"])
def test_model_outputs_one_logit_per_class(name, batch):
    images, _ = batch
    model = build_model(name, num_classes=10)

    assert model(images).shape == (8, 10)


@pytest.mark.parametrize("name", ["baseline", "resnet"])
def test_model_contains_required_components(name):
    model = build_model(name)
    kinds = {type(m) for m in model.modules()}

    assert nn.Conv2d in kinds
    assert nn.BatchNorm2d in kinds
    assert nn.Dropout in kinds
    assert isinstance(model.head[-1], nn.Linear)
    assert model.head[-1].out_features == 10


@pytest.mark.parametrize("name", ["baseline", "resnet"])
def test_model_is_small(name):
    assert count_parameters(build_model(name)) < 500_000


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg")


def test_dropout_rate_is_configurable():
    model = BaselineCNN(dropout=0.5)
    rates = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]

    assert rates and all(p == 0.5 for p in rates)


def test_eval_mode_is_deterministic(batch):
    images, _ = batch
    model = ResidualCNN().eval()

    with torch.no_grad():
        assert torch.equal(model(images), model(images))


def test_residual_block_keeps_shape_with_identity_shortcut():
    block = ResidualBlock(16, 16)

    assert isinstance(block.shortcut, nn.Identity)
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)


def test_residual_block_projects_shortcut_when_channels_change():
    block = ResidualBlock(16, 32)

    assert isinstance(block.shortcut, nn.Sequential)
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_residual_block_projects_shortcut_when_downsampling():
    block = ResidualBlock(16, 16, stride=2)

    assert not isinstance(block.shortcut, nn.Identity)
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 16, 4, 4)


def test_residual_block_adds_input_to_residual_branch():
    block = ResidualBlock(4, 4).eval()
    # Zero the last BatchNorm scale so the residual branch outputs zero.
    nn.init.zeros_(block.bn2.weight)
    nn.init.zeros_(block.bn2.bias)
    x = torch.rand(1, 4, 5, 5)  # non-negative, so ReLU(x) == x

    with torch.no_grad():
        assert torch.allclose(block(x), x)


def test_residual_model_differs_from_baseline_only_by_shortcuts():
    baseline = count_parameters(BaselineCNN())
    residual = count_parameters(ResidualCNN())
    # 1x1 projections (3->32, 32->64, 64->128) plus their BatchNorms.
    projections = (3 * 32 + 32 * 64 + 64 * 128) + 2 * (32 + 64 + 128)

    assert residual - baseline == projections
