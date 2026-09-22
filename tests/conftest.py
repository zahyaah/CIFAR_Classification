import pytest
import torch
from torchvision.datasets import FakeData

from cifar10.data import build_transforms


@pytest.fixture
def fake_train_set():
    """Stand-in for CIFAR-10: PIL images of 32x32, 10 classes, no download."""
    return FakeData(size=64, image_size=(3, 32, 32), num_classes=10,
                    transform=build_transforms(train=True), random_offset=0)


@pytest.fixture
def fake_eval_set():
    return FakeData(size=64, image_size=(3, 32, 32), num_classes=10,
                    transform=build_transforms(train=False), random_offset=0)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
