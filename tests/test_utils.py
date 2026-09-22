import random

import numpy as np
import torch

from cifar10.utils import get_device, seed_everything


def test_seed_everything_makes_torch_numpy_and_random_repeatable():
    seed_everything(123)
    first = (torch.rand(3), np.random.rand(3), random.random())
    seed_everything(123)
    second = (torch.rand(3), np.random.rand(3), random.random())

    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_get_device_honours_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_an_available_backend():
    assert get_device("auto").type in {"cuda", "mps", "cpu"}
