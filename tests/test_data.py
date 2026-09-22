import numpy as np
import torch
from PIL import Image

from cifar10.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CLASSES,
    build_transforms,
    denormalize,
    make_loaders,
    split_indices,
)


def test_there_are_ten_classes():
    assert len(CLASSES) == 10


def test_eval_transform_returns_normalized_float_tensor():
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))

    tensor = build_transforms(train=False)(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    expected = (1.0 - torch.tensor(CIFAR10_MEAN)) / torch.tensor(CIFAR10_STD)
    assert torch.allclose(tensor[:, 0, 0], expected, atol=1e-5)


def test_eval_transform_is_deterministic():
    image = Image.effect_noise((32, 32), 64).convert("RGB")
    transform = build_transforms(train=False)

    assert torch.equal(transform(image), transform(image))


def test_train_transform_augments():
    image = Image.effect_noise((32, 32), 64).convert("RGB")
    transform = build_transforms(train=True)
    torch.manual_seed(0)

    outputs = [transform(image) for _ in range(10)]

    assert all(o.shape == (3, 32, 32) for o in outputs)
    assert any(not torch.equal(outputs[0], o) for o in outputs[1:])


def test_denormalize_inverts_normalization():
    image = Image.effect_noise((32, 32), 64).convert("RGB")
    normalized = build_transforms(train=False)(image)

    restored = denormalize(normalized)

    assert restored.min() >= 0.0 and restored.max() <= 1.0
    raw = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    assert torch.allclose(restored, raw, atol=1e-5)


def test_split_indices_are_disjoint_and_cover_everything():
    train, val = split_indices(n=100, val_size=20, seed=0)

    assert len(train) == 80 and len(val) == 20
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(100))


def test_split_indices_same_seed_same_split():
    assert split_indices(100, 20, seed=7) == split_indices(100, 20, seed=7)


def test_split_indices_different_seed_different_split():
    assert split_indices(100, 20, seed=1) != split_indices(100, 20, seed=2)


def test_make_loaders_yields_batches_with_expected_shapes(fake_train_set, fake_eval_set):
    train_loader, val_loader, test_loader = make_loaders(
        fake_train_set, fake_eval_set, fake_eval_set,
        batch_size=16, val_size=16, seed=0, num_workers=0,
    )

    images, labels = next(iter(train_loader))
    assert images.shape == (16, 3, 32, 32)
    assert labels.shape == (16,)
    assert len(train_loader.dataset) == 48
    assert len(val_loader.dataset) == 16
    assert len(test_loader.dataset) == 64


def test_make_loaders_train_order_is_reproducible(fake_train_set, fake_eval_set):
    def first_labels():
        loader, _, _ = make_loaders(fake_train_set, fake_eval_set, fake_eval_set,
                                    batch_size=16, val_size=16, seed=3, num_workers=0)
        return next(iter(loader))[1]

    assert torch.equal(first_labels(), first_labels())


def test_make_loaders_val_uses_eval_transform(fake_train_set, fake_eval_set):
    _, val_loader, _ = make_loaders(fake_train_set, fake_eval_set, fake_eval_set,
                                    batch_size=16, val_size=16, seed=0, num_workers=0)

    first = next(iter(val_loader))[0]
    second = next(iter(val_loader))[0]

    assert torch.equal(first, second)
