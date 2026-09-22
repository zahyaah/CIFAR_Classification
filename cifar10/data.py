"""CIFAR-10 loading, preprocessing and DataLoaders."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from cifar10.utils import seed_worker

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Per-channel mean and std of the 50,000 CIFAR-10 training images.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool) -> v2.Compose:
    """Return the train (augmenting) or eval (deterministic) transform.

    Uses the torchvision v2 API; ``ToTensor`` is deprecated in favour of
    ``ToImage`` + ``ToDtype(float32, scale=True)``.
    Source: https://docs.pytorch.org/vision/stable/transforms.html
    """
    # Augment on uint8 images, then convert: torchvision recommends uint8 for geometric ops.
    augment = [v2.RandomCrop(32, padding=4, padding_mode="reflect"),
               v2.RandomHorizontalFlip()] if train else []
    return v2.Compose([
        v2.ToImage(),
        *augment,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ``Normalize`` so images can be displayed. Works on (C,H,W) or (N,C,H,W)."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Return disjoint (train, val) index lists. Same seed, same split."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def make_loaders(
    train_set: Dataset,
    train_eval_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders.

    ``train_set`` and ``train_eval_set`` hold the same images with different
    transforms. The validation split is taken from ``train_eval_set`` so
    validation images are never augmented.
    """
    train_idx, val_idx = split_indices(len(train_set), val_size, seed)
    workers = dict(num_workers=num_workers, pin_memory=pin_memory,
                   persistent_workers=num_workers > 0)
    # Seeded generator + worker_init_fn make shuffling and augmentation repeatable.
    # Source: https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    train_loader = DataLoader(
        Subset(train_set, train_idx), batch_size=batch_size, shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), **workers,
    )
    val_loader = DataLoader(Subset(train_eval_set, val_idx), batch_size=batch_size * 2,
                            shuffle=False, **workers)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **workers)
    return train_loader, val_loader, test_loader


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and return train/val/test loaders."""
    root = str(data_dir)
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=build_transforms(train=True))
    train_eval_set = datasets.CIFAR10(root, train=True, download=False, transform=build_transforms(train=False))
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=build_transforms(train=False))
    return make_loaders(train_set, train_eval_set, test_set, batch_size, val_size, seed,
                        num_workers, pin_memory)
