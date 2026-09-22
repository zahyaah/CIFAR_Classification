"""Seeding and device selection."""
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Follows https://docs.pytorch.org/docs/stable/notes/randomness.html.
    Bit-exact results are only guaranteed on the same device, PyTorch version and hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive NumPy and Python seeds from the torch worker seed.

    Must stay a top-level function so worker processes can import it under the
    ``spawn`` start method (the default on macOS and Windows).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preference: str = "auto") -> torch.device:
    """Return ``preference`` as a device, or the fastest available one for ``"auto"``."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
