"""Save and restore training state.

Pattern from https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html:
store state_dicts plus plain Python values, and load with ``weights_only=True``
so loading never unpickles arbitrary objects.
"""
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None,
                    scaler=None, **extra: Any) -> None:
    """Write model/optimizer/scheduler/scaler state and ``extra`` values to ``path``.

    ``extra`` must hold only primitives, lists and dicts so ``weights_only`` loading works.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state_dict": model.state_dict(), **extra}
    for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            state[f"{key}_state_dict"] = obj.state_dict()
    # Write then rename, so an interrupted save never leaves a truncated checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None,
                    scaler=None, map_location: str | torch.device = "cpu") -> dict:
    """Load state into the given objects and return the full checkpoint dict."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and f"{key}_state_dict" in state:
            obj.load_state_dict(state[f"{key}_state_dict"])
    return state
