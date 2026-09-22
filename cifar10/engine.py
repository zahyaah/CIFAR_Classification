"""Training and evaluation loops."""
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar10.checkpoint import load_checkpoint, save_checkpoint
from cifar10.config import TrainConfig

HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time")


def autocast_context(device: torch.device, enabled: bool):
    """Mixed-precision context: float16 on CUDA, bfloat16 on CPU, off elsewhere.

    Source: https://docs.pytorch.org/docs/stable/amp.html. MPS is not listed as
    an autocast device in those docs, so AMP stays off on Apple GPUs.
    """
    if not enabled or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    scheduler=None, scaler: torch.amp.GradScaler | None = None,
                    amp: bool = False) -> tuple[float, float]:
    """Run one pass over ``loader``. Steps ``scheduler`` after every batch.

    Returns (mean loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch = targets.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Return (mean loss, accuracy) with the model in eval mode."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, targets) for the whole loader, on CPU, in loader order."""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        all_logits.append(model(images.to(device)).float().cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    # Weight decay on conv/linear weights only. Decaying BatchNorm scales and
    # biases pushes them toward zero without improving generalization.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.max_lr, momentum=config.momentum, nesterov=True,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int):
    """One-cycle schedule, stepped per batch.

    Source: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
    )


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        config: TrainConfig, device: torch.device, resume: bool = False) -> dict[str, list[float]]:
    """Train for ``config.epochs`` epochs, validating and checkpointing after each one.

    Writes ``last.pt`` every epoch and ``best.pt`` when validation accuracy improves,
    plus ``history.json`` and ``config.json``, under ``output_dir/run_name``.
    With ``resume=True`` training continues from ``last.pt``.
    """
    run_dir = Path(config.output_dir) / config.run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    use_scaler = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler) if use_scaler else None

    history: dict[str, list[float]] = {k: [] for k in HISTORY_KEYS}
    start_epoch, best_val_acc = 0, -1.0
    if resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if state["config"]["epochs"] != config.epochs:
            # The one-cycle schedule's length is fixed by the epoch count it was built with.
            raise ValueError(f"Checkpoint was trained for epochs={state['config']['epochs']}, "
                             f"config asks for epochs={config.epochs}.")
        state = load_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, map_location=device)
        start_epoch, history, best_val_acc = state["epoch"], state["history"], state["best_val_acc"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, config.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                scheduler=scheduler, scaler=scaler, amp=config.amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - start

        for key, value in zip(HISTORY_KEYS, (train_loss, train_acc, val_loss, val_acc, lr, elapsed)):
            history[key].append(value)
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        state = dict(epoch=epoch, history=history, best_val_acc=best_val_acc, config=config.to_dict())
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, **state)
        if is_best:
            save_checkpoint(run_dir / "best.pt", model, **state)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch {epoch:3d}/{config.epochs} | lr {lr:.4f} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {elapsed:.1f}s" + (" *" if is_best else ""),
              flush=True)
    return history
