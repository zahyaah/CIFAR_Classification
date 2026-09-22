"""Train a model on CIFAR-10.

Usage: python -m cifar10.train --model baseline --run-name baseline
Every TrainConfig field is a command-line flag (--max-lr, --epochs, ...).
"""
import argparse
from dataclasses import fields

from cifar10.config import TrainConfig
from cifar10.data import build_dataloaders
from cifar10.engine import fit
from cifar10.models import MODELS, build_model, count_parameters
from cifar10.utils import get_device, seed_everything


def parse_args() -> tuple[TrainConfig, bool]:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif f.name == "model":
            parser.add_argument(flag, choices=sorted(MODELS), default=default)
        else:
            parser.add_argument(flag, type=type(default) if default is not None else str, default=default)
    parser.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    return TrainConfig(**args), resume


def main() -> None:
    config, resume = parse_args()
    seed_everything(config.seed)
    device = get_device(config.device)

    train_loader, val_loader, _ = build_dataloaders(
        config.data_dir, config.batch_size, config.val_size, config.seed,
        config.num_workers, pin_memory=device.type == "cuda",
    )
    model = build_model(config.model, dropout=config.dropout)
    print(f"model={config.model} params={count_parameters(model):,} device={device} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    fit(model, train_loader, val_loader, config, device, resume=resume)


if __name__ == "__main__":
    main()
