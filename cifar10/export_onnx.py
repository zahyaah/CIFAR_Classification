"""Export a trained model to ONNX and check it against PyTorch with ONNX Runtime.

Usage: python -m cifar10.export_onnx --run-dir outputs/resnet
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cifar10.checkpoint import load_checkpoint
from cifar10.config import TrainConfig
from cifar10.data import build_transforms, CLASSES
from cifar10.models import build_model


def export_to_onnx(model: nn.Module, path: str | Path) -> Path:
    """Export ``model`` with a dynamic batch dimension.

    Uses the ``torch.export``-based exporter (``dynamo=True``, the default since
    PyTorch 2.9) with ``dynamic_shapes`` in place of the deprecated ``dynamic_axes``.
    Source: https://docs.pytorch.org/docs/stable/onnx_export.html
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.randn(2, 3, 32, 32)
    program = torch.onnx.export(
        model, (example,), dynamo=True,
        input_names=["images"], output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4096)},),
    )
    # Weights are under 2 GB, so keep them inside the .onnx file.
    program.save(str(path), external_data=False)
    return path


def main() -> None:
    import onnxruntime as ort
    from torchvision import datasets

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-images", type=int, default=256)
    args = parser.parse_args()

    state = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = TrainConfig.from_dict(state["config"])
    model = build_model(config.model, dropout=config.dropout)
    load_checkpoint(args.run_dir / "best.pt", model)
    onnx_path = export_to_onnx(model, args.run_dir / "model.onnx")

    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=build_transforms(train=False))
    images = torch.stack([test_set[i][0] for i in range(args.num_images)])
    labels = np.array([test_set[i][1] for i in range(args.num_images)])
    with torch.no_grad():
        torch_logits = model(images).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    [onnx_logits] = session.run(None, {session.get_inputs()[0].name: images.numpy()})
    max_diff = float(np.abs(onnx_logits - torch_logits).max())
    agreement = float((onnx_logits.argmax(1) == torch_logits.argmax(1)).mean())
    accuracy = float((onnx_logits.argmax(1) == labels).mean())

    print(f"Exported {onnx_path}")
    print(f"ONNX Runtime vs PyTorch on {args.num_images} test images: "
          f"max |logit diff| = {max_diff:.2e}, prediction agreement = {agreement:.2%}, "
          f"ONNX accuracy = {accuracy:.2%}")
    print("First 8 predictions:", [CLASSES[i] for i in onnx_logits.argmax(1)[:8]])


if __name__ == "__main__":
    main()
