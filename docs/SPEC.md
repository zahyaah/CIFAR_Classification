# Spec: CIFAR-10 Image Classification

Source brief: [AI Engineer - CIFAR-10 Image Classification.md](../AI%20Engineer%20-%20CIFAR-10%20Image%20Classification.md)

## Objective

Train a small CNN from scratch on CIFAR-10 with PyTorch, then improve it with one change from Part 2 of the brief and compare the two models on the same setup.

Part 2 choice: **Option A, residual block.** See [ADR-002](decisions/ADR-002-residual-block.md).

Optional extensions in scope: unit tests, mixed-precision training (flag), ONNX export with ONNX Runtime inference.
Out of scope: distributed training, serving with FastAPI/Flask.

## Assumptions

1. PyTorch, not TensorFlow ([ADR-001](decisions/ADR-001-pytorch.md)).
2. Reference hardware is an Apple M1 laptop (MPS backend). Code also runs on CUDA and CPU.
3. The official 50k/10k train/test split is used. 5,000 training images are held out for validation, so the test set is only touched once per model, after training.
4. Baseline and residual models get identical data, seed, optimizer, schedule and epoch count. Only the architecture differs.

## Tech Stack

Python 3.12, torch 2.14.0, torchvision 0.29.0, numpy, matplotlib, pytest, onnx + onnxscript + onnxruntime, Jupyter. Exact pins in [requirements.txt](../requirements.txt).

## Commands

```
Setup:     python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
Test:      .venv/bin/python -m pytest
Train:     .venv/bin/python -m cifar10.train --model baseline --run-name baseline
Evaluate:  .venv/bin/python -m cifar10.evaluate --run-dir outputs/baseline
Export:    .venv/bin/python -m cifar10.export_onnx --run-dir outputs/resnet
Compare:   .venv/bin/python -m cifar10.compare outputs/baseline outputs/resnet
Notebook:  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

## Project Structure

```
cifar10/            Python package: data, models, training, evaluation, export
tests/              pytest unit tests (no network, no dataset download)
notebooks/          Results notebook: curves, test accuracy, misclassified images, comparison
docs/SPEC.md        This file
docs/decisions/     Architecture decision records
outputs/<run>/      Per-run metrics, figures (committed) and checkpoints (git-ignored)
```

## Code Style

Type hints on public functions, docstrings on anything a reader calls from outside the module, comments only for non-obvious reasons. Configuration lives in one dataclass and is saved with every run.

```python
def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Return disjoint (train, val) index lists. Same seed, same split."""
```

## Testing Strategy

- pytest, tests in `tests/`, one file per module.
- Unit tests use random tensors or `torchvision.datasets.FakeData`. No test downloads CIFAR-10.
- Covered: transforms, split, loaders, model shapes and components, residual block shortcut, metrics, training step, checkpoint round-trip, seeding, plotting helpers, ONNX parity.
- Full training runs are verified by their saved metrics, not by tests.

## Boundaries

- Always: run `pytest` before committing; keep the test set out of model selection; save config with every run.
- Ask first: new dependencies; changing the Part 2 option; changing the train/val/test protocol.
- Never: use pretrained weights; tune on the test set; commit the dataset or checkpoints.

## Success Criteria

1. Data pipeline uses `DataLoader` with workers, normalization, random crop + flip augmentation, batching.
2. Baseline CNN has conv layers, BatchNorm, Dropout and a linear classification head, built from scratch.
3. Training loop reports loss and accuracy per epoch for train and validation, steps a learning-rate schedule and writes `last.pt` and `best.pt` checkpoints.
4. Evaluation produces loss/accuracy curves, test accuracy and a grid of misclassified test images.
5. Two runs with the same seed and device produce the same validation metrics for the first epoch.
6. `requirements.txt` pins every direct dependency.
7. Residual model trained under the same protocol, with a baseline-vs-residual table and overlaid curves.
8. ONNX export matches PyTorch logits within 1e-4 absolute tolerance.
9. README covers project structure, training setup, design decisions and how to run.

## Open Questions

None blocking. Epoch count (30) was chosen to fit a laptop time budget; see ADR-003.
