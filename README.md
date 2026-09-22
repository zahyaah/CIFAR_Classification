# CIFAR-10 Image Classification

A small CNN trained from scratch on CIFAR-10 with PyTorch, plus a residual variant for Part 2 of the brief (Option A). Both models train under the same protocol, so the comparison isolates the effect of the skip connections.

## Results

Test accuracy after 30 epochs, measured once per run on the 10,000 held-out test images, with the checkpoint chosen by validation accuracy. Each model was trained twice, with seeds 42 and 43 and nothing else changed.

| Model | Parameters | Seed 42 | Seed 43 | Mean | Minutes on an M1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 288,746 | 89.88% | 89.91% | 89.90% | 10 to 14 |
| Residual CNN | 299,530 | 89.85% | 89.98% | 89.92% | 13 to 14 |

The residual model finishes 0.02 points ahead on average. Two seeds of the same model differ by 0.03 points (baseline) and 0.13 points (residual), so the gap between the architectures is smaller than the spread between reruns of either one. At this depth the skip connections buy no accuracy.

One difference does survive across both seeds: the residual model leads during the first half of training. Its mean validation accuracy over epochs 1 to 15 is 73.6% against 72.4% at seed 42, and 74.1% against 72.2% at seed 43. The shortcuts help the optimizer early and the baseline catches up as the learning rate falls.

That fits what residual connections were built for. He et al. introduced them to train networks of 50 and 100 layers, where plain stacks stop converging. Six convolutional layers with BatchNorm converge without help, so there is no degradation for the shortcuts to fix. A clear gain would need a deeper network, which would change depth, width and parameter count at the same time and break the like-for-like comparison ([ADR-002](docs/decisions/ADR-002-residual-block.md)).

Per-class accuracy follows the usual CIFAR-10 pattern: automobiles, ships and trucks above 94%, cats at 79.5% and dogs at 83.8% for the baseline. The confusion matrices in `outputs/` show most errors falling between cat and dog, and between bird and airplane.

Full numbers, curves, confusion matrices and misclassified images: [notebooks/results.ipynb](notebooks/results.ipynb).



## Quick start

Requires Python 3.12. Tested on an Apple M1 (MPS backend); CUDA and CPU also work.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                                              # unit tests, no download needed
python -m cifar10.train --model baseline --run-name baseline  # downloads CIFAR-10 to data/ on first run
python -m cifar10.train --model resnet   --run-name resnet
python -m cifar10.evaluate --run-dir outputs/baseline
python -m cifar10.evaluate --run-dir outputs/resnet
python -m cifar10.compare outputs/baseline outputs/resnet
python -m cifar10.export_onnx --run-dir outputs/resnet        # optional
jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

`scripts/train_all.sh` runs the training, evaluation and comparison steps in order for both models. It takes an optional seed (`scripts/train_all.sh 43`), which is how the second pair of runs in the results table was produced.

Every field of `TrainConfig` ([cifar10/config.py](cifar10/config.py)) is also a command-line flag, for example `--epochs 10 --max-lr 0.05 --device cpu`. Add `--resume` to continue an interrupted run from its `last.pt`. On CUDA or CPU, `--amp` turns on mixed precision.

## Project structure

```
cifar10/
  config.py        TrainConfig dataclass, saved as config.json with each run
  data.py          transforms, train/val split, DataLoaders
  models.py        BaselineCNN, ResidualBlock, ResidualCNN
  engine.py        train_one_epoch, evaluate, predict, fit (loop, LR schedule, checkpoints)
  checkpoint.py    save/load training state
  metrics.py       accuracy, confusion matrix, per-class accuracy
  visualize.py     curves, image grids, confusion matrix plots
  train.py         CLI: train one model
  evaluate.py      CLI: test accuracy, confusion matrix, misclassified images
  compare.py       CLI: baseline vs residual table and curves
  export_onnx.py   CLI: ONNX export and ONNX Runtime check
scripts/
  train_all.sh     train, evaluate and compare both models in one go
tests/             pytest suite, one file per module
notebooks/
  results.ipynb    curves, test accuracy, misclassified images, comparison, ONNX inference
docs/
  SPEC.md          requirements and success criteria
  decisions/       architecture decision records (ADR-001 to ADR-003)
outputs/<run>/     config.json, history.json, test_metrics.json, figures
                   (last.pt, best.pt and model.onnx are written here but git-ignored)
```

## Data pipeline

[cifar10/data.py](cifar10/data.py)

- `torchvision.datasets.CIFAR10` loads the official 50,000 / 10,000 train/test split.
- A seeded permutation holds out 5,000 training images for validation. The test set is only read by `cifar10.evaluate` and the notebook.
- Preprocessing uses the torchvision v2 API: `ToImage`, then (training only) a 4-pixel reflect-padded `RandomCrop` and `RandomHorizontalFlip`, then `ToDtype(float32, scale=True)` and `Normalize` with the CIFAR-10 training-set mean and standard deviation. Validation images come from a second dataset object with the eval transform, so they are never augmented.
- `DataLoader` runs 4 worker processes with persistent workers, batches of 128 for training and 256 for evaluation, and pinned memory on CUDA.

## Model design

[cifar10/models.py](cifar10/models.py)

**BaselineCNN (288,746 parameters)**

```
input 3x32x32
stage 1: [conv3x3 -> BatchNorm -> ReLU] x2, 32 channels     32x32
maxpool 2x2
stage 2: [conv3x3 -> BatchNorm -> ReLU] x2, 64 channels     16x16
maxpool 2x2
stage 3: [conv3x3 -> BatchNorm -> ReLU] x2, 128 channels    8x8
head:    global average pool -> Dropout(0.3) -> Linear(128, 10)
```

Design choices:

- Two 3x3 convolutions per stage give a 5x5 receptive field for the cost of 18 weights per channel pair, compared with 25 for one 5x5 convolution.
- BatchNorm after every convolution allows a learning rate of 0.1. The convolutions have no bias because BatchNorm's shift replaces it.
- Global average pooling replaces a large flatten + fully connected layer. The head has 1,290 parameters instead of about 82,000 for `Linear(128*8*8, 10)`, which leaves less to overfit.
- Dropout sits in the head, where it regularizes the one layer that sees all features. BatchNorm and data augmentation regularize the convolutional stages.

**ResidualCNN (299,530 parameters)**

The same network with each stage's two convolutions wrapped in a `ResidualBlock`: conv-BN-ReLU-conv-BN, plus a shortcut, then ReLU. The shortcut is the identity when shapes match and a 1x1 conv + BatchNorm projection when the channel count changes (all three stages here). The projections account for all 10,784 extra parameters. [ADR-002](docs/decisions/ADR-002-residual-block.md) explains why this option was chosen over B and C, and why the network was not made deeper.

## Training setup

[cifar10/engine.py](cifar10/engine.py), details in [ADR-003](docs/decisions/ADR-003-training-protocol.md)

| Setting | Value |
|---|---|
| Loss | cross-entropy |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Weight decay | 5e-4 on conv and linear weights, none on BatchNorm and biases |
| LR schedule | OneCycleLR, max LR 0.1, stepped every batch |
| Epochs / batch size | 30 / 128 |
| Dropout | 0.3 |
| Validation | 5,000 held-out training images |
| Seed | 42 |

After every epoch `fit` logs train and validation loss and accuracy, the learning rate and the epoch time. It writes `history.json`, overwrites `last.pt` (model, optimizer, scheduler, epoch, history, config) and writes `best.pt` whenever validation accuracy improves. Evaluation always restores `best.pt`.

## Reproducibility

- `seed_everything` seeds Python, NumPy and PyTorch and turns off cuDNN autotuning ([cifar10/utils.py](cifar10/utils.py)).
- The validation split, the shuffling generator and each DataLoader worker derive their seeds from the same value, following the [PyTorch randomness notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- `requirements.txt` pins every direct dependency to the version used for the results above.
- Each run saves its full config next to its metrics.
- The results table uses two seeds per model, because a single run cannot separate an improvement from seed noise.

Checked on this machine: two one-epoch runs with seed 42 on the MPS backend produced identical train and validation loss and accuracy (1.3938 / 0.4854 and 0.9397 / 0.6710).

PyTorch does not guarantee bit-identical results across devices or releases, and some MPS kernels are not deterministic, so treat that as evidence rather than a guarantee. Note also that the learning-rate schedule spans the whole run, so a one-epoch run and the first epoch of a 30-epoch run see different learning rates and end at different accuracies.

## Optional extensions

- **Unit tests:** `python -m pytest` runs 55 tests in about 20 seconds without downloading CIFAR-10. They cover transforms, the split, loaders, model shapes and components, the residual shortcut, metrics, the training step, the scheduler stepping, checkpoint round-trips, resuming after a crash, plotting and ONNX parity.
- **ONNX:** `cifar10.export_onnx` exports with the `torch.export`-based exporter and a dynamic batch dimension, then compares ONNX Runtime logits with PyTorch on test images.
- **Mixed precision:** `--amp` runs the forward pass under `torch.autocast` (float16 with `GradScaler` on CUDA, bfloat16 on CPU). Checked by running a full epoch with `--device cpu --amp` (63.18% validation accuracy) and by four unit tests in `tests/test_amp.py`. PyTorch's AMP docs do not list MPS, so the flag falls back to float32 there and the reported M1 runs are float32. The CUDA path is written to the documented API but was not run, since this machine has no CUDA GPU.

Distributed training and model serving are out of scope.

## Documentation

- [docs/SPEC.md](docs/SPEC.md): requirements, assumptions and success criteria
- [ADR-001](docs/decisions/ADR-001-pytorch.md): PyTorch over TensorFlow
- [ADR-002](docs/decisions/ADR-002-residual-block.md): residual block as the Part 2 improvement
- [ADR-003](docs/decisions/ADR-003-training-protocol.md): splits, optimizer, schedule and budget
