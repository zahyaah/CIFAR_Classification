# ADR-003: Training and evaluation protocol

## Status
Accepted

## Date
2026-09-22

## Context
Both models must be trained and compared on a laptop GPU within a few hours, with a result a reviewer can trust. Choosing checkpoints or settings on the test set would inflate the reported accuracy.

## Decision
- **Splits:** hold out 5,000 of the 50,000 training images as a validation set, chosen by a seeded permutation. The 10,000 test images are only used by `cifar10.evaluate`, once per finished run.
- **Checkpoint selection:** `best.pt` is the epoch with the highest validation accuracy. `last.pt` holds the full training state for `--resume`.
- **Preprocessing:** per-channel normalization with CIFAR-10 training-set statistics. Training images get a 4-pixel reflect-padded random crop and a horizontal flip. Validation and test images are only normalized.
- **Optimizer:** SGD, momentum 0.9, Nesterov, weight decay 5e-4 on conv and linear weights only (none on BatchNorm and bias parameters).
- **Schedule:** `OneCycleLR` with `max_lr=0.1`, stepped every batch, PyTorch defaults otherwise (30% warm-up, cosine annealing).
- **Budget:** 30 epochs, batch size 128, dropout 0.3 before the linear layer.
- **Reproducibility:** one seed (42) drives Python, NumPy and PyTorch RNGs, the split, the shuffling generator and per-worker seeds.

## Alternatives Considered

### Report the best test accuracy seen during training
- Rejected: that selects a checkpoint on the test set.

### Adam or AdamW
- Pros: less sensitive to the learning rate.
- Cons: SGD with momentum and a one-cycle schedule is the common, well-tested recipe for small CIFAR-10 CNNs and generalizes at least as well.
- Rejected: no reason to depart from the standard recipe for a baseline.

### Step decay or cosine schedule per epoch
- Rejected: one-cycle reaches good accuracy in fewer epochs, which matters on a laptop.

## Consequences
- Validation accuracy drives every choice. Test accuracy is reported, never optimized.
- Changing `--epochs` changes the schedule shape, so `--resume` refuses a checkpoint trained with a different epoch count.
- MPS kernels are not guaranteed to be bit-exact between runs. Two runs with the same seed get the same split, order and augmentation, and their metrics may still differ in the last decimal places.
