# ADR-002: Part 2 improvement is a residual block (Option A)

## Status
Accepted

## Date
2026-09-22

## Context
Part 2 asks for one improvement over the baseline and a comparison between the two. The comparison only means something if a single factor changes between runs.

## Decision
Implement a basic ResNet block (He et al., 2015, https://arxiv.org/abs/1512.03385): two 3x3 conv + BatchNorm layers, a shortcut added before the final ReLU, and a 1x1 conv + BatchNorm projection on the shortcut when the channel count or resolution changes.

`ResidualCNN` keeps the baseline layout (three stages of two 3x3 convolutions at 32, 64 and 128 channels, max-pooling between stages, same head) and wraps each stage's conv pair in a `ResidualBlock`. The two models differ by the shortcuts alone: three 1x1 projections and their BatchNorms, about 10.8k extra parameters on a base of about 288k. Both runs use the same seed, data split, optimizer, schedule and epoch count.

## Alternatives Considered

### Option B: callback or training utility (early stopping, gradient clipping, logger)
- Pros: cheap to add.
- Cons: with a fixed one-cycle schedule, early stopping mostly shortens training rather than improving the model, and gradient clipping rarely changes results for a BatchNorm CNN of this size.
- Rejected: small expected effect on the model itself.

### Option C: hyperparameter tuning
- Pros: can produce the largest accuracy gain.
- Cons: a useful search over learning rate, batch size, optimizer and dropout needs dozens of runs. At roughly a minute per epoch on an M1, that exceeds the time budget.
- Rejected: compute cost.

### Deeper residual network (for example ResNet-18)
- Pros: higher accuracy.
- Cons: changes depth, width and parameter count at the same time as adding shortcuts, so the comparison cannot attribute the gain to the residual connection. It also stops being a "small CNN".
- Rejected: confounds the comparison.

## Consequences
- The accuracy gap between the runs measures the effect of shortcuts on a shallow network. Shortcuts matter most for deep networks, so expect a modest gap here.
- Results come from one seed per model. A gap smaller than about 0.3 percentage points is within typical seed-to-seed noise for CIFAR-10 and should not be read as a real difference.

## Outcome (recorded 2026-09-22, after the runs)

Test accuracy, 30 epochs, two seeds per model:

| Model | Seed 42 | Seed 43 | Mean |
|---|---:|---:|---:|
| Baseline | 89.88% | 89.91% | 89.90% |
| Residual | 89.85% | 89.98% | 89.92% |

The predicted "modest gap" turned out to be no gap. The 0.02-point mean difference is smaller than the seed-to-seed spread of either model (0.03 and 0.13 points), so this experiment cannot claim an accuracy gain for the shortcuts.

The one effect that held across both seeds is faster early training: mean validation accuracy over epochs 1 to 15 was 73.6% against 72.4% (seed 42) and 74.1% against 72.2% (seed 43).

The decision stands. The comparison answers the question it was designed to answer, and a null result on a six-layer network is the expected result for a technique built to make 50-layer networks trainable. Anyone extending this work should add depth, not more shortcuts at this depth, and should report more than two seeds.
