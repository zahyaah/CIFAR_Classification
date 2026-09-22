#!/bin/zsh
# Train both models with identical settings, then evaluate and compare them.
# Usage: scripts/train_all.sh [seed]   (a seed other than 42 appends -s<seed> to the run names)
# Safe to re-run: --resume continues each run from its last.pt.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEED=${1:-42}
SUFFIX=""
[[ "$SEED" != "42" ]] && SUFFIX="-s$SEED"

for model in baseline resnet; do
  run=$model$SUFFIX
  mkdir -p outputs/$run
  $PY -m cifar10.train --model $model --run-name $run --seed $SEED --resume >> outputs/$run/train.log 2>&1 \
    || { echo "training $run failed, see outputs/$run/train.log"; exit 1; }
  $PY -m cifar10.evaluate --run-dir outputs/$run >> outputs/$run/evaluate.log 2>&1 \
    || { echo "evaluating $run failed"; exit 1; }
done
$PY -m cifar10.compare outputs/baseline$SUFFIX outputs/resnet$SUFFIX > outputs/compare$SUFFIX.log 2>&1
echo "done" >> outputs/compare$SUFFIX.log
