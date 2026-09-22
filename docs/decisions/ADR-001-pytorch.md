# ADR-001: Use PyTorch

## Status
Accepted

## Date
2026-09-22

## Context
The brief allows TensorFlow or PyTorch. Training runs on an Apple M1 laptop, so the framework needs a working Apple GPU backend. The brief also asks for an explicit training loop, a custom residual block and, optionally, ONNX export.

## Decision
Use PyTorch 2.14 with torchvision 0.29. Train on the MPS backend when available, CUDA or CPU otherwise.

## Alternatives Considered

### TensorFlow / Keras
- Pros: `tf.data` pipeline, built-in callbacks, TensorBoard.
- Cons: Apple GPU support comes through the separate `tensorflow-metal` plugin, which lags TensorFlow releases. `model.fit` hides the training loop the brief asks to show.
- Rejected: the training loop is a graded requirement and PyTorch makes it explicit by default.

## Consequences
- The training loop in `cifar10/engine.py` is plain Python that a reviewer can read top to bottom.
- `torch.onnx.export` covers the ONNX extension without another converter.
- Automatic mixed precision is documented for CUDA and CPU only, so the `--amp` flag has no effect on MPS.
