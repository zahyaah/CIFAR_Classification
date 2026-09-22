import numpy as np
import onnxruntime as ort
import torch

from cifar10.export_onnx import export_to_onnx
from cifar10.models import build_model


def test_onnx_export_matches_pytorch_for_any_batch_size(tmp_path):
    torch.manual_seed(0)
    model = build_model("resnet").eval()
    path = tmp_path / "model.onnx"

    export_to_onnx(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for batch_size in (1, 5):
        images = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            expected = model(images).numpy()
        [actual] = session.run(None, {input_name: images.numpy()})
        np.testing.assert_allclose(actual, expected, atol=1e-4)
