import torch

from cifar10.checkpoint import load_checkpoint, save_checkpoint
from cifar10.models import build_model


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    model = build_model("baseline")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(4, 3, 32, 32)).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model=model, optimizer=optimizer, epoch=3,
                    history={"val_acc": [0.1, 0.2]}, config={"model": "baseline"})
    fresh_model = build_model("baseline")
    fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1, momentum=0.9)
    state = load_checkpoint(path, fresh_model, optimizer=fresh_optimizer)

    assert state["epoch"] == 3
    assert state["history"] == {"val_acc": [0.1, 0.2]}
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh_model.state_dict()[key]), key
    assert fresh_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_save_checkpoint_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "ckpt.pt"

    save_checkpoint(path, model=build_model("baseline"), epoch=0)

    assert path.exists()
