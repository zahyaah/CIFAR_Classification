import json

from cifar10.compare import comparison_rows, to_markdown


def write_run(run_dir, model, val_acc, test_acc, params):
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": model, "epochs": len(val_acc)}))
    (run_dir / "history.json").write_text(json.dumps({"val_acc": val_acc, "epoch_time": [10.0] * len(val_acc)}))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_acc": test_acc, "test_loss": 0.5,
                                                           "num_parameters": params}))


def test_comparison_rows_read_run_outputs(tmp_path):
    write_run(tmp_path / "a", "baseline", [0.5, 0.8, 0.7], 0.79, 1000)

    [row] = comparison_rows([tmp_path / "a"])

    assert row == {"run": "a", "model": "baseline", "parameters": 1000, "epochs": 3,
                   "best_val_acc": 0.8, "test_acc": 0.79, "test_loss": 0.5, "sec_per_epoch": 10.0}


def test_to_markdown_renders_one_line_per_run(tmp_path):
    write_run(tmp_path / "a", "baseline", [0.8], 0.79, 1000)
    write_run(tmp_path / "b", "resnet", [0.85], 0.84, 1100)

    table = to_markdown(comparison_rows([tmp_path / "a", tmp_path / "b"]))

    lines = table.strip().splitlines()
    assert len(lines) == 4
    assert "84.00%" in lines[3]
