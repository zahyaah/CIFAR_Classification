"""Compare finished runs.

Usage: python -m cifar10.compare outputs/baseline outputs/resnet
"""
import argparse
import json
from pathlib import Path

from cifar10.visualize import plot_comparison


def comparison_rows(run_dirs: list[str | Path]) -> list[dict]:
    """One summary dict per run, read from config.json, history.json and test_metrics.json."""
    rows = []
    for run_dir in map(Path, run_dirs):
        config = json.loads((run_dir / "config.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        test = json.loads((run_dir / "test_metrics.json").read_text())
        rows.append({
            "run": run_dir.name,
            "model": config["model"],
            "parameters": test["num_parameters"],
            "epochs": len(history["val_acc"]),
            "best_val_acc": max(history["val_acc"]),
            "test_acc": test["test_acc"],
            "test_loss": test["test_loss"],
            "sec_per_epoch": sum(history["epoch_time"]) / len(history["epoch_time"]),
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Run | Model | Parameters | Epochs | Best val acc | Test acc | Test loss | s/epoch |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['model']} | {r['parameters']:,} | {r['epochs']} | "
                     f"{r['best_val_acc']:.2%} | {r['test_acc']:.2%} | {r['test_loss']:.4f} | "
                     f"{r['sec_per_epoch']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare finished runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()

    table = to_markdown(comparison_rows(args.run_dirs))
    histories = {d.name: json.loads((d / "history.json").read_text()) for d in args.run_dirs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.md").write_text(table)
    plot_comparison(histories, args.out_dir / "validation_curves.png")
    print(table)


if __name__ == "__main__":
    main()
