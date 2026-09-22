"""Training configuration. One dataclass, saved as config.json with every run."""
from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    model: str = "baseline"
    epochs: int = 30
    batch_size: int = 128
    max_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    dropout: float = 0.3
    label_smoothing: float = 0.0
    val_size: int = 5000
    seed: int = 42
    num_workers: int = 4
    amp: bool = False
    device: str = "auto"
    data_dir: str = "data"
    output_dir: str = "outputs"
    run_name: str | None = None

    @property
    def run_dir_name(self) -> str:
        return self.run_name or self.model

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})
