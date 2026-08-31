from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/UrbanFloodBench"
    model_glob: str = "Model_*"
    warmup_steps: int = 10
    context_steps: int = 4
    horizon_steps: int = 12
    window_stride: int = 4
    cache_events: bool = True
    strict_schema: bool = True
    standardize: bool = True
    rainfall_known_future: bool = True
    split_seed: int = 20260831


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    processor_layers: int = 4
    dropout: float = 0.1
    token_count: int = 32
    token_heads: int = 4
    token_layers: int = 2
    use_global_tokens: bool = True
    use_coupling: bool = True
    use_hydraulic_head: bool = True
    use_flux_decoder: bool = True
    use_mass_correction: bool = False
    adapter_rank: int = 16
    max_history: int = 8


@dataclass
class LossConfig:
    water_level: float = 1.0
    water_volume: float = 0.25
    inlet_flow: float = 0.10
    edge_flow_1d: float = 0.15
    edge_flow_2d: float = 0.25
    edge_velocity_1d: float = 0.05
    edge_velocity_2d: float = 0.10
    mass_local: float = 0.20
    mass_global: float = 0.10
    dry_nonnegative: float = 0.02
    flood_threshold_ft: float = 0.164041995  # 0.05 m


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_events: int = 1
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.995
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.1
    curriculum_epochs: int = 50
    masked_state_prob: float = 0.25
    masked_sensor_prob: float = 0.50
    early_stopping_patience: int = 12
    checkpoint_every: int = 5
    device: str = "auto"
    num_workers: int = 0
    windows_per_event: int = 24
    val_events_per_epoch: int = 4
    seed: int = 20260831
    output_dir: str = "outputs/paper"


@dataclass
class EvaluationConfig:
    flood_thresholds_m: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.30, 0.50])
    arrival_threshold_m: float = 0.05
    event_balanced: bool = True
    save_predictions: bool = True
    sparse_sensor_fractions: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05])
    few_shot_events: list[int] = field(default_factory=lambda: [1, 3, 5])
    sensor_layout_seeds: list[int] = field(default_factory=lambda: [20260831, 20260832, 20260833])


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    protocol: str = "leave_one_city_out"
    target_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, payload: dict[str, Any] | None):
    payload = payload or {}
    valid = set(cls.__dataclass_fields__)
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {"data", "model", "loss", "train", "evaluation", "protocol", "target_model"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment config keys: {sorted(unknown)}")
    return ExperimentConfig(
        data=_merge_dataclass(DataConfig, payload.get("data")),
        model=_merge_dataclass(ModelConfig, payload.get("model")),
        loss=_merge_dataclass(LossConfig, payload.get("loss")),
        train=_merge_dataclass(TrainConfig, payload.get("train")),
        evaluation=_merge_dataclass(EvaluationConfig, payload.get("evaluation")),
        protocol=payload.get("protocol", "leave_one_city_out"),
        target_model=payload.get("target_model"),
    )
