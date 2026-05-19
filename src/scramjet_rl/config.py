from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


class DataRanges(BaseModel):
    mach: tuple[float, float]
    altitude_m: tuple[float, float]
    ramp_angle_deg: tuple[float, float]


class DataConfigSchema(BaseModel):
    output_path: str
    num_samples: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    ranges: DataRanges
    seed: int = 0


class SurrogateConfigSchema(BaseModel):
    dataset_path: str
    model_path: str
    model_type: Literal["cnn", "resnet", "unet", "metric_mlp"] = "cnn"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_fraction: float = 0.15
    seed: int = 0
    field_loss_weight: float = 1.0
    metric_loss_weight: float = 0.1
    log_dir: str = "outputs/experiments"


class RLConfigSchema(BaseModel):
    surrogate_path: str | list[str] | None = None
    surrogate_paths: str | list[str] | None = None
    output_path: str
    algorithm: Literal["sac", "ppo"] = "sac"


def validate_config(config: dict[str, Any], kind: str) -> None:
    if kind == "data":
        DataConfigSchema(**config)
    elif kind == "surrogate":
        SurrogateConfigSchema(**config)
    elif kind == "rl":
        RLConfigSchema(**config)
    else:
        raise ValueError(f"Unknown config kind {kind!r}")
