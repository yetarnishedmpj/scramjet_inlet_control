from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except ModuleNotFoundError:
        data = _load_simple_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{context} config missing required keys: {', '.join(missing)}")


def validate_config(config: dict[str, Any], kind: str) -> None:
    if kind == "data":
        require_keys(config, ["output_path", "num_samples", "height", "width", "ranges"], kind)
        ranges = config["ranges"]
        require_keys(ranges, ["mach", "altitude_m", "ramp_angle_deg"], "data.ranges")
        _validate_positive_int(config, "num_samples")
        _validate_positive_int(config, "height")
        _validate_positive_int(config, "width")
    elif kind == "surrogate":
        require_keys(config, ["dataset_path", "model_path"], kind)
        model_type = str(config.get("model_type", "cnn"))
        if model_type not in {"cnn", "resnet", "unet", "metric_mlp"}:
            raise ValueError("surrogate model_type must be one of: cnn, resnet, unet, metric_mlp")
    elif kind == "rl":
        require_keys(config, ["surrogate_path", "output_path"], kind)
        algorithm = str(config.get("algorithm", "sac")).lower()
        if algorithm not in {"sac", "ppo"}:
            raise ValueError("rl algorithm must be 'sac' or 'ppo'")
    else:
        raise ValueError(f"Unknown config kind {kind!r}")


def _validate_positive_int(config: dict[str, Any], key: str) -> None:
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small config subset used by this scaffold when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(marker in value.lower() for marker in [".", "e"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
