from __future__ import annotations

from pathlib import Path

import numpy as np

from scramjet_rl.data.schema import validate_hdf5_schema
from scramjet_rl.data.synthetic import write_hdf5


def import_arrays(
    inputs_path: str | Path,
    pressure_path: str | Path,
    temperature_path: str | Path,
    metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    arrays = {
        "inputs": _load_array(inputs_path, expected_rank=2).astype(np.float32),
        "pressure": _load_array(pressure_path, expected_rank=3).astype(np.float32),
        "temperature": _load_array(temperature_path, expected_rank=3).astype(np.float32),
        "metrics": _load_array(metrics_path, expected_rank=2).astype(np.float32),
    }
    write_hdf5(output_path, arrays)
    validate_hdf5_schema(output_path)
    return Path(output_path)


def _load_array(path: str | Path, expected_rank: int) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path} must contain exactly one array, found {data.files}")
        values = data[data.files[0]]
    elif path.suffix.lower() in {".csv", ".txt"}:
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported array format for {path}. Use .npy, .npz, .csv, or .txt.")
    if values.ndim != expected_rank:
        raise ValueError(f"{path} must have rank {expected_rank}, got shape {values.shape}")
    return values
