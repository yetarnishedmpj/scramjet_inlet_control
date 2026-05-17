from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

REQUIRED_DATASETS = {
    "inputs": (2, 4),
    "pressure": (3, None),
    "temperature": (3, None),
    "metrics": (2, 4),
}


def validate_hdf5_schema(path: str | Path) -> dict[str, tuple[int, ...]]:
    path = Path(path)
    shapes: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as file:
        missing = sorted(set(REQUIRED_DATASETS) - set(file.keys()))
        if missing:
            raise ValueError(f"Missing datasets in {path}: {', '.join(missing)}")

        sample_count = int(file["inputs"].shape[0])
        field_shape = tuple(file["pressure"].shape)
        for name, (rank, second_dim) in REQUIRED_DATASETS.items():
            shape = tuple(file[name].shape)
            shapes[name] = shape
            if len(shape) != rank:
                raise ValueError(f"{name} must have rank {rank}, got shape {shape}")
            if shape[0] != sample_count:
                raise ValueError(f"{name} sample count mismatch: expected {sample_count}, got {shape[0]}")
            if second_dim is not None and shape[1] != second_dim:
                raise ValueError(f"{name} second dimension must be {second_dim}, got {shape[1]}")

        if tuple(file["temperature"].shape) != field_shape:
            raise ValueError("pressure and temperature fields must have identical shapes")

        for name in REQUIRED_DATASETS:
            values = file[name][:]
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")

    return shapes
