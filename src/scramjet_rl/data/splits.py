from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def create_splits(
    dataset_path: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, Path]:
    with h5py.File(dataset_path, "r") as file:
        sample_count = int(file["inputs"].shape[0])
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    test_count = int(sample_count * test_fraction)
    val_count = int(sample_count * val_fraction)
    test_indices = indices[:test_count]
    val_indices = indices[test_count : test_count + val_count]
    train_indices = indices[test_count + val_count :]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_indices.npy",
        "val": output_dir / "val_indices.npy",
        "test": output_dir / "test_indices.npy",
    }
    np.save(paths["train"], train_indices)
    np.save(paths["val"], val_indices)
    np.save(paths["test"], test_indices)
    return paths


def load_split_indices(split_dir: str | Path) -> dict[str, np.ndarray]:
    split_dir = Path(split_dir)
    return {
        "train": np.load(split_dir / "train_indices.npy"),
        "val": np.load(split_dir / "val_indices.npy"),
        "test": np.load(split_dir / "test_indices.npy"),
    }
