from pathlib import Path

import numpy as np

from scramjet_rl.data.splits import create_splits, load_split_indices
from scramjet_rl.data.synthetic import SyntheticConfig, generate_synthetic_fields, write_hdf5


def test_create_splits_is_reproducible(tmp_path: Path):
    dataset = tmp_path / "dataset.h5"
    write_hdf5(
        dataset,
        generate_synthetic_fields(
            SyntheticConfig(
                num_samples=20,
                height=4,
                width=8,
                seed=1,
                mach_range=(4.0, 8.0),
                altitude_range_m=(10000.0, 30000.0),
                ramp_angle_range_deg=(4.0, 18.0),
            )
        ),
    )

    create_splits(dataset, tmp_path / "splits_a", seed=5)
    create_splits(dataset, tmp_path / "splits_b", seed=5)

    a = load_split_indices(tmp_path / "splits_a")
    b = load_split_indices(tmp_path / "splits_b")
    assert np.array_equal(a["train"], b["train"])
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 20
