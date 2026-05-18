from pathlib import Path

import numpy as np

from scramjet_rl.data.importers import import_arrays
from scramjet_rl.data.schema import validate_hdf5_schema


def test_import_arrays_from_npy(tmp_path: Path):
    inputs = np.zeros((2, 4), dtype=np.float32)
    pressure = np.zeros((2, 4, 5), dtype=np.float32)
    temperature = np.ones((2, 4, 5), dtype=np.float32)
    metrics = np.zeros((2, 4), dtype=np.float32)
    np.save(tmp_path / "inputs.npy", inputs)
    np.save(tmp_path / "pressure.npy", pressure)
    np.save(tmp_path / "temperature.npy", temperature)
    np.save(tmp_path / "metrics.npy", metrics)

    output = import_arrays(
        tmp_path / "inputs.npy",
        tmp_path / "pressure.npy",
        tmp_path / "temperature.npy",
        tmp_path / "metrics.npy",
        tmp_path / "dataset.h5",
    )

    assert validate_hdf5_schema(output)["pressure"] == (2, 4, 5)
