from pathlib import Path

from scramjet_rl.data.schema import validate_hdf5_schema
from scramjet_rl.data.synthetic import SyntheticConfig, generate_synthetic_fields, write_hdf5


def test_hdf5_schema_validates(tmp_path: Path):
    path = tmp_path / "dataset.h5"
    arrays = generate_synthetic_fields(
        SyntheticConfig(
            num_samples=3,
            height=8,
            width=16,
            seed=2,
            mach_range=(4.0, 8.0),
            altitude_range_m=(10000.0, 30000.0),
            ramp_angle_range_deg=(4.0, 18.0),
        )
    )
    write_hdf5(path, arrays)
    shapes = validate_hdf5_schema(path)
    assert shapes["inputs"] == (3, 4)
    assert shapes["pressure"] == (3, 8, 16)
