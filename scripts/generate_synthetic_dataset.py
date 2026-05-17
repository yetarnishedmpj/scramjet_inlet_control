from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml, validate_config
from scramjet_rl.data.synthetic import SyntheticConfig, generate_synthetic_fields, write_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    validate_config(raw, "data")
    ranges = raw["ranges"]
    config = SyntheticConfig(
        num_samples=int(raw["num_samples"]),
        height=int(raw["height"]),
        width=int(raw["width"]),
        seed=int(raw.get("seed", 0)),
        mach_range=tuple(ranges["mach"]),
        altitude_range_m=tuple(ranges["altitude_m"]),
        ramp_angle_range_deg=tuple(ranges["ramp_angle_deg"]),
    )
    arrays = generate_synthetic_fields(config)
    write_hdf5(raw["output_path"], arrays)
    print(f"Wrote {raw['output_path']} with {config.num_samples} samples")


if __name__ == "__main__":
    main()
