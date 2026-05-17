from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_sweep_manifest(path: str | Path, count: int = 128, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "mach", "altitude_m", "ramp_angle_deg"],
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "case_id": f"case_{index:05d}",
                    "mach": f"{rng.uniform(4.0, 8.0):.5f}",
                    "altitude_m": f"{rng.uniform(10000.0, 30000.0):.2f}",
                    "ramp_angle_deg": f"{rng.uniform(4.0, 18.0):.5f}",
                }
            )
    return path
