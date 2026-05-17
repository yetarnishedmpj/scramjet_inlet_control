from __future__ import annotations

import csv
from pathlib import Path


def rollout_to_manifest(rollout_csv: str | Path, output_path: str | Path) -> Path:
    rollout_csv = Path(rollout_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with rollout_csv.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "mach", "altitude_m", "ramp_angle_deg", "episode", "step"],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            # Evaluation rollouts store normalized observation indirectly through env info. If full Mach and
            # altitude are not available, keep conservative replay defaults and preserve the ramp command.
            writer.writerow(
                {
                    "case_id": f"replay_{index:05d}",
                    "mach": row.get("mach", "6.0"),
                    "altitude_m": row.get("altitude_m", "20000.0"),
                    "ramp_angle_deg": row.get("ramp_angle_deg", "10.0"),
                    "episode": row.get("episode", "0"),
                    "step": row.get("step", str(index)),
                }
            )
    return output_path
