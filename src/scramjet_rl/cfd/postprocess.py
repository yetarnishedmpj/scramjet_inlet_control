from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scramjet_rl.data.synthetic import isa_density_kg_m3, write_hdf5


def interpolate_points_to_grid(
    points: np.ndarray,
    values: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(float(x.min()), float(x.max()), width)
    yi = np.linspace(float(y.min()), float(y.max()), height)
    grid = np.empty((height, width), dtype=np.float32)

    # Inverse-distance weighting is slow but robust for small sampled fields and has no SciPy dependency.
    for row, yy in enumerate(yi):
        for col, xx in enumerate(xi):
            dist2 = (x - xx) ** 2 + (y - yy) ** 2
            nearest = int(np.argmin(dist2))
            if dist2[nearest] < 1e-12:
                grid[row, col] = values[nearest]
                continue
            weights = 1.0 / np.maximum(dist2, 1e-12)
            grid[row, col] = float(np.sum(weights * values) / np.sum(weights))
    return grid


def sampled_csv_to_hdf5(
    manifest_path: str | Path,
    sampled_dir: str | Path,
    output_path: str | Path,
    height: int = 32,
    width: int = 64,
) -> Path:
    rows = _read_manifest(manifest_path)
    sampled_dir = Path(sampled_dir)
    inputs = []
    pressure_fields = []
    temperature_fields = []
    metrics = []

    for row in rows:
        case_id = row["case_id"]
        sample_path = sampled_dir / f"{case_id}.csv"
        points, pressure, temperature = _read_sample_csv(sample_path)
        pressure_grid = interpolate_points_to_grid(points, pressure, height, width)
        temperature_grid = interpolate_points_to_grid(points, temperature, height, width)

        mach = float(row["mach"])
        altitude = float(row["altitude_m"])
        angle = float(row["ramp_angle_deg"])
        density = float(isa_density_kg_m3(np.asarray([altitude], dtype=np.float32))[0])
        inputs.append([mach, altitude, density, angle])
        pressure_fields.append(pressure_grid)
        temperature_fields.append(temperature_grid)
        metrics.append(_estimate_metrics(pressure_grid, mach, angle))

    arrays = {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "pressure": np.asarray(pressure_fields, dtype=np.float32),
        "temperature": np.asarray(temperature_fields, dtype=np.float32),
        "metrics": np.asarray(metrics, dtype=np.float32),
    }
    write_hdf5(output_path, arrays)
    return Path(output_path)


def _estimate_metrics(pressure: np.ndarray, mach: float, angle: float) -> list[float]:
    target_angle = 6.0 + 1.6 * (mach - 4.0)
    shock_error = abs(angle - target_angle) / 8.0
    outlet = float(np.mean(pressure[:, -4:]))
    inlet = float(np.mean(pressure[:, :4])) + 1e-6
    pressure_recovery = float(np.clip(outlet / inlet / (1.0 + 0.08 * mach**2), 0.0, 1.0))
    unstart_probability = float(1.0 / (1.0 + np.exp(-(shock_error - 0.65) * 10.0)))
    efficiency = float(np.clip(1.0 - shock_error - 0.6 * unstart_probability, 0.0, 1.0))
    return [shock_error, pressure_recovery, unstart_probability, efficiency]


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_sample_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    required = {"x", "y", "pressure", "temperature"}
    if not required.issubset(rows.dtype.names or ()):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    points = np.column_stack([rows["x"], rows["y"]]).astype(np.float32)
    return points, rows["pressure"].astype(np.float32), rows["temperature"].astype(np.float32)
