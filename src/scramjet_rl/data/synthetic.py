from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    num_samples: int
    height: int
    width: int
    seed: int
    mach_range: tuple[float, float]
    altitude_range_m: tuple[float, float]
    ramp_angle_range_deg: tuple[float, float]


def isa_density_kg_m3(altitude_m: np.ndarray) -> np.ndarray:
    rho0 = 1.225
    scale_height = 8500.0
    return rho0 * np.exp(-altitude_m / scale_height)


def generate_synthetic_fields(config: SyntheticConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    mach = rng.uniform(*config.mach_range, size=config.num_samples).astype(np.float32)
    altitude = rng.uniform(*config.altitude_range_m, size=config.num_samples).astype(np.float32)
    angle = rng.uniform(*config.ramp_angle_range_deg, size=config.num_samples).astype(np.float32)
    density = isa_density_kg_m3(altitude).astype(np.float32)

    x = np.linspace(0.0, 1.0, config.width, dtype=np.float32)[None, None, :]
    y = np.linspace(0.0, 1.0, config.height, dtype=np.float32)[None, :, None]

    pressure = np.empty((config.num_samples, config.height, config.width), dtype=np.float32)
    temperature = np.empty_like(pressure)
    metrics = np.empty((config.num_samples, 4), dtype=np.float32)

    for i in range(config.num_samples):
        m = float(mach[i])
        a = float(angle[i])
        rho = float(density[i])

        target_angle = 6.0 + 1.6 * (m - 4.0)
        shock_error = (a - target_angle) / 8.0
        shock_x = np.clip(0.72 - 0.045 * (a - target_angle), 0.15, 0.95)
        shock_slope = 0.42 + 0.035 * (m - 4.0)
        shock_line = shock_x - shock_slope * (y - 0.5)
        shock = 1.0 / (1.0 + np.exp(-(x - shock_line) * 90.0))

        compression = 1.0 + 0.11 * m**2 * shock
        boundary_layer = 0.08 * np.exp(-((y - 0.0) / 0.16) ** 2)
        ramp_effect = 0.03 * a * np.maximum(0.0, 1.0 - x)

        base_pressure = 101325.0 * rho / 1.225
        base_temperature = 220.0 + 8.0 * (m - 4.0)

        pressure[i] = base_pressure * (compression + boundary_layer + ramp_effect)
        temperature[i] = base_temperature * (1.0 + 0.12 * shock + 0.015 * a * y)

        unstart_probability = float(1.0 / (1.0 + np.exp(-(abs(shock_error) - 0.65) * 10.0)))
        pressure_recovery = float(np.clip(0.93 - 0.16 * abs(shock_error) - 0.02 * (m - 4.0), 0.1, 1.0))
        efficiency = float(np.clip(1.0 - abs(shock_error) - 0.6 * unstart_probability, 0.0, 1.0))
        metrics[i] = [abs(shock_error), pressure_recovery, unstart_probability, efficiency]

    inputs = np.column_stack([mach, altitude, density, angle]).astype(np.float32)
    return {
        "inputs": inputs,
        "pressure": pressure,
        "temperature": temperature,
        "metrics": metrics,
    }


def write_hdf5(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for key, value in arrays.items():
            file.create_dataset(key, data=value, compression="gzip")
