from __future__ import annotations

import numpy as np


def shock_location_from_pressure(pressure: np.ndarray) -> tuple[float, float]:
    if pressure.ndim != 2:
        raise ValueError(f"pressure must be 2D, got shape {pressure.shape}")
    grad_y, grad_x = np.gradient(pressure.astype(np.float32))
    magnitude = np.hypot(grad_x, grad_y)
    row, col = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    x_norm = float(col / max(pressure.shape[1] - 1, 1))
    y_norm = float(row / max(pressure.shape[0] - 1, 1))
    return x_norm, y_norm


def shock_on_lip_error(pressure: np.ndarray, lip_x: float = 0.75, lip_y: float = 0.5) -> float:
    shock_x, shock_y = shock_location_from_pressure(pressure)
    return float(np.hypot(shock_x - lip_x, shock_y - lip_y))
