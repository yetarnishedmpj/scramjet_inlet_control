from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scramjet_rl.data.synthetic import isa_density_kg_m3
from scramjet_rl.surrogate.predictor import EnsembleSurrogatePredictor, SurrogatePredictor


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 120
    sensor_height: int = 8
    sensor_width: int = 16
    initial_mach_range: tuple[float, float] = (4.0, 8.0)
    initial_altitude_range: tuple[float, float] = (10000.0, 30000.0)
    initial_ramp_angle_range: tuple[float, float] = (6.0, 14.0)
    max_delta_angle_deg: float = 0.5
    actuator_time_constant: float = 0.35
    max_angle_rate_deg_per_step: float = 0.75
    min_ramp_angle_deg: float = 4.0
    max_ramp_angle_deg: float = 18.0
    uncertainty_penalty_weight: float = 0.0
    reward_efficiency_weight: float = 10.0
    reward_pressure_recovery_weight: float = 2.0
    reward_unstart_penalty: float = 1000.0
    reward_movement_rate_weight: float = 0.25
    reward_movement_accel_weight: float = 0.1


class ScramjetInletEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate_path: str | list[str],
        config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        if isinstance(surrogate_path, list):
            self.predictor = EnsembleSurrogatePredictor(surrogate_path)
            self.uses_ensemble = True
        else:
            self.predictor = SurrogatePredictor(surrogate_path)
            self.uses_ensemble = False
        sensor_size = 2 * self.config.sensor_height * self.config.sensor_width
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5 + sensor_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.mach = 0.0
        self.altitude = 0.0
        self.ramp_angle = 0.0
        self.angle_rate = 0.0
        self.last_metrics = np.zeros(4, dtype=np.float32)
        self.last_fields = np.zeros((2, 32, 64), dtype=np.float32)
        self.last_uncertainty = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.mach = float(self.rng.uniform(*self.config.initial_mach_range))
        self.altitude = float(self.rng.uniform(*self.config.initial_altitude_range))
        self.ramp_angle = float(self.rng.uniform(*self.config.initial_ramp_angle_range))
        self.angle_rate = 0.0
        observation = self._observe()
        return observation, {}

    def step(self, action: np.ndarray):
        previous_rate = self.angle_rate
        commanded_delta = float(np.clip(action[0], -1.0, 1.0) * self.config.max_delta_angle_deg)
        self.angle_rate += self.config.actuator_time_constant * (commanded_delta - self.angle_rate)
        self.angle_rate = float(
            np.clip(
                self.angle_rate,
                -self.config.max_angle_rate_deg_per_step,
                self.config.max_angle_rate_deg_per_step,
            )
        )
        self.ramp_angle = float(
            np.clip(
                self.ramp_angle + self.angle_rate,
                self.config.min_ramp_angle_deg,
                self.config.max_ramp_angle_deg,
            )
        )

        self.mach = float(np.clip(self.mach + self.rng.normal(0.0, 0.01), 4.0, 8.0))
        self.altitude = float(
            np.clip(self.altitude + self.rng.normal(0.0, 15.0), 10000.0, 30000.0)
        )
        self.step_count += 1

        observation = self._observe()
        shock_error, pressure_recovery, unstart_probability, efficiency = self.last_metrics
        movement_penalty = (
            self.config.reward_movement_rate_weight * abs(self.angle_rate)
            + self.config.reward_movement_accel_weight * abs(self.angle_rate - previous_rate)
        )
        uncertainty_penalty = self.config.uncertainty_penalty_weight * self.last_uncertainty
        reward = (
            self.config.reward_efficiency_weight * efficiency
            + self.config.reward_pressure_recovery_weight * pressure_recovery
            - movement_penalty
            - uncertainty_penalty
        )
        terminated = bool(unstart_probability > 0.8)
        if terminated:
            reward -= self.config.reward_unstart_penalty
        truncated = self.step_count >= self.config.max_steps
        info = {
            "shock_error": float(shock_error),
            "pressure_recovery": float(pressure_recovery),
            "unstart_probability": float(unstart_probability),
            "efficiency": float(efficiency),
            "ramp_angle_deg": float(self.ramp_angle),
            "angle_rate_deg_per_step": float(self.angle_rate),
            "surrogate_uncertainty": float(self.last_uncertainty),
        }
        return observation, float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        density = float(isa_density_kg_m3(np.asarray([self.altitude], dtype=np.float32))[0])
        inputs = np.asarray([self.mach, self.altitude, density, self.ramp_angle], dtype=np.float32)
        prediction = self.predictor.predict(inputs)
        if self.uses_ensemble:
            self.last_fields = prediction["field_mean"][0].astype(np.float32)
            self.last_metrics = prediction["metric_mean"][0].astype(np.float32)
            self.last_uncertainty = float(np.mean(prediction["metric_std"][0]))
        else:
            fields, metrics = prediction
            self.last_fields = fields[0].astype(np.float32)
            self.last_metrics = metrics[0].astype(np.float32)
            self.last_uncertainty = 0.0
        sensors = self._downsample(self.last_fields).reshape(-1)
        state = np.asarray(
            [
                self.mach / 8.0,
                self.altitude / 30000.0,
                density,
                self.ramp_angle / 18.0,
                self.angle_rate / max(self.config.max_angle_rate_deg_per_step, 1e-6),
            ],
            dtype=np.float32,
        )
        return np.concatenate([state, sensors.astype(np.float32)])

    def _downsample(self, fields: np.ndarray) -> np.ndarray:
        _, height, width = fields.shape
        h_idx = np.linspace(0, height - 1, self.config.sensor_height).astype(int)
        w_idx = np.linspace(0, width - 1, self.config.sensor_width).astype(int)
        sampled = fields[:, h_idx][:, :, w_idx]
        scale = np.asarray([100000.0, 300.0], dtype=np.float32)[:, None, None]
        return sampled / scale
