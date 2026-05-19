from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from scramjet_rl.config import ensure_parent
from scramjet_rl.envs.scramjet_inlet_env import EnvConfig, ScramjetInletEnv
from scramjet_rl.logging import append_csv, timestamp


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = log_path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            latest_ep = self.model.ep_info_buffer[-1]
            append_csv(
                self.log_path,
                {
                    "timesteps": self.num_timesteps,
                    "reward": latest_ep["r"],
                    "length": latest_ep["l"],
                },
            )


def _env_config(raw: dict) -> EnvConfig:
    return EnvConfig(
        max_steps=int(raw.get("max_steps", 120)),
        sensor_height=int(raw.get("sensor_height", 8)),
        sensor_width=int(raw.get("sensor_width", 16)),
        initial_mach_range=tuple(raw.get("initial_mach_range", [4.0, 8.0])),
        initial_altitude_range=tuple(raw.get("initial_altitude_range", [10000.0, 30000.0])),
        initial_ramp_angle_range=tuple(raw.get("initial_ramp_angle_range", [6.0, 14.0])),
        max_delta_angle_deg=float(raw.get("max_delta_angle_deg", 0.5)),
        actuator_time_constant=float(raw.get("actuator_time_constant", 0.35)),
        max_angle_rate_deg_per_step=float(raw.get("max_angle_rate_deg_per_step", 0.75)),
        min_ramp_angle_deg=float(raw.get("min_ramp_angle_deg", 4.0)),
        max_ramp_angle_deg=float(raw.get("max_ramp_angle_deg", 18.0)),
        uncertainty_penalty_weight=float(raw.get("uncertainty_penalty_weight", 0.0)),
        reward_efficiency_weight=float(raw.get("reward_efficiency_weight", 10.0)),
        reward_pressure_recovery_weight=float(raw.get("reward_pressure_recovery_weight", 2.0)),
        reward_unstart_penalty=float(raw.get("reward_unstart_penalty", 1000.0)),
        reward_movement_rate_weight=float(raw.get("reward_movement_rate_weight", 0.25)),
        reward_movement_accel_weight=float(raw.get("reward_movement_accel_weight", 0.1)),
    )


def train_agent(config: dict) -> Path:
    surrogate_path = config.get("surrogate_paths", config["surrogate_path"])
    env = ScramjetInletEnv(surrogate_path, _env_config(config.get("env", {})))
    check_env(env, warn=True)

    algorithm = str(config.get("algorithm", "sac")).lower()
    seed = int(config.get("seed", 0))

    run_id = str(config.get("run_id", timestamp()))
    log_dir = Path(str(config.get("log_dir", "outputs/experiments"))) / run_id
    history_path = log_dir / "rl_history.csv"
    callback = RewardLoggerCallback(history_path)

    if algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=int(config.get("learning_starts", 100)),
            batch_size=int(config.get("batch_size", 64)),
            buffer_size=int(config.get("buffer_size", 100000)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            gamma=float(config.get("gamma", 0.99)),
            tau=float(config.get("tau", 0.005)),
            ent_coef=config.get("ent_coef", "auto"),
        )
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=int(config.get("n_steps", 2048)),
            batch_size=int(config.get("batch_size", 64)),
            n_epochs=int(config.get("n_epochs", 10)),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            gamma=float(config.get("gamma", 0.99)),
        )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Expected 'sac' or 'ppo'.")

    model.learn(total_timesteps=int(config.get("total_timesteps", 2000)), callback=callback)
    output_path = ensure_parent(config["output_path"])
    model.save(output_path)
    return output_path


def train_sac(config: dict) -> Path:
    config = dict(config)
    config["algorithm"] = "sac"
    return train_agent(config)
