from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env

from scramjet_rl.config import ensure_parent
from scramjet_rl.envs.scramjet_inlet_env import EnvConfig, ScramjetInletEnv


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
    )


def train_agent(config: dict) -> Path:
    env = ScramjetInletEnv(config["surrogate_path"], _env_config(config.get("env", {})))
    check_env(env, warn=True)
    algorithm = str(config.get("algorithm", "sac")).lower()
    if algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=int(config.get("seed", 0)),
            learning_starts=100,
            batch_size=64,
        )
    elif algorithm == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, seed=int(config.get("seed", 0)))
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Expected 'sac' or 'ppo'.")
    model.learn(total_timesteps=int(config.get("total_timesteps", 2000)))
    output_path = ensure_parent(config["output_path"])
    model.save(output_path)
    return output_path


def train_sac(config: dict) -> Path:
    config = dict(config)
    config["algorithm"] = "sac"
    return train_agent(config)
