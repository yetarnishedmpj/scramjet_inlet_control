from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scramjet_rl.envs.scramjet_inlet_env import EnvConfig, ScramjetInletEnv
from scramjet_rl.rl.train import _env_config


def target_angle_for_mach(mach: float) -> float:
    return float(np.clip(6.0 + 1.6 * (mach - 4.0), 4.0, 18.0))


def evaluate_baseline(config: dict, episodes: int = 5, output_path: str | Path | None = None) -> dict[str, float]:
    env = ScramjetInletEnv(config["surrogate_path"], _env_config(config.get("env", {})))
    rows = []
    rewards = []
    unstarts = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=int(config.get("seed", 0)) + episode)
        del observation
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            target = target_angle_for_mach(env.mach)
            error = target - env.ramp_angle
            action = np.asarray([np.clip(error / env.config.max_delta_angle_deg, -1.0, 1.0)], dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "mach": env.mach,
                    "altitude_m": env.altitude,
                    "target_ramp_angle_deg": target,
                    **info,
                }
            )
            step += 1
        rewards.append(total_reward)
        if rows and rows[-1]["unstart_probability"] > 0.8:
            unstarts += 1

    if output_path is not None:
        _write_rows(output_path, rows)
    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "unstart_rate": float(unstarts / max(episodes, 1)),
    }


def _write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
