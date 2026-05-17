from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from scramjet_rl.envs.scramjet_inlet_env import EnvConfig, ScramjetInletEnv
from scramjet_rl.rl.train import _env_config


def evaluate_policy(config: dict, episodes: int = 5, output_path: str | Path | None = None) -> dict[str, float]:
    env_config = _env_config(config.get("env", {}))
    env = ScramjetInletEnv(config["surrogate_path"], env_config)
    algorithm = str(config.get("algorithm", "sac")).lower()
    model_cls = SAC if algorithm == "sac" else PPO
    model = model_cls.load(config["policy_path"])

    rows = []
    episode_rewards = []
    unstarts = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=int(config.get("seed", 0)) + episode)
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            rows.append({"episode": episode, "step": step, "reward": reward, **info})
            rows[-1]["mach"] = env.mach
            rows[-1]["altitude_m"] = env.altitude
            step += 1
        episode_rewards.append(total_reward)
        if rows and rows[-1]["unstart_probability"] > 0.8:
            unstarts += 1

    if output_path is not None:
        _write_rows(output_path, rows)

    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
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
