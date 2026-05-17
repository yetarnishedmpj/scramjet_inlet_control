from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml
from scramjet_rl.rl.evaluate import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rollout-csv", default="outputs/policy_rollouts.csv")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config["policy_path"] = args.policy_path
    metrics = evaluate_policy(config, episodes=args.episodes, output_path=args.rollout_csv)
    for name, value in metrics.items():
        print(f"{name}: {value:.6g}")


if __name__ == "__main__":
    main()
