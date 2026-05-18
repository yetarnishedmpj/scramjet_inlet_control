from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml
from scramjet_rl.rl.baseline import evaluate_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rollout-csv", default="outputs/baseline_rollouts.csv")
    args = parser.parse_args()
    for name, value in evaluate_baseline(load_yaml(args.config), args.episodes, args.rollout_csv).items():
        print(f"{name}: {value:.6g}")


if __name__ == "__main__":
    main()
