from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml, validate_config
from scramjet_rl.rl.train import train_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config, "rl")
    model_path = train_agent(config)
    print(f"Wrote SAC policy to {model_path}")


if __name__ == "__main__":
    main()
