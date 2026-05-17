from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml, validate_config
from scramjet_rl.surrogate.train import train_surrogate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config, "surrogate")
    model_path = train_surrogate(config)
    print(f"Wrote surrogate checkpoint to {model_path}")


if __name__ == "__main__":
    main()
