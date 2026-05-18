from __future__ import annotations

import argparse

from scramjet_rl.data.splits import create_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="data/splits/default")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, path in create_splits(
        args.dataset,
        args.output_dir,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
