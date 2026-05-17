from __future__ import annotations

import argparse
from pathlib import Path

from scramjet_rl.plots import plot_dataset_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/synthetic_inlet.h5")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/sample_fields.png")
    args = parser.parse_args()

    print(f"Wrote {plot_dataset_sample(args.dataset, args.index, args.output)}")


if __name__ == "__main__":
    main()
