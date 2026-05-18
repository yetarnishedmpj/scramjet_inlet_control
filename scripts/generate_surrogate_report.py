from __future__ import annotations

import argparse

from scramjet_rl.surrogate.evaluate import generate_surrogate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/synthetic_inlet.h5")
    parser.add_argument("--checkpoint", default="models/surrogate_cnn.pt")
    parser.add_argument("--output-dir", default="outputs/reports/surrogate")
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    print(
        "Wrote surrogate report to "
        f"{generate_surrogate_report(args.dataset, args.checkpoint, args.output_dir, args.sample_count)}"
    )


if __name__ == "__main__":
    main()
