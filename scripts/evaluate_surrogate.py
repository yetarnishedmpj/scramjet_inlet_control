from __future__ import annotations

import argparse

from scramjet_rl.surrogate.evaluate import evaluate_surrogate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/synthetic_inlet.h5")
    parser.add_argument("--checkpoint", default="models/surrogate_cnn.pt")
    parser.add_argument("--plot-output", default=None)
    args = parser.parse_args()
    metrics = evaluate_surrogate(args.dataset, args.checkpoint)
    for name, value in metrics.items():
        print(f"{name}: {value:.6g}")
    if args.plot_output is not None:
        from scramjet_rl.surrogate.evaluate import plot_prediction_sample

        print(f"Wrote {plot_prediction_sample(args.dataset, args.checkpoint, args.plot_output)}")


if __name__ == "__main__":
    main()
