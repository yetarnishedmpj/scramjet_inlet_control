from __future__ import annotations

import argparse

from scramjet_rl.cfd.postprocess import sampled_csv_to_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="cfd/sweep_manifest.csv")
    parser.add_argument("--sampled-dir", default="data/raw_openfoam/sampled")
    parser.add_argument("--output", default="data/processed/openfoam_inlet.h5")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    path = sampled_csv_to_hdf5(args.manifest, args.sampled_dir, args.output, args.height, args.width)
    print(f"Wrote processed CFD dataset to {path}")


if __name__ == "__main__":
    main()
