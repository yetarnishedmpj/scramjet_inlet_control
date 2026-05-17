from __future__ import annotations

import argparse

from scramjet_rl.cfd.manifest import write_sweep_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cfd/sweep_manifest.csv")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = write_sweep_manifest(args.output, args.count, args.seed)
    print(f"Wrote CFD sweep manifest to {path}")


if __name__ == "__main__":
    main()
