from __future__ import annotations

import argparse

from scramjet_rl.data.schema import validate_hdf5_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    shapes = validate_hdf5_schema(args.path)
    for name, shape in shapes.items():
        print(f"{name}: {shape}")


if __name__ == "__main__":
    main()
