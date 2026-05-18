from __future__ import annotations

import argparse

from scramjet_rl.data.importers import import_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--pressure", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        "Wrote dataset to "
        f"{import_arrays(args.inputs, args.pressure, args.temperature, args.metrics, args.output)}"
    )


if __name__ == "__main__":
    main()
