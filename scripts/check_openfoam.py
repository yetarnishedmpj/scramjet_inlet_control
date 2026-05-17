from __future__ import annotations

import argparse

from scramjet_rl.cfd.openfoam import check_openfoam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="rhoCentralFoam")
    args = parser.parse_args()
    result = check_openfoam(args.solver)
    print(result.message)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
