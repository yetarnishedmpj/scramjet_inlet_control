from __future__ import annotations

import argparse
from pathlib import Path

from scramjet_rl.cfd.cases import run_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", default="data/raw_openfoam/cases")
    parser.add_argument("--solver", default="rhoCentralFoam")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    case_dirs = sorted(path for path in Path(args.cases_dir).iterdir() if path.is_dir())
    run_cases(case_dirs, args.solver, args.dry_run)


if __name__ == "__main__":
    main()
