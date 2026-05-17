from __future__ import annotations

import argparse

from scramjet_rl.cfd.cases import materialize_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="cfd/sweep_manifest.csv")
    parser.add_argument("--template-dir", default="cfd/templates/openfoam_case")
    parser.add_argument("--output-dir", default="data/raw_openfoam/cases")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cases = materialize_cases(args.manifest, args.template_dir, args.output_dir, args.overwrite)
    print(f"Materialized {len(cases)} cases under {args.output_dir}")


if __name__ == "__main__":
    main()
