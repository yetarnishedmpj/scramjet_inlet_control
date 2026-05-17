from __future__ import annotations

import argparse

from scramjet_rl.cfd.replay import rollout_to_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-csv", required=True)
    parser.add_argument("--output", default="cfd/replay_manifest.csv")
    args = parser.parse_args()
    print(f"Wrote replay manifest to {rollout_to_manifest(args.rollout_csv, args.output)}")


if __name__ == "__main__":
    main()
