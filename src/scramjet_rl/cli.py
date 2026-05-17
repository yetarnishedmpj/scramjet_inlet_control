from __future__ import annotations

import argparse

from scramjet_rl.config import load_yaml, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="scramjet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("generate-data")
    data.add_argument("--config", required=True)

    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument("path")

    manifest = subparsers.add_parser("make-cfd-manifest")
    manifest.add_argument("--output", default="cfd/sweep_manifest.csv")
    manifest.add_argument("--count", type=int, default=128)
    manifest.add_argument("--seed", type=int, default=0)

    cfd_check = subparsers.add_parser("check-openfoam")
    cfd_check.add_argument("--solver", default="rhoCentralFoam")

    materialize = subparsers.add_parser("materialize-cfd")
    materialize.add_argument("--manifest", default="cfd/sweep_manifest.csv")
    materialize.add_argument("--template-dir", default="cfd/templates/openfoam_case")
    materialize.add_argument("--output-dir", default="data/raw_openfoam/cases")
    materialize.add_argument("--overwrite", action="store_true")

    postprocess = subparsers.add_parser("postprocess-cfd")
    postprocess.add_argument("--manifest", default="cfd/sweep_manifest.csv")
    postprocess.add_argument("--sampled-dir", default="data/raw_openfoam/sampled")
    postprocess.add_argument("--output", default="data/processed/openfoam_inlet.h5")
    postprocess.add_argument("--height", type=int, default=32)
    postprocess.add_argument("--width", type=int, default=64)

    plot = subparsers.add_parser("plot-sample")
    plot.add_argument("--dataset", default="data/processed/synthetic_inlet.h5")
    plot.add_argument("--index", type=int, default=0)
    plot.add_argument("--output", default="outputs/sample_fields.png")

    train_surrogate = subparsers.add_parser("train-surrogate")
    train_surrogate.add_argument("--config", required=True)

    eval_surrogate = subparsers.add_parser("evaluate-surrogate")
    eval_surrogate.add_argument("--dataset", default="data/processed/synthetic_inlet.h5")
    eval_surrogate.add_argument("--checkpoint", default="models/surrogate_cnn.pt")
    eval_surrogate.add_argument("--plot-output", default=None)

    train_agent = subparsers.add_parser("train-agent")
    train_agent.add_argument("--config", required=True)

    eval_agent = subparsers.add_parser("evaluate-agent")
    eval_agent.add_argument("--config", required=True)
    eval_agent.add_argument("--policy-path", required=True)
    eval_agent.add_argument("--episodes", type=int, default=5)
    eval_agent.add_argument("--rollout-csv", default="outputs/policy_rollouts.csv")

    replay = subparsers.add_parser("replay-rollout")
    replay.add_argument("--rollout-csv", required=True)
    replay.add_argument("--output", default="cfd/replay_manifest.csv")

    args = parser.parse_args()
    _dispatch(args)


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "generate-data":
        from scramjet_rl.data.synthetic import SyntheticConfig, generate_synthetic_fields, write_hdf5

        raw = load_yaml(args.config)
        validate_config(raw, "data")
        ranges = raw["ranges"]
        config = SyntheticConfig(
            num_samples=int(raw["num_samples"]),
            height=int(raw["height"]),
            width=int(raw["width"]),
            seed=int(raw.get("seed", 0)),
            mach_range=tuple(ranges["mach"]),
            altitude_range_m=tuple(ranges["altitude_m"]),
            ramp_angle_range_deg=tuple(ranges["ramp_angle_deg"]),
        )
        write_hdf5(raw["output_path"], generate_synthetic_fields(config))
        print(f"Wrote {raw['output_path']} with {config.num_samples} samples")
    elif args.command == "validate-dataset":
        from scramjet_rl.data.schema import validate_hdf5_schema

        for name, shape in validate_hdf5_schema(args.path).items():
            print(f"{name}: {shape}")
    elif args.command == "make-cfd-manifest":
        from scramjet_rl.cfd.manifest import write_sweep_manifest

        print(f"Wrote CFD sweep manifest to {write_sweep_manifest(args.output, args.count, args.seed)}")
    elif args.command == "check-openfoam":
        from scramjet_rl.cfd.openfoam import check_openfoam

        result = check_openfoam(args.solver)
        print(result.message)
    elif args.command == "materialize-cfd":
        from scramjet_rl.cfd.cases import materialize_cases

        cases = materialize_cases(args.manifest, args.template_dir, args.output_dir, args.overwrite)
        print(f"Materialized {len(cases)} cases under {args.output_dir}")
    elif args.command == "postprocess-cfd":
        from scramjet_rl.cfd.postprocess import sampled_csv_to_hdf5

        print(
            "Wrote processed CFD dataset to "
            f"{sampled_csv_to_hdf5(args.manifest, args.sampled_dir, args.output, args.height, args.width)}"
        )
    elif args.command == "plot-sample":
        from scramjet_rl.plots import plot_dataset_sample

        print(f"Wrote {plot_dataset_sample(args.dataset, args.index, args.output)}")
    elif args.command == "train-surrogate":
        from scramjet_rl.surrogate.train import train_surrogate

        config = load_yaml(args.config)
        validate_config(config, "surrogate")
        print(f"Wrote surrogate checkpoint to {train_surrogate(config)}")
    elif args.command == "evaluate-surrogate":
        from scramjet_rl.surrogate.evaluate import evaluate_surrogate, plot_prediction_sample

        metrics = evaluate_surrogate(args.dataset, args.checkpoint)
        for name, value in metrics.items():
            print(f"{name}: {value:.6g}")
        if args.plot_output:
            print(f"Wrote {plot_prediction_sample(args.dataset, args.checkpoint, args.plot_output)}")
    elif args.command == "train-agent":
        from scramjet_rl.rl.train import train_agent

        config = load_yaml(args.config)
        validate_config(config, "rl")
        print(f"Wrote policy to {train_agent(config)}")
    elif args.command == "evaluate-agent":
        from scramjet_rl.rl.evaluate import evaluate_policy

        config = load_yaml(args.config)
        config["policy_path"] = args.policy_path
        for name, value in evaluate_policy(config, args.episodes, args.rollout_csv).items():
            print(f"{name}: {value:.6g}")
    elif args.command == "replay-rollout":
        from scramjet_rl.cfd.replay import rollout_to_manifest

        print(f"Wrote replay manifest to {rollout_to_manifest(args.rollout_csv, args.output)}")


if __name__ == "__main__":
    main()
