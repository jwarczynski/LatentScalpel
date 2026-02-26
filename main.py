"""CLI entry point for the GENIE SAE experiment pipeline.

Subcommands:
    collect-activations  - Collect activations from GENIE model
    train-sae            - Train Top-K SAEs on stored activations
    evaluate             - Evaluate SAE reconstruction impact
    find-top-examples    - Find top-activating dataset examples per SAE feature
    interpret-features   - Interpret SAE features via LLM-as-judge protocol

Usage:
    uv run python main.py collect-activations configs/activation_collection.yaml
    uv run python main.py train-sae configs/train_sae.yaml --layer_idx=3
    uv run python main.py train-sae configs/train_sae.yaml --layers 0 1 2 3 4 5 --submit --infra.cluster=slurm
    uv run python main.py evaluate configs/evaluation.yaml
    uv run python main.py find-top-examples configs/find_top_examples.yaml
    uv run python main.py interpret-features configs/interpret_features.yaml

All Exca configs support ``--infra.folder`` and ``--infra.cluster`` overrides
via CLI args, enabling caching and slurm submission without editing config files.
"""

from __future__ import annotations

import argparse
import json
import sys

from exca import ConfDict


def _load_config_dict(path: str, overrides: list[str] | None = None) -> dict:
    """Load a JSON/YAML config and merge CLI overrides via ConfDict."""
    import yaml

    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    cfdict = ConfDict(data)
    if overrides:
        cli_overrides = ConfDict.from_args(overrides)
        cfdict.update(cli_overrides)

    return dict(cfdict)


def cmd_collect_activations(args: argparse.Namespace) -> None:
    from geniesae.configs import ActivationCollectionConfig

    data = _load_config_dict(args.config, args.overrides)
    config = ActivationCollectionConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
        print(f"UID: {config.infra.uid()}")
        if config.infra.uid_folder():
            print(f"Folder: {config.infra.uid_folder()}")
    else:
        output_dir = config.apply()
        print(f"Activations saved to: {output_dir}")


def _submit_train_job_array(base_config, layers: list[int]) -> None:
    """Submit a Slurm job array — one job per layer via exca's job_array.

    When ``base_config.resume_from`` points to a directory, each job
    gets its own ``layer_XX.ckpt`` path resolved automatically by
    ``resolve_checkpoint_path()``.
    """
    with base_config.infra.job_array() as array:
        for layer_idx in layers:
            overrides: dict = {"layer_idx": layer_idx}
            task = base_config.infra.clone_obj(overrides)
            array.append(task)
    print(f"Submitted job array for layers {layers}")
    for task in array:
        print(f"  layer {task.layer_idx}: status={task.infra.status()}")


def _submit_train_single(config) -> None:
    """Submit a single layer training job to Slurm (non-blocking)."""
    config.infra.job()
    print(f"Job submitted for layer {config.layer_idx}. Status: {config.infra.status()}")


def _run_train_inline(config) -> None:
    """Train a single layer SAE in the current process (blocking)."""
    print(
        f"[main] train-sae: activation_dir={config.activation_dir}, "
        f"layer_idx={config.layer_idx}, output_dir={config.output_dir}",
        flush=True,
    )
    ckpt_path = config.apply()
    print(f"SAE checkpoint saved to: {ckpt_path}")


def cmd_train_sae(args: argparse.Namespace) -> None:
    from geniesae.configs import SAETrainingConfig

    data = _load_config_dict(args.config, args.overrides)
    layers = args.layers

    # CLI --resume_from overrides config value
    if args.resume_from is not None:
        data["resume_from"] = args.resume_from

    if args.submit and layers:
        base_config = SAETrainingConfig(**{**data, "layer_idx": layers[0]})
        _submit_train_job_array(base_config, layers)
    elif args.submit:
        _submit_train_single(SAETrainingConfig(**data))
    else:
        _run_train_inline(SAETrainingConfig(**data))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from geniesae.configs import EvaluationConfig

    data = _load_config_dict(args.config, args.overrides)
    config = EvaluationConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
        print(f"UID: {config.infra.uid()}")
    else:
        results = config.apply()
        print(f"Evaluation complete. Baseline loss: {results['baseline_loss']}")


def cmd_find_top_examples(args: argparse.Namespace) -> None:
    from geniesae.configs import TopExamplesConfig

    data = _load_config_dict(args.config, args.overrides)
    if args.features:
        data["features"] = args.features
    config = TopExamplesConfig(**data)

    if args.submit:
        config.infra.job()
    else:
        config.apply()


def cmd_interpret_features(args: argparse.Namespace) -> None:
    from geniesae.configs import InterpretFeaturesConfig

    data = _load_config_dict(args.config, args.overrides)
    if args.features:
        data["features"] = args.features
    config = InterpretFeaturesConfig(**data)

    if args.submit:
        config.infra.job()
    else:
        config.apply()

def cmd_test_notify(args: argparse.Namespace) -> None:
    from scripts.test_notify import NotifyTestConfig

    data = _load_config_dict(args.config, args.overrides)
    config = NotifyTestConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"Test result: {result}")

def cmd_collect_trajectory(args: argparse.Namespace) -> None:
    from geniesae.configs import TrajectoryConfig

    data = _load_config_dict(args.config, args.overrides)
    config = TrajectoryConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"Trajectory data saved to: {result}")


def cmd_collect_plaid_activations(args: argparse.Namespace) -> None:
    from geniesae.configs import PlaidCollectionConfig

    data = _load_config_dict(args.config, args.overrides)
    config = PlaidCollectionConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"PLAID activations saved to: {result}")

def cmd_evaluate_plaid(args: argparse.Namespace) -> None:
    from geniesae.configs import PlaidEvaluationConfig

    data = _load_config_dict(args.config, args.overrides)
    config = PlaidEvaluationConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        results = config.apply()
        print(f"PLAID evaluation complete.")

def cmd_collect_t5_activations(args: argparse.Namespace) -> None:
    from geniesae.configs import T5CollectionConfig

    data = _load_config_dict(args.config, args.overrides)
    config = T5CollectionConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"T5 activations saved to: {result}")

def cmd_correlate_features(args: argparse.Namespace) -> None:
    from geniesae.configs import CorrelationConfig

    data = _load_config_dict(args.config, args.overrides)
    config = CorrelationConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"Correlation results saved to: {result}")




def cmd_collect_plaid_trajectory(args: argparse.Namespace) -> None:
    from geniesae.configs import PlaidTrajectoryConfig

    data = _load_config_dict(args.config, args.overrides)
    config = PlaidTrajectoryConfig(**data)

    if args.submit:
        config.infra.job()
        print(f"Job submitted. Status: {config.infra.status()}")
    else:
        result = config.apply()
        print(f"PLAID trajectory data saved to: {result}")





def main() -> None:
    parser = argparse.ArgumentParser(description="GENIE SAE experiment pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    def _add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", help="Path to JSON/YAML config file")
        p.add_argument(
            "--submit", action="store_true",
            help="Submit job non-blocking via infra.job() instead of running inline",
        )

    p_collect = subparsers.add_parser(
        "collect-activations", help="Collect activations from GENIE model",
    )
    _add_common_args(p_collect)
    p_collect.set_defaults(func=cmd_collect_activations)

    p_train = subparsers.add_parser(
        "train-sae", help="Train a Top-K SAE for one or more layers",
    )
    p_train.add_argument("config", help="Path to JSON/YAML config file")
    p_train.add_argument(
        "--submit", action="store_true",
        help="Submit job(s) non-blocking via exca infra instead of running inline",
    )
    p_train.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layer indices to train (used with --submit for job array). "
             "Example: --layers 0 1 2 3 4 5",
    )
    p_train.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to .ckpt file or directory of layer_XX.ckpt files to resume from.",
    )
    p_train.set_defaults(func=cmd_train_sae)

    p_eval = subparsers.add_parser(
        "evaluate", help="Evaluate SAE reconstruction impact",
    )
    _add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_top = subparsers.add_parser(
        "find-top-examples", help="Find top-activating dataset examples per SAE feature",
    )
    _add_common_args(p_top)
    p_top.add_argument(
        "--features", type=int, nargs="+", default=None,
        help="Feature indices to process (default: all features)",
    )
    p_top.set_defaults(func=cmd_find_top_examples)

    p_interp = subparsers.add_parser(
        "interpret-features", help="Interpret SAE features via LLM-as-judge protocol",
    )
    _add_common_args(p_interp)
    p_interp.add_argument(
        "--features", type=int, nargs="+", default=None,
        help="Feature indices to interpret (default: all features)",
    )
    p_interp.set_defaults(func=cmd_interpret_features)

    p_test_notify = subparsers.add_parser(
        "test-notify", help="Dummy job to test ntfy notifications (no GPU)",
    )
    _add_common_args(p_test_notify)
    p_test_notify.set_defaults(func=cmd_test_notify)

    p_trajectory = subparsers.add_parser(
        "collect-trajectory",
        help="Collect SAE feature activations along the full denoising trajectory",
    )
    _add_common_args(p_trajectory)
    p_trajectory.set_defaults(func=cmd_collect_trajectory)

    p_plaid_collect = subparsers.add_parser(
        "collect-plaid-activations",
        help="Collect activations from PLAID diffusion model",
    )
    _add_common_args(p_plaid_collect)
    p_plaid_collect.set_defaults(func=cmd_collect_plaid_activations)

    p_plaid_eval = subparsers.add_parser(
        "evaluate-plaid",
        help="Evaluate SAE reconstruction impact on PLAID model",
    )
    _add_common_args(p_plaid_eval)
    p_plaid_eval.set_defaults(func=cmd_evaluate_plaid)

    p_plaid_traj = subparsers.add_parser(
        "collect-plaid-trajectory",
        help="Collect SAE feature activations along PLAID's denoising trajectory",
    )
    _add_common_args(p_plaid_traj)
    p_plaid_traj.set_defaults(func=cmd_collect_plaid_trajectory)

    p_t5_collect = subparsers.add_parser(
        "collect-t5-activations",
        help="Collect activations from T5 encoder-decoder model",
    )
    _add_common_args(p_t5_collect)
    p_t5_collect.set_defaults(func=cmd_collect_t5_activations)

    p_correlate = subparsers.add_parser(
        "correlate-features",
        help="Compute cross-model feature correlation between Genie and T5 SAEs",
    )
    _add_common_args(p_correlate)
    p_correlate.set_defaults(func=cmd_correlate_features)

    args, overrides = parser.parse_known_args()
    args.overrides = overrides if overrides else None
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
