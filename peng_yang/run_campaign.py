"""Run the minimal-degree Hermite experiments behind the convergence figures.

One invocation trains every requested grid/seed pair, evaluates both the grid-date
and left-constant reconstruction errors, saves the representative checkpoint used
by the quantile figure, and writes the aggregated convergence summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from .checkpoints import save_result
from .config import ExperimentConfig
from .evaluation import evaluate, evaluate_piecewise_reconstruction
from .problems import get_example
from .solver import train_backward


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STEPS = (5, 10, 20, 40, 80, 160)
DEFAULT_SEEDS = (42, 43, 44)
SLUGS = {
    "4.3": "example_4_3",
    "5.3.1": "example_5_3_1",
    "5.3.2": "example_5_3_2",
}


def parse_args() -> argparse.Namespace:
    """Define and parse the complete, reproducible experiment command line."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the three Peng--Yang ABSDE benchmarks."
    )
    parser.add_argument(
        "--example",
        choices=("all", *SLUGS),
        default="all",
        help="benchmark to run; the default reproduces all paper experiments",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--train-paths", type=int, default=1_000_000)
    parser.add_argument("--grid-test-paths", type=int, default=200_000)
    parser.add_argument("--reconstruction-test-paths", type=int, default=100_000)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="override the exact-solution defaults (1 for 4.3; 2 for 5.3)",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result directory (defaults to peng_yang/results)",
    )
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--checkpoint-seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run an inexpensive smoke campaign under /tmp",
    )
    return parser.parse_args()


def convergence_order(previous: dict[str, Any], current: dict[str, Any], key: str) -> float:
    """Estimate the local exponent p in error = O(h**p) from adjacent grids."""
    return math.log(previous[key] / current[key]) / math.log(
        previous["h"] / current["h"]
    )


def aggregate_reconstruction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate reconstruction errors across seeds and attach local log-log orders."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)

    summary: list[dict[str, Any]] = []
    for steps in sorted(grouped):
        replicates = grouped[steps]
        y_values = [row["reconstruction_y_mse_sup"] for row in replicates]
        z_values = [row["reconstruction_z_mse_integrated"] for row in replicates]
        summary.append(
            {
                "N": steps,
                "h": replicates[0]["h"],
                "hermite_degree": replicates[0]["hermite_degree"],
                "replicates": len(replicates),
                "y_mse_sup_mean": mean(y_values),
                "y_mse_sup_std": stdev(y_values) if len(y_values) > 1 else 0.0,
                "z_mse_integrated_mean": mean(z_values),
                "z_mse_integrated_std": stdev(z_values)
                if len(z_values) > 1
                else 0.0,
                "y_mse_sup_order": None,
                "z_mse_integrated_order": None,
            }
        )

    for previous, current in zip(summary, summary[1:]):
        current["y_mse_sup_order"] = convergence_order(
            previous, current, "y_mse_sup_mean"
        )
        current["z_mse_integrated_order"] = convergence_order(
            previous, current, "z_mse_integrated_mean"
        )
    return summary


def write_json(path: Path, payload: Any) -> None:
    """Write strict, human-readable JSON and create the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_experiment(
    args: argparse.Namespace,
    example_key: str,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train, independently evaluate, checkpoint, and aggregate one complete campaign."""
    example = get_example(example_key)
    slug = SLUGS[example_key]
    rows: list[dict[str, Any]] = []

    for steps in args.steps:
        for seed in args.seeds:
            config = ExperimentConfig(
                example=example_key,
                time_steps=steps,
                train_paths=args.train_paths,
                hermite_degree=args.degree,
                ridge=args.ridge,
                seed=seed,
                dtype=args.dtype,
            )
            result = train_backward(config, example, args.device, progress=False)

            if steps == args.checkpoint_step and seed == args.checkpoint_seed:
                checkpoint = (
                    output / "checkpoints" / f"{slug}_n{steps}_seed{seed}.pt"
                )
                save_result(result, checkpoint)

            # The grid audit diagnoses regression error at the dates of the scheme.
            grid_audit = evaluate(
                result.models,
                example,
                config,
                args.device,
                test_paths=args.grid_test_paths,
            )
            # The reconstruction audit produces the two quantities plotted in the paper.
            reconstruction = evaluate_piecewise_reconstruction(
                result.models,
                example,
                config,
                args.device,
                test_paths=args.reconstruction_test_paths,
                subdivisions=args.subdivisions,
            )
            row = {
                "hermite_degree": config.hermite_degree,
                "N": steps,
                "h": config.dt,
                "seed": seed,
                "train_paths": args.train_paths,
                "grid_test_paths": args.grid_test_paths,
                "reconstruction_test_paths": args.reconstruction_test_paths,
                "subdivisions": args.subdivisions,
                "grid_y_mse_sup": grid_audit.metrics.y_mse_sup,
                "grid_z_mse_integrated": grid_audit.metrics.z_mse_integrated,
                "reconstruction_y_mse_sup": reconstruction.y_mse_sup,
                "reconstruction_z_mse_integrated": reconstruction.z_mse_integrated,
                "training_wall_seconds": result.wall_seconds,
            }
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)

    summary = aggregate_reconstruction(rows)
    write_json(output / f"{slug}_campaign.json", rows)
    write_json(output / f"{slug}_reconstruction_summary.json", summary)
    return rows, summary


def main() -> None:
    """Run one or all benchmark campaigns and report every generated file."""
    args = parse_args()
    if args.quick:
        args.steps = (5, 10)
        args.seeds = (42,)
        args.train_paths = 10_000
        args.grid_test_paths = 5_000
        args.reconstruction_test_paths = 5_000
        args.subdivisions = 4
    output = args.output or (
        Path("/tmp/peng-yang-smoke") if args.quick else PROJECT_ROOT / "results"
    )
    examples = tuple(SLUGS) if args.example == "all" else (args.example,)
    for example_key in examples:
        _, summary = run_experiment(args, example_key, output)
        slug = SLUGS[example_key]
        print(
            f"Wrote {len(summary)} grids to "
            f"{output / f'{slug}_reconstruction_summary.json'}"
        )


if __name__ == "__main__":
    main()
