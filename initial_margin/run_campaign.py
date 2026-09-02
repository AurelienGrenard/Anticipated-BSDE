"""Run the complete direct and proxy initial-margin experiment.

Each grid and seed uses one common path sample for the direct MKABSDE, both
proxy BSDEs, and the no-IM control. The retained JSON contains every raw run
and the cross-seed summaries displayed by the result notebook.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from .config import ExperimentConfig
from .evaluation import compare_results
from .risk import paper_risk_config
from .solvers import solve_all


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse financial, discretization, and output settings."""
    parser = argparse.ArgumentParser(
        description="Reproduce the call and butterfly initial-margin experiments."
    )
    parser.add_argument(
        "--product",
        choices=("all", "call", "butterfly"),
        default="all",
        help="product to run; the default reproduces both paper experiments",
    )
    parser.add_argument("--strike", type=float, default=20.0)
    parser.add_argument("--butterfly-half-width", type=float, default=2.0)
    parser.add_argument("--steps", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=tuple(range(42, 52)),
        help="independent seeds (paper default: 42 through 51)",
    )
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument(
        "--quadrature-nodes",
        type=int,
        default=None,
        help="override the paper defaults (24 for calls, 64 for butterflies)",
    )
    parser.add_argument("--margin-period", type=float, default=0.02)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="result directory (defaults to initial_margin/results)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one inexpensive grid and seed under /tmp",
    )
    return parser.parse_args()


def local_order(previous: float, current: float, h_ratio: float) -> float:
    """Return the adjacent-grid empirical convergence exponent."""
    return math.log(previous / current) / math.log(h_ratio)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average seeds at every grid and retain variability and proxy distances."""
    summary = []
    for steps in sorted({row["time_steps"] for row in rows}):
        group = [row for row in rows if row["time_steps"] == steps]
        item: dict[str, Any] = {
            "time_steps": steps,
            "h": group[0]["h"],
            "hermite_degree": group[0]["config"]["hermite_degree"],
            "risk_method": group[0]["risk_estimator"]["method"],
            "replicates": len(group),
        }
        for solver in (
            "direct_mkabsde",
            "nonlinear_proxy",
            "linear_proxy",
            "no_im_control",
        ):
            for metric in ("price", "delta"):
                values = [row["metrics"][solver][metric] for row in group]
                item[f"{solver}_{metric}_mean"] = mean(values)
                item[f"{solver}_{metric}_std"] = (
                    stdev(values) if len(values) > 1 else 0.0
                )
            if solver != "no_im_control":
                for metric in (
                    "control_variate_price",
                    "control_variate_delta",
                    "price_correction_vs_numerical_no_im",
                    "price_correction_vs_black_scholes",
                ):
                    values = [row["metrics"][solver][metric] for row in group]
                    item[f"{solver}_{metric}_mean"] = mean(values)
                    item[f"{solver}_{metric}_std"] = (
                        stdev(values) if len(values) > 1 else 0.0
                    )
        for proxy in ("nonlinear_proxy", "linear_proxy"):
            for metric in ("y_grid_mse_sup", "z_grid_mse_integrated"):
                values = [
                    row["metrics"][f"direct_vs_{proxy}"][metric] for row in group
                ]
                item[f"direct_vs_{proxy}_{metric}_mean"] = mean(values)
        summary.append(item)
    if len(summary) >= 3:
        for solver in ("direct_mkabsde", "nonlinear_proxy", "linear_proxy"):
            key = f"{solver}_control_variate_price_mean"
            coarse_difference = abs(summary[-3][key] - summary[-2][key])
            fine_difference = abs(summary[-2][key] - summary[-1][key])
            summary[-1][f"{solver}_price_self_convergence_order"] = (
                local_order(
                    coarse_difference,
                    fine_difference,
                    summary[-2]["h"] / summary[-1]["h"],
                )
                if fine_difference > 0.0
                else None
            )
    return summary


def run_product(
    args: argparse.Namespace,
    product: str,
    output_directory: Path,
) -> Path:
    """Run one product campaign and persist its raw rows and summary."""
    quadrature_nodes = args.quadrature_nodes or (
        24 if product == "call" else 64
    )
    base = ExperimentConfig(
        product=product,
        strike=args.strike,
        butterfly_half_width=args.butterfly_half_width,
        paths=args.paths,
        hermite_degree=args.degree,
        cvar_quadrature_nodes=quadrature_nodes,
        margin_period=args.margin_period,
    )
    rows = []
    risk_settings = paper_risk_config("quadrature", product)
    if args.quadrature_nodes is not None:
        risk_settings = replace(
            risk_settings,
            quadrature_nodes=args.quadrature_nodes,
        )
    for steps in args.steps:
        for seed in args.seeds:
            config = replace(base, time_steps=steps, seed=seed)
            # solve_all reuses this seed's paths across equations, so their
            # differences contain less Monte Carlo noise than separate runs.
            results = solve_all(
                config,
                args.device,
                risk_settings=risk_settings,
            )[-1]
            row = {
                "time_steps": steps,
                "h": config.dt,
                "seed": seed,
                "config": config.as_dict(),
                "risk_estimator": risk_settings.as_dict(),
                "metrics": compare_results(results, config),
            }
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            del results
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    payload = {"rows": rows, "summary": aggregate(rows)}
    output = (
        output_directory / f"atm_{product}_experiment.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    return output


def main() -> None:
    """Execute one or both complete common-random-number campaigns."""
    args = parse_args()
    if args.quick:
        args.steps = (50,)
        args.seeds = (42,)
        args.paths = 20_000
    output_directory = args.output_directory or (
        Path("/tmp/initial-margin-smoke")
        if args.quick
        else PROJECT_ROOT / "results"
    )
    products = (
        ("call", "butterfly") if args.product == "all" else (args.product,)
    )
    for product in products:
        run_product(args, product, output_directory)


if __name__ == "__main__":
    main()
