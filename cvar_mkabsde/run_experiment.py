"""Run direct, non-linear-proxy, and linear-proxy CVaR BSDE experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from cvar_mkabsde.config import ExperimentConfig
from cvar_mkabsde.evaluation import compare_results
from cvar_mkabsde.solvers import solve_all


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse financial, discretization, and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=("call", "butterfly"), default="call")
    parser.add_argument("--strike", type=float, default=20.0)
    parser.add_argument("--butterfly-half-width", type=float, default=2.0)
    parser.add_argument("--steps", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--quadrature-nodes", type=int, default=24)
    parser.add_argument("--margin-period", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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


def main() -> None:
    """Execute every requested common-random-number experiment and write strict JSON."""
    args = parse_args()
    base = ExperimentConfig(
        product=args.product,
        strike=args.strike,
        butterfly_half_width=args.butterfly_half_width,
        paths=args.paths,
        hermite_degree=args.degree,
        cvar_quadrature_nodes=args.quadrature_nodes,
        margin_period=args.margin_period,
    )
    rows = []
    for steps in args.steps:
        for seed in args.seeds:
            config = replace(base, time_steps=steps, seed=seed)
            _, _, _, results = solve_all(config, args.device)
            row = {
                "time_steps": steps,
                "h": config.dt,
                "seed": seed,
                "config": config.as_dict(),
                "metrics": compare_results(results, config),
            }
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    payload = {"rows": rows, "summary": aggregate(rows)}
    output = args.output or (
        PROJECT_ROOT / "results" / f"atm_{args.product}_experiment.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
