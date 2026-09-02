"""Compare three conditional-CVaR engines inside the same direct MKABSDE."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .evaluation import evaluate_solution
from .problem import black_scholes_product_value
from .risk import RiskEstimatorConfig, paper_risk_config
from .simulation import simulate_paths
from .solvers import solve_direct
from .timing import timed_call


PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = ("quadrature", "nested_mc", "learning")


def _write_json(path: Path, payload: Any) -> None:
    """Persist strict, indented JSON after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime(device: torch.device) -> dict[str, Any]:
    """Describe the software stack and timing device used by the campaign."""
    runtime: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        runtime["gpu_name"] = torch.cuda.get_device_properties(device).name
    return runtime


def _settings(method: str, product: str, quick: bool) -> RiskEstimatorConfig:
    """Return calibrated paper settings or deliberately inexpensive smoke settings."""
    settings = paper_risk_config(method, product)  # type: ignore[arg-type]
    if not quick:
        return settings
    if method == "quadrature":
        return replace(settings, state_batch_size=2_000)
    if method == "nested_mc":
        return replace(
            settings,
            nested_inner_samples=1_024,
            nested_training_states=256,
            state_batch_size=64,
        )
    return replace(
        settings,
        learning_width=16,
        learning_batch_size=4_096,
        learning_initial_updates=128,
        learning_updates_per_date=8,
    )


def _learning_diagnostics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    """Summarize active-date exceedance rates of the VaR learner."""
    active = [row for row in rows if int(row["step"]) > 0]
    exceedance = [float(row["exceedance_rate"]) for row in active]
    return {
        "mean_exceedance_rate": statistics.mean(exceedance),
        "max_exceedance_deviation_from_one_percent": max(
            abs(value - 0.01) for value in exceedance
        ),
    }


def _run_seed(
    product: str,
    steps: int,
    paths: int,
    seed: int,
    degree: int,
    device: torch.device,
    quick: bool,
) -> dict[str, Any]:
    """Run all three direct solvers on the same outer trajectories."""
    quadrature_nodes = 24 if product == "call" else 64
    config = ExperimentConfig(
        product=product,
        time_steps=steps,
        paths=paths,
        seed=seed,
        hermite_degree=degree,
        cvar_quadrature_nodes=quadrature_nodes,
    )
    brownian, increments, spots = simulate_paths(config, device)
    spot0 = torch.tensor([config.spot], device=device, dtype=brownian.dtype)
    black_scholes_price = float(black_scholes_product_value(0.0, spot0, config))
    outputs: dict[str, Any] = {}
    for method in METHODS:
        settings = _settings(method, product, quick)
        result, timing = timed_call(
            lambda settings=settings: solve_direct(
                config,
                brownian,
                increments,
                spots,
                risk_settings=settings,
            ),
            device,
        )
        metrics = evaluate_solution(result, config)
        diagnostics = (
            _learning_diagnostics(result.risk_diagnostics)
            if method == "learning"
            else {}
        )
        outputs[method] = {
            "settings": settings.as_dict(),
            "price": metrics["price"],
            "im_correction": metrics["price"] - black_scholes_price,
            "delta": metrics["delta"],
            "gpu_seconds": timing.gpu_seconds,
            "wall_seconds": timing.wall_seconds,
            "diagnostics": diagnostics,
        }
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prices = [outputs[method]["price"] for method in METHODS]
    return {
        "product": product,
        "time_steps": steps,
        "paths": paths,
        "seed": seed,
        "hermite_degree": degree,
        "black_scholes_price": black_scholes_price,
        "methods": outputs,
        "largest_within_seed_price_gap": max(prices) - min(prices),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return a sample mean and standard deviation for replicated quantities."""
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate method prices, IM corrections, timings, and paired differences."""
    summary = []
    for product in sorted({row["product"] for row in rows}):
        selected = [row for row in rows if row["product"] == product]
        quadrature_by_seed = {
            row["seed"]: row["methods"]["quadrature"]["price"]
            for row in selected
        }
        for method in METHODS:
            prices = [row["methods"][method]["price"] for row in selected]
            corrections = [
                row["methods"][method]["im_correction"] for row in selected
            ]
            gpu_times = [
                row["methods"][method]["gpu_seconds"] for row in selected
            ]
            price_mean, price_std = _mean_std(prices)
            correction_mean, correction_std = _mean_std(corrections)
            differences = [
                row["methods"][method]["price"]
                - quadrature_by_seed[row["seed"]]
                for row in selected
            ]
            summary.append(
                {
                    "product": product,
                    "method": method,
                    "replicates": len(selected),
                    "price_mean": price_mean,
                    "price_std": price_std,
                    "im_correction_mean": correction_mean,
                    "im_correction_std": correction_std,
                    "gpu_seconds_mean": statistics.mean(gpu_times),
                    "price_difference_vs_quadrature_mean": statistics.mean(
                        differences
                    ),
                    "price_difference_vs_quadrature_max_abs": max(
                        abs(value) for value in differences
                    ),
                }
            )
    return summary


def _markdown(summary: list[dict[str, Any]]) -> str:
    """Render the three-method comparison as a compact human-readable table."""
    lines = [
        "# Conditional-CVaR engines in the direct MKABSDE",
        "",
        "All methods use identical outer paths, Hermite regressions, and backward",
        "updates. Only the conditional-risk estimator changes.",
        "",
        "| Product | Method | Price | Seed std | IM correction | GPU time | "
        "Mean difference vs quadrature |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['product']} | {row['method']} | "
            f"{row['price_mean']:.8f} | {row['price_std']:.2e} | "
            f"{row['im_correction_mean']:.8f} | "
            f"{row['gpu_seconds_mean']:.2f} s | "
            f"{row['price_difference_vs_quadrature_mean']:+.2e} |"
        )
    lines.extend(
        [
            "",
            "Agreement concerns the computed time-zero price at the retained grid;",
            "it is not a comparison with an unavailable exact MKABSDE solution.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse the reproducible direct-MKABSDE estimator comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=("all", "call", "butterfly"), default="all")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all requested products and write raw, summary, and Markdown artifacts."""
    args = parse_args()
    if args.quick:
        args.steps = 50
        args.paths = 5_000
        args.seeds = (42,)
    device = torch.device(args.device)
    products = ("call", "butterfly") if args.product == "all" else (args.product,)
    output = args.output or (
        Path("/tmp/initial-margin-risk-smoke.json")
        if args.quick
        else PROJECT_ROOT / "results" / "cvar_method_comparison.json"
    )
    rows = []
    for product in products:
        for seed in args.seeds:
            row = _run_seed(
                product,
                args.steps,
                args.paths,
                seed,
                args.degree,
                device,
                args.quick,
            )
            rows.append(row)
            _write_json(
                output,
                {"runtime": _runtime(device), "rows": rows, "summary": []},
            )
            prices = {
                method: row["methods"][method]["price"] for method in METHODS
            }
            print(product, seed, prices, flush=True)
    summary = aggregate(rows)
    _write_json(
        output,
        {"runtime": _runtime(device), "rows": rows, "summary": summary},
    )
    report = output.with_suffix(".md")
    report.write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
