"""Compare three conditional-CVaR engines inside the same direct MKABSDE."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from dataclasses import replace
from itertools import product as combinations
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .evaluation import evaluate_solution
from .problem import black_scholes_product_value
from .risk import RiskEstimatorConfig, paper_risk_config
from .reproducibility import provenance
from .simulation import simulate_paths
from .solvers import solve_direct
from .timing import timed_call


METHODS = ("quadrature", "nested_mc", "learning")


def _write_json(path: Path, payload: Any) -> None:
    """Persist strict, indented JSON after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime(device: torch.device) -> dict[str, Any]:
    """Attach the common solver/source provenance to the comparison."""
    return provenance(device)


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
    margin_period: float,
) -> dict[str, Any]:
    """Run all three direct solvers on the same outer trajectories."""
    quadrature_nodes = ExperimentConfig.cvar_quadrature_nodes
    config = ExperimentConfig(
        product=product,
        time_steps=steps,
        paths=paths,
        margin_period=margin_period,
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
            "mva": metrics["price"] - black_scholes_price,
            "mva_percent_of_black_scholes": (
                100.0
                * (metrics["price"] - black_scholes_price)
                / black_scholes_price
            ),
            "gpu_seconds": timing.gpu_seconds,
            "wall_seconds": timing.wall_seconds,
            "diagnostics": diagnostics,
        }
        del result
        print(f"  {method}: MVA/BS = {outputs[method]['mva_percent_of_black_scholes']:.6f}%", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prices = [outputs[method]["price"] for method in METHODS]
    return {
        "product": product,
        "margin_period": margin_period,
        "config": config.as_dict(),
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
    """Aggregate prices, MVA measures, timings, and paired differences."""
    summary = []
    # The original single-horizon archive predates the margin_period field.
    keys = sorted({(r["product"], r.get("margin_period", 0.02)) for r in rows})
    for product, horizon in keys:
        selected = [r for r in rows if (r["product"], r.get("margin_period", 0.02)) == (product, horizon)]
        if len({r["seed"] for r in selected}) != len(selected):
            raise ValueError("duplicate seed in product/horizon group")
        quadrature_by_seed = {
            row["seed"]: row["methods"]["quadrature"]["price"]
            for row in selected
        }
        for method in METHODS:
            prices = [row["methods"][method]["price"] for row in selected]
            mvas = [
                row["methods"][method]["mva"] for row in selected
            ]
            relative_mvas = [
                row["methods"][method]["mva_percent_of_black_scholes"]
                for row in selected
            ]
            gpu_times = [
                row["methods"][method]["gpu_seconds"] for row in selected
            ]
            price_mean, price_std = _mean_std(prices)
            mva_mean, mva_std = _mean_std(mvas)
            relative_mva_mean, relative_mva_std = _mean_std(relative_mvas)
            differences = [
                row["methods"][method]["price"]
                - quadrature_by_seed[row["seed"]]
                for row in selected
            ]
            summary.append(
                {
                    "product": product,
                    **({"margin_period": horizon} if "margin_period" in selected[0] else {}),
                    "method": method,
                    "replicates": len(selected),
                    "price_mean": price_mean,
                    "price_std": price_std,
                    "mva_mean": mva_mean,
                    "mva_std": mva_std,
                    "mva_percent_of_black_scholes_mean": relative_mva_mean,
                    "mva_percent_of_black_scholes_std": relative_mva_std,
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
        "All methods use identical outer paths, degree-two MVA/control regressions",
        "in the paper runs, and the same backward update. Only the risk fit changes.",
        "",
        "| Product | Horizon (years) | Method | Price (mean +/- seed std) | "
        "MVA (% of BS; mean +/- seed std) | GPU time | "
        "Mean difference vs quadrature |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['product']} | {row['margin_period']:.2f} | {row['method']} | "
            f"{row['price_mean']:.8f} +/- {row['price_std']:.2e} | "
            f"{row['mva_percent_of_black_scholes_mean']:.6f}% +/- "
            f"{row['mva_percent_of_black_scholes_std']:.2e} pp | "
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
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.02],
                        help="Margin periods in years; each must lie on the time grid.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an incomplete --output after its last full seed.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if len(set(args.deltas)) != len(args.deltas) or len(set(args.seeds)) != len(args.seeds):
        parser.error("use distinct horizons and seeds")
    if min(args.deltas) <= 0 or args.steps < 1 or args.paths < 1:
        parser.error("horizons, steps and path count must be positive")
    if args.resume and not args.output:
        parser.error("--resume requires an explicit --output")
    return args


def _resume_rows(output, plan, args, runtime):
    """Recover a completed prefix only when paths, estimator settings and solvers match."""
    saved = json.loads(output.read_text())
    if saved.get("complete", True) or output.with_suffix(".md").exists():
        raise ValueError("cannot resume a completed audit")
    previous = saved["runtime"]
    if any(previous[k] != runtime[k] for k in ("python", "numpy", "scipy", "torch", "cuda", "gpu", "device")):
        raise ValueError("cannot resume with different software or device")
    runner = "initial_margin/compare_risk_estimators.py"
    if ({k: v for k, v in previous["source_sha256"].items() if k != runner}
            != {k: v for k, v in runtime["source_sha256"].items() if k != runner}):
        raise ValueError("cannot resume with different numerical solver sources")
    rows = saved["rows"]
    if len(rows) > len(plan):
        raise ValueError("checkpoint exceeds the requested campaign")
    for row, (product, horizon, seed) in zip(rows, plan):
        expected = ExperimentConfig(product=product, margin_period=horizon, seed=seed,
                                    paths=args.paths, time_steps=args.steps, hermite_degree=args.degree)
        if row["config"] != expected.as_dict() or any(
            row["methods"][m]["settings"] != _settings(m, product, args.quick).as_dict() for m in METHODS
        ):
            raise ValueError("checkpoint is not a matching prefix of the requested campaign")
    history = list(saved.get("previous_segments", []))
    if rows and previous != runtime:
        history.append({"through_row": len(rows), "runtime": previous})
    return rows, history


def main() -> None:
    """Run all requested products and write raw, summary, and Markdown artifacts."""
    args = parse_args()
    torch.set_num_threads(4)
    if args.quick:
        args.steps = 50
        args.paths = 5_000
        args.seeds = (42,)
    device = torch.device(args.device)
    products = ("call", "butterfly") if args.product == "all" else (args.product,)
    for horizon in args.deltas:
        ExperimentConfig(time_steps=args.steps, margin_period=horizon)
    output = args.output or Path(tempfile.mkdtemp(prefix="absde-es-")) / "comparison.json"
    if not args.resume and (output.exists() or output.with_suffix(".md").exists()):
        raise FileExistsError(f"choose a fresh output path: {output}")
    runtime = _runtime(device)
    plan = list(combinations(products, args.deltas, args.seeds))
    rows, history = _resume_rows(output, plan, args, runtime) if args.resume else ([], [])
    metadata = {"runtime": runtime, "previous_segments": history}
    for product, horizon, seed in plan[len(rows):]:
        print(f"{product}, horizon={horizon}, seed={seed}", flush=True)
        rows.append(_run_seed(
            product, args.steps, args.paths, seed, args.degree, device, args.quick, horizon
        ))
        _write_json(output, {"complete": False, **metadata, "rows": rows, "summary": []})
    summary = aggregate(rows)
    _write_json(
        output,
        {"complete": True, **metadata, "rows": rows, "summary": summary},
    )
    report = output.with_suffix(".md")
    report.write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
