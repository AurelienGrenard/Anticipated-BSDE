"""Reproduce the larger-margin-period table and independent refinement checks.

All runs reuse the existing MVA solvers. Defaults reproduce the paper table;
--reference replaces regression by grid integration, and --quick makes an
inexpensive CPU-compatible pipeline check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from .config import ExperimentConfig
from .deterministic_reference import DeterministicReference, call_proxy_mvas
from .problem import black_scholes_product_value
from .simulation import simulate_paths
from .solvers import solve_direct, solve_proxy
from .timing import timed_call


METHODS = ("direct", "nonlinear", "linear")
ROOT = Path(__file__).resolve().parents[1]


def provenance(device: torch.device) -> dict[str, Any]:
    """Record versions and exact source hashes, including uncommitted source edits."""
    sources = [
        path for folder in ("hermite", "initial_margin")
        for path in sorted((ROOT / folder).rglob("*.py"))
    ]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "git_revision": revision,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


def run_hermite(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    """Fit the three MVA equations on common paths, timing each solver separately."""
    paths = simulate_paths(config, device)
    spot = torch.tensor([config.spot], device=device, dtype=torch.float64)
    base = float(black_scholes_product_value(0.0, spot, config)[0])
    methods = {}
    for name in METHODS:
        # Lambdas run immediately; each solver receives the same three path tensors.
        solve = (lambda: solve_direct(config, *paths)) if name == "direct" else (
            lambda: solve_proxy(config, *paths, linear=name == "linear")
        )
        result, timing = timed_call(solve, device)
        price = float(result.y_values[0, 0])
        methods[name] = {
            "price": price, "mva": price - base,
            "mva_percent": 100.0 * (price - base) / base,
            "gpu_seconds": timing.gpu_seconds if device.type == "cuda" else None,
            "wall_seconds": timing.wall_seconds,
        }
        del result
    return {"black_scholes_price": base, "methods": methods}


def run_reference(
    config: ExperimentConfig, space_steps: int, expectation_nodes: int
) -> dict[str, Any]:
    """Integrate conditional functions on a grid for all three time-discrete equations."""
    engine = DeterministicReference(
        config, space_steps=space_steps, expectation_nodes=expectation_nodes,
        cvar_nodes=config.cvar_quadrature_nodes,
    )
    direct, direct_timing = timed_call(engine.solve, torch.device("cpu"))
    proxies, proxy_timing = timed_call(engine.solve_proxies, torch.device("cpu"))
    base = float(direct["black_scholes_price"])
    mvas = {"direct": direct["mva"], **proxies}
    return {
        "black_scholes_price": base,
        "methods": {
            name: {"price": base + mva, "mva": mva, "mva_percent": 100.0 * mva / base}
            for name, mva in mvas.items()
        },
        "direct_wall_seconds": direct_timing.wall_seconds,
        "proxies_wall_seconds": proxy_timing.wall_seconds,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate independent seeds, rejecting duplicate seeds or mixed configurations."""
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        config = row["config"]
        key = (config["product"], config["margin_period"], config["time_steps"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (product, delta, steps), selected in groups.items():
        template = {key: value for key, value in selected[0]["config"].items() if key != "seed"}
        seeds = [row["config"]["seed"] for row in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seeds in an aggregation group")
        for row in selected:
            if {key: value for key, value in row["config"].items() if key != "seed"} != template:
                raise ValueError("mixed numerical or financial configurations")
        methods = {}
        for name in METHODS:
            values = [row["result"]["methods"][name] for row in selected]
            methods[name] = {}
            for metric in ("price", "mva", "mva_percent"):
                samples = [value[metric] for value in values]
                methods[name][metric + "_mean"] = statistics.mean(samples)
                methods[name][metric + "_std"] = (
                    statistics.stdev(samples) if len(samples) > 1 else None
                )
        summary.append({
            "product": product, "margin_period": delta, "time_steps": steps,
            "replicates": len(selected), "paths": template["paths"],
            "hermite_degree": template["hermite_degree"], "methods": methods,
        })
    return summary


def table_rows(summary: list[dict[str, Any]]) -> str:
    """Export the manuscript's four-decimal table rows directly from stored means."""
    lines = []
    for row in summary:
        values = " & ".join(
            "$" + f"{row['methods'][name]['mva_percent_mean']:.4f}" + "$" for name in METHODS
        )
        lines.append(
            row["product"].capitalize() + " & $" + f"{row['margin_period']:.2f}"
            + "$ & " + values + " " + r"\\"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Expose the table settings and explicit time/space/quadrature refinement controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--products", nargs="+", choices=("call", "butterfly"),
                        default=["call", "butterfly"])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.02, 0.10, 0.50])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--nodes-multiplier", type=int, default=1)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--space-steps", type=int, default=6_000)
    parser.add_argument("--expectation-nodes", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.steps, args.paths, args.seeds = 50, 500, [42, 43]
        args.space_steps, args.expectation_nodes = 300, 8
    if min(args.steps, args.paths, args.nodes_multiplier, args.space_steps,
           args.expectation_nodes) <= 0 or any(delta <= 0 for delta in args.deltas):
        parser.error("grid sizes, path count, quadrature multiplier and deltas must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be distinct")
    if len(args.deltas) != len(set(args.deltas)) or len(args.products) != len(set(args.products)):
        parser.error("deltas and products must be distinct")
    # Validate every horizon before creating any output or performing expensive work.
    for delta in args.deltas:
        ExperimentConfig(time_steps=args.steps, margin_period=delta, hermite_degree=args.degree)
    if args.reference:
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu for a smoke run")
    return args


def main() -> None:
    """Run a fresh campaign and retain raw rows, sample statistics, and LaTeX rows."""
    args = parse_args()
    torch.set_num_threads(1)
    # Fail on existing folders: reproduction must not silently overwrite paper data.
    output = args.output_dir
    if output is None:
        output = (Path(tempfile.mkdtemp(prefix="absde-margin-periods-")) if args.quick
                  else ROOT / "initial_margin" / "results" / (
                      "margin_period_reference" if args.reference else "margin_periods"
                  ))
        if not args.quick:
            output.mkdir(parents=True, exist_ok=False)
    else:
        output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    metadata = {
        "schema_version": 1, "mode": "reference" if args.reference else "hermite",
        "settings": {key: value for key, value in vars(args).items() if key != "output_dir"},
        "runtime": provenance(device),
        "timing_scope": "solver only; excludes path simulation and JSON writing",
        "dispersion": "sample standard deviation across seeds, not a bias/error estimate",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for product in args.products:
        for delta in args.deltas:
            for seed in (args.seeds[:1] if args.reference else args.seeds):
                config = ExperimentConfig(
                    product=product, time_steps=args.steps, paths=args.paths, seed=seed,
                    hermite_degree=args.degree, margin_period=delta,
                    cvar_quadrature_nodes=(24 if product == "call" else 64) * args.nodes_multiplier,
                )
                result = (run_reference(config, args.space_steps, args.expectation_nodes)
                          if args.reference else run_hermite(config, device))
                row = {"config": config.as_dict(), "result": result}
                if product == "call":
                    row["exact_call_proxy_mvas"] = call_proxy_mvas(config)
                with (output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                print(f"{product} Delta={delta:g} N={args.steps} seed={seed}: "
                      + ", ".join(f"{name}={result['methods'][name]['mva_percent']:.6f}%"
                                  for name in METHODS), flush=True)
    summary = aggregate(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "table_rows.tex").write_text(table_rows(summary))
    print(f"Saved {len(rows)} runs in {output}", flush=True)


if __name__ == "__main__":
    main()
