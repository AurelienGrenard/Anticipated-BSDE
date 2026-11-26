"""Build the two robustness comparisons from published runs, without refitting."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

PRODUCTS = ("call", "butterfly")
HORIZONS = (0.02, 0.10, 0.50)
ESTIMATORS = ("quadrature", "nested_mc", "learning")


def _paired_runs(results: Path, audit_file: Path | None) -> dict:
    """Join ES and proxy runs by product, margin period and seed, never by list order."""
    proxy_runs = {}
    for line in (results / "margin_periods/runs.jsonl").read_text().splitlines():
        row = json.loads(line)
        cfg = row["config"]
        key = cfg["product"], cfg["margin_period"], cfg["seed"]
        if key in proxy_runs:
            raise ValueError(f"duplicate proxy run: {key}")
        proxy_runs[key] = row

    by_key = {}
    files = ([audit_file] if audit_file else
             [results / "cvar_method_comparison.json", results / "cvar_method_long_horizons.json"])
    for path in files:
        audit = json.loads(path.read_text())
        legacy = audit_file is None and path.name == "cvar_method_comparison.json"
        if not audit.get("complete", legacy):
            raise ValueError(f"incomplete comparison: {path}")
        for row in audit["rows"]:
            key = row["product"], row.get("margin_period", 0.02), row["seed"]
            if key in by_key:
                raise ValueError(f"duplicate ES run: {key}")
            by_key[key] = row
    if set(by_key) != set(proxy_runs):
        raise ValueError("the ES comparison and margin-period campaign must have the same paired runs")
    pairs = {}
    for key, audit in by_key.items():
        proxy = proxy_runs[key]
        config = proxy["config"]
        if any(audit[k] != config[k] for k in ("time_steps", "paths", "hermite_degree")):
            raise ValueError(f"unmatched regression or simulation settings: {key}")
        if "config" in audit and audit["config"] != config:
            raise ValueError(f"unmatched financial or numerical parameters: {key}")
        pairs[key] = audit["methods"], proxy["result"]["methods"]
    return pairs


def method_comparison(results: Path, audit_file: Path | None = None) -> list[dict]:
    """Report MVA sample means, sample SDs and signed within-seed gaps to both proxies."""
    pairs = _paired_runs(results, audit_file)
    comparisons = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            seeds = sorted(s for p, h, s in pairs if (p, h) == (product, horizon))
            if len(seeds) != 3:
                raise ValueError(f"expected three paired paper runs for {product}, horizon={horizon}")
            measures = {}
            for method in ESTIMATORS:
                values, paired = [], []
                for seed in seeds:
                    estimators, proxies = pairs[product, horizon, seed]
                    value = estimators[method]["mva_percent_of_black_scholes"]
                    values.append(value)
                    # A signed relative difference (%) is not a percentage-point difference.
                    paired.append({"seed": seed, **{
                        name: 100 * (value / proxies[name]["mva_percent"] - 1)
                        for name in ("nonlinear", "linear")
                    }})
                measures[method] = {"mean": mean(values), "sd": stdev(values), "paired_gaps": paired}
            comparisons.append({"product": product, "margin_period": horizon, "methods": measures})
    return comparisons


def spatial_comparison(results: Path) -> list[dict]:
    """Use ten baseline runs for the N sweep and three matched runs at longer horizons."""
    base_ref = json.loads((results / "deterministic_reference.json").read_text())
    fine = {(r["product"], 0.02, r["time_steps"]): r["mva_percent_of_black_scholes"]
            for r in base_ref["rows"]}
    hermite = {}
    for product in PRODUCTS:
        data = json.loads((results / f"atm_{product}_experiment.json").read_text())
        for r in data["summary"]:
            hermite[product, 0.02, r["time_steps"]] = r["direct_mkabsde_mva_percent_of_black_scholes_mean"]
    for store, folder in ((hermite, "margin_periods"), (fine, "margin_period_reference")):
        for r in json.loads((results / folder / "summary.json").read_text()):
            if r["margin_period"] != 0.02:
                store[r["product"], r["margin_period"], r["time_steps"]] = r["methods"]["direct"]["mva_percent_mean"]
    rows = []
    for horizon, steps in ((0.02, 50), (0.02, 100), (0.02, 200), (0.10, 200), (0.50, 200)):
        cells = {}
        for product in PRODUCTS:
            key = product, horizon, steps
            h, ref = hermite[key], fine[key]
            cells[product] = {"hermite": h, "fine_grid": ref, "relative_gap_percent": 100 * (h / ref - 1)}
        rows.append({"margin_period": horizon, "time_steps": steps, "products": cells})
    return rows
