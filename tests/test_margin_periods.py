"""Checks for the larger-Delta table, its references, and reproducibility contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from scipy.integrate import quad

from initial_margin.config import ExperimentConfig
from initial_margin.deterministic_reference import DeterministicReference, call_proxy_mvas
from initial_margin.margin_periods import METHODS, aggregate, run_hermite, table_rows
from initial_margin.problem import black_scholes_product_delta, gaussian_cvar_constant


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "initial_margin" / "results"


@pytest.mark.parametrize("delta", [0.02, 0.10, 0.50, 1.0, 1.5])
def test_call_linear_formula_integrates_the_truncated_horizon(delta: float) -> None:
    """Check the closed form against numerical integration, including Delta >= T."""
    config = ExperimentConfig(margin_period=delta, time_steps=200)
    spot = torch.tensor([config.spot], dtype=torch.float64)
    hedge = float(black_scholes_product_delta(0.0, spot, config)[0])
    integral = quad(
        lambda t: math.sqrt(min(delta, config.maturity - t)),
        0.0, config.maturity, epsabs=1e-12,
        points=[max(0.0, config.maturity - delta)],
    )[0]
    expected = (config.funding_spread * gaussian_cvar_constant(config.alpha)
                * config.volatility * config.spot * hedge * integral)
    assert call_proxy_mvas(config)["linear"] == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("product", ["call", "butterfly"])
def test_new_reference_proxies_have_zero_mva_without_funding(product: str) -> None:
    """Check grid integration and the analytic call proxies preserve the zero-source identity."""
    config = ExperimentConfig(product=product, margin_period=0.5, funding_spread=0.0)
    engine = DeterministicReference(config, space_steps=300, expectation_nodes=8)
    assert engine.solve_proxies() == {"linear": 0.0, "nonlinear": 0.0}
    assert engine.solve()["mva"] == 0.0
    if product == "call":
        assert call_proxy_mvas(config) == pytest.approx({"linear": 0.0, "nonlinear": 0.0})
    else:
        with pytest.raises(ValueError, match="only for a call"):
            call_proxy_mvas(config)


def test_grid_call_proxies_approach_the_closed_forms() -> None:
    """Check a modest grid is consistent with both independent call formulas at large Delta."""
    config = ExperimentConfig(margin_period=0.5, time_steps=100)
    values = DeterministicReference(config, space_steps=3_000).solve_proxies()
    exact = call_proxy_mvas(config)
    for name in ("linear", "nonlinear"):
        assert abs(values[name] - exact[name]) < 7e-4


def test_hermite_sweep_is_repeatable_and_uses_cpu_wall_timing() -> None:
    """Repeat a small seeded large-Delta run and check price/MVA identities and timing labels."""
    torch.set_num_threads(1)
    config = ExperimentConfig(time_steps=10, margin_period=0.5, paths=256)
    first = run_hermite(config, torch.device("cpu"))
    second = run_hermite(config, torch.device("cpu"))
    for name in METHODS:
        value = first["methods"][name]
        assert value["price"] == second["methods"][name]["price"]
        assert value["price"] == pytest.approx(first["black_scholes_price"] + value["mva"])
        assert value["mva_percent"] == pytest.approx(100 * value["mva"] / first["black_scholes_price"])
        assert value["gpu_seconds"] is None
        assert value["wall_seconds"] > 0


def test_aggregation_uses_sample_sd_and_rejects_invalid_pooling() -> None:
    """Check SD is not a standard error and incompatible runs cannot be silently pooled."""
    rows = []
    for seed, value in [(42, 1.0), (43, 3.0)]:
        rows.append({
            "config": replace(ExperimentConfig(), seed=seed).as_dict(),
            "result": {"methods": {
                name: {"price": value, "mva": value, "mva_percent": value} for name in METHODS
            }},
        })
    summary = aggregate(rows)
    assert summary[0]["methods"]["direct"]["mva_percent_mean"] == 2.0
    assert summary[0]["methods"]["direct"]["mva_percent_std"] == pytest.approx(math.sqrt(2))
    assert "$2.0000$" in table_rows(summary)
    assert aggregate(rows[:1])[0]["methods"]["direct"]["mva_percent_std"] is None
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate([rows[0], rows[0]])
    mixed = copy.deepcopy(rows)
    mixed[1]["config"]["hermite_degree"] = 4
    with pytest.raises(ValueError, match="mixed"):
        aggregate(mixed)


def test_cli_protects_existing_outputs_and_rejects_off_grid_delays(tmp_path: Path) -> None:
    """Ensure a rerun cannot overwrite artifacts and invalid horizons fail before computation."""
    command = [sys.executable, "-m", "initial_margin.margin_periods", "--device", "cpu",
               "--output-dir", str(tmp_path)]
    existing = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert existing.returncode != 0
    assert "FileExistsError" in existing.stderr
    invalid = subprocess.run(command + ["--deltas", "0.023"], cwd=ROOT, capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "must be an integer" in invalid.stderr
    assert not list(tmp_path.iterdir())


def test_retained_margin_table_is_derived_from_all_raw_runs() -> None:
    """Rebuild each published mean and LaTeX row from the eighteen retained degree-two fits."""
    folder = RESULTS / "margin_periods"
    rows = [json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines()]
    summary = json.loads((folder / "summary.json").read_text())
    assert len(rows) == 18
    assert summary == aggregate(rows)
    assert (folder / "table_rows.tex").read_text() == table_rows(summary)
    assert {(row["product"], row["margin_period"]) for row in summary} == {
        (product, delta) for product in ("call", "butterfly") for delta in (0.02, 0.10, 0.50)
    }
    for row in rows:
        config = row["config"]
        assert config["hermite_degree"] == 2
        assert config["time_steps"] == 200 and config["paths"] == 200_000
        assert config["seed"] in (42, 43, 44)
    for row in summary:
        for method in METHODS:
            assert row["methods"][method]["mva_percent_std"] < 0.006


def test_retained_sweep_records_the_current_solver_sources() -> None:
    """Bind all new campaigns to their exact source files and software/timing metadata."""
    folders = sorted(RESULTS.glob("margin_period*"))
    assert len(folders) == 5
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text())
        runtime = metadata["runtime"]
        assert runtime["python"] and runtime["torch"] and runtime["device"]
        assert metadata["schema_version"] == 1
        for name, digest in runtime["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_retained_hermite_time_check_preserves_large_delta_gaps() -> None:
    """Compare the same seed on N=200 and N=400 without mistaking this for a convergence rate."""
    coarse = [json.loads(line) for line in (RESULTS / "margin_periods" / "runs.jsonl").read_text().splitlines()]
    fine = [json.loads(line) for line in (RESULTS / "margin_period_time_check" / "runs.jsonl").read_text().splitlines()]
    assert len(fine) == 4
    by_key = {
        (row["config"]["product"], row["config"]["margin_period"]): row
        for row in coarse if row["config"]["seed"] == 42
    }
    for row in fine:
        config = row["config"]
        assert config["time_steps"] == 400 and config["seed"] == 42
        if config["margin_period"] == 0.5:
            previous = by_key[(config["product"], config["margin_period"])]["result"]["methods"]
            values = row["result"]["methods"]
            gap = values["direct"]["mva"] - values["nonlinear"]["mva"]
            assert gap < 0
            assert abs(gap) > 10 * abs(values["direct"]["mva"] - previous["direct"]["mva"])


def test_retained_large_delta_gaps_survive_independent_refinement() -> None:
    """Check the model gap dominates grid/quadrature checks without treating them as exact errors."""
    summaries = {
        name: json.loads((RESULTS / name / "summary.json").read_text())
        for name in ("margin_periods", "margin_period_reference",
                     "margin_period_double_q", "margin_period_refined")
    }
    indexed = {
        name: {(row["product"], row["margin_period"]): row for row in rows}
        for name, rows in summaries.items()
    }
    for key, row in indexed["margin_periods"].items():
        direct = row["methods"]["direct"]["mva_mean"]
        same = indexed["margin_period_reference"][key]["methods"]["direct"]["mva_mean"]
        doubled = indexed["margin_period_double_q"][key]["methods"]["direct"]["mva_mean"]
        refined = indexed["margin_period_refined"][key]["methods"]["direct"]["mva_mean"]
        assert abs(direct - same) < 3e-5
        assert abs(doubled - same) < 1e-5
        assert abs(refined - same) < 7e-5
        if key[1] == 0.5:
            proxy = row["methods"]["nonlinear"]["mva_mean"]
            refined_proxy = indexed["margin_period_refined"][key]["methods"]["nonlinear"]["mva_mean"]
            assert abs(direct - proxy) > 10 * abs(direct - refined)
            assert (direct - proxy) * (refined - refined_proxy) > 0
