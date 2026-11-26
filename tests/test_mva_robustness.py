"""Audit the long-horizon estimators without rerunning costly GPU experiments."""

import json
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from initial_margin.compare_risk_estimators import _resume_rows, _settings, aggregate, main
from initial_margin.config import ExperimentConfig
from tools.mva_robustness import ESTIMATORS, PRODUCTS, method_comparison, spatial_comparison

RESULTS = Path(__file__).resolve().parents[1] / "initial_margin/results"


def test_matched_runs_and_unchanged_settings():
    """Match every outer seed and quadrature price to the independent horizon sweep."""
    long = json.loads((RESULTS / "cvar_method_long_horizons.json").read_text())
    baseline = json.loads((RESULTS / "cvar_method_comparison.json").read_text())["rows"]
    settings = {(r["product"], m): r["methods"][m]["settings"] for r in baseline for m in ESTIMATORS}
    proxy_runs = {}
    for line in (RESULTS / "margin_periods/runs.jsonl").read_text().splitlines():
        row = json.loads(line)
        cfg = row["config"]
        proxy_runs[cfg["product"], cfg["margin_period"], cfg["seed"]] = row
    assert long["complete"] and long["summary"] == aggregate(long["rows"])
    assert len(long["rows"]) == 12
    # An uninterrupted reproduction has no previous segment. The retained restart
    # changed only the CLI; every numerical source must have the identical hash.
    active = long["runtime"]["source_sha256"]
    for segment in long.get("previous_segments", []):
        assert 0 < segment["through_row"] <= len(long["rows"])
        previous = segment["runtime"]["source_sha256"]
        assert previous.keys() == active.keys()
        assert all(previous[path] == active[path] for path in active
                   if path != "initial_margin/compare_risk_estimators.py")
    assert {(r["product"], r["margin_period"], r["seed"]) for r in long["rows"]} == set(
        product(PRODUCTS, (0.10, 0.50), (42, 43, 44)))
    for r in long["rows"]:
        ref = proxy_runs[r["product"], r["margin_period"], r["seed"]]
        assert r["config"] == ref["config"]
        assert r["methods"]["quadrature"]["price"] == pytest.approx(
            ref["result"]["methods"]["direct"]["price"], abs=1e-11, rel=0)
        for m in ESTIMATORS:
            assert r["methods"][m]["settings"] == settings[r["product"], m]
            assert r["methods"][m]["gpu_seconds"] > 0


def test_exported_comparisons_match_seed_level_prices():
    """Check paired differences and the distinct three- and ten-seed spatial means."""
    methods = method_comparison(RESULTS)
    assert len(methods) == 6
    assert all(len(block["paired_gaps"]) == 3
               for row in methods for block in row["methods"].values())
    spatial = spatial_comparison(RESULTS)
    assert [(r["margin_period"], r["time_steps"]) for r in spatial] == [
        (0.02, 50), (0.02, 100), (0.02, 200), (0.10, 200), (0.50, 200)]
    for p in PRODUCTS:
        for s in spatial[3:]:
            q = next(r for r in methods if (r["product"], r["margin_period"]) == (p, s["margin_period"]))
            assert s["products"][p]["hermite"] == pytest.approx(q["methods"]["quadrature"]["mean"], abs=1e-9, rel=0)
    baseline = json.loads((RESULTS / "atm_call_experiment.json").read_text())
    fine = next(r for r in baseline["summary"] if r["time_steps"] == 200)
    assert spatial[2]["products"]["call"]["hermite"] == fine["direct_mkabsde_mva_percent_of_black_scholes_mean"]


def test_paired_gap_uses_the_same_seed_and_relative_mva_units():
    """Rebuild one delicate butterfly comparison from its two raw time-zero prices."""
    key = ("butterfly", 0.10, 43)
    runs = json.loads((RESULTS / "cvar_method_long_horizons.json").read_text())["rows"]
    risk = next(r for r in runs if (r["product"], r["margin_period"], r["seed"]) == key)
    proxies = (json.loads(line) for line in (RESULTS / "margin_periods/runs.jsonl").read_text().splitlines())
    proxy = next(r for r in proxies if tuple(r["config"][k] for k in ("product", "margin_period", "seed")) == key)
    base = risk["black_scholes_price"]
    ratio = ((risk["methods"]["learning"]["price"] - base) /
             (proxy["result"]["methods"]["nonlinear"]["price"] - base))
    group = next(r for r in method_comparison(RESULTS) if (r["product"], r["margin_period"]) == key[:2])
    paired = next(r for r in group["methods"]["learning"]["paired_gaps"] if r["seed"] == key[2])
    assert paired["nonlinear"] == pytest.approx(100 * (ratio - 1), abs=1e-10)


def test_campaign_rejects_off_grid_horizons_and_protects_output(tmp_path, monkeypatch):
    """Reject off-grid lags and do not overwrite an earlier audit or its report."""
    output = tmp_path / "comparison.json"
    args = ["compare_risk_estimators", "--device", "cpu", "--steps", "50", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args + ["--deltas", "0.03"])
    with pytest.raises(ValueError, match="integer"):
        main()
    assert not output.exists()
    output.touch()
    monkeypatch.setattr(sys, "argv", args + ["--deltas", "0.1"])
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == b""


def test_fresh_three_horizon_audit_can_regenerate_the_paper_table(tmp_path):
    """A single fresh audit is interchangeable with the two retained archives."""
    files = [RESULTS / name for name in ("cvar_method_comparison.json", "cvar_method_long_horizons.json")]
    rows = [row for path in files for row in json.loads(path.read_text())["rows"]]
    replacement = tmp_path / "audit.json"
    replacement.write_text(json.dumps({"complete": True, "rows": list(reversed(rows))}))
    assert method_comparison(RESULTS, replacement) == method_comparison(RESULTS)
    replacement.write_text(json.dumps({"complete": False, "rows": rows}))
    with pytest.raises(ValueError, match="incomplete"):
        method_comparison(RESULTS, replacement)


def test_resume_preserves_prior_provenance_and_rejects_mismatches(tmp_path):
    """Only the runner may change; a changed seed, solver, or finished run must fail."""
    cfg = ExperimentConfig(product="butterfly", margin_period=.1, time_steps=200)
    args = SimpleNamespace(paths=cfg.paths, steps=cfg.time_steps, degree=cfg.hermite_degree, quick=False)
    plan = [(cfg.product, cfg.margin_period, cfg.seed)]
    row = {"config": cfg.as_dict(), "methods": {
        m: {"settings": _settings(m, cfg.product, False).as_dict()} for m in ESTIMATORS}}
    runner = "initial_margin/compare_risk_estimators.py"
    versions = dict.fromkeys(("python", "numpy", "scipy", "torch", "cuda", "gpu", "device"), "original")
    old = {**versions, "source_sha256": {runner: "old-cli", "solver": "same"}}
    new = {**versions, "source_sha256": {runner: "new-cli", "solver": "same"}}
    checkpoint = tmp_path / "audit.json"
    payload = {"complete": False, "rows": [row], "runtime": old}
    checkpoint.write_text(json.dumps(payload))
    rows, history = _resume_rows(checkpoint, plan, args, new)
    assert rows == [row] and history == [{"through_row": 1, "runtime": old}]
    with pytest.raises(ValueError, match="matching prefix"):
        _resume_rows(checkpoint, [(cfg.product, cfg.margin_period, cfg.seed + 1)], args, new)
    with pytest.raises(ValueError, match="solver sources"):
        _resume_rows(checkpoint, plan, args, {**new, "source_sha256": {"solver": "changed"} })
    checkpoint.write_text(json.dumps({**payload, "complete": True}))
    with pytest.raises(ValueError, match="completed"):
        _resume_rows(checkpoint, plan, args, new)
