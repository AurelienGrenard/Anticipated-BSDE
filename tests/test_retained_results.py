"""Certification checks for the retained ATM numerical experiments."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "initial_margin"
    / "results"
    / "atm_call_experiment.json"
)
BUTTERFLY_RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "initial_margin"
    / "results"
    / "atm_butterfly_experiment.json"
)
REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "initial_margin"
    / "results"
    / "deterministic_reference.json"
)
RISK_COMPARISON_PATH = (
    Path(__file__).resolve().parents[1]
    / "initial_margin"
    / "results"
    / "cvar_method_comparison.json"
)


def test_retained_proxy_reproduces_article_atm_call() -> None:
    """Check the finest non-linear proxy agrees with the article's reported value."""
    payload = json.loads(RESULT_PATH.read_text())
    finest = payload["summary"][-1]
    assert finest["time_steps"] == 200
    price = finest["nonlinear_proxy_price_mean"]
    assert abs(price - 2.1959) < 1e-3


def test_direct_price_is_stable_under_refinement() -> None:
    """Check the direct MKABSDE estimate has stabilized on the two finest grids."""
    payload = json.loads(RESULT_PATH.read_text())
    coarse, fine = payload["summary"][-2:]
    difference = abs(
        coarse["direct_mkabsde_price_mean"]
        - fine["direct_mkabsde_price_mean"]
    )
    assert difference < 5e-5


def test_retained_linear_proxy_reproduces_article_atm_butterfly() -> None:
    """Check the finest linear butterfly proxy agrees with the article table."""
    payload = json.loads(BUTTERFLY_RESULT_PATH.read_text())
    finest = payload["summary"][-1]
    price = finest["linear_proxy_price_mean"]
    assert abs(price - 0.3112) < 2e-4


def test_direct_butterfly_price_is_stable_under_refinement() -> None:
    """Check the direct butterfly estimate is stable on the two finest grids."""
    payload = json.loads(BUTTERFLY_RESULT_PATH.read_text())
    coarse, fine = payload["summary"][-2:]
    difference = abs(
        coarse["direct_mkabsde_price_mean"]
        - fine["direct_mkabsde_price_mean"]
    )
    assert difference < 1e-5


def test_direct_regressions_match_the_deterministic_reference() -> None:
    """Check the retained direct prices agree with same-grid non-MC references."""
    reference_payload = json.loads(REFERENCE_PATH.read_text())
    reference = {
        (row["product"], row["time_steps"]): row["price"]
        for row in reference_payload["rows"]
    }
    for product, path, tolerance in (
        ("call", RESULT_PATH, 5e-5),
        ("butterfly", BUTTERFLY_RESULT_PATH, 1e-5),
    ):
        payload = json.loads(path.read_text())
        for row in payload["summary"]:
            estimate = row["direct_mkabsde_price_mean"]
            assert abs(estimate - reference[(product, row["time_steps"])]) < tolerance


def test_retained_relative_mva_uses_the_black_scholes_price() -> None:
    """Check every reported MVA percentage has the stated financial meaning."""
    for path in (RESULT_PATH, BUTTERFLY_RESULT_PATH):
        payload = json.loads(path.read_text())
        black_scholes = payload["rows"][0]["metrics"]["black_scholes_no_im"][
            "price"
        ]
        for row in payload["summary"]:
            for solver in ("direct_mkabsde", "nonlinear_proxy", "linear_proxy"):
                mva = row[f"{solver}_mva_mean"]
                relative_mva = row[
                    f"{solver}_mva_percent_of_black_scholes_mean"
                ]
                assert abs(relative_mva - 100.0 * mva / black_scholes) < 1e-14


def test_three_cvar_engines_agree_on_the_direct_price() -> None:
    """Check independent CVaR engines support the same MKABSDE conclusion."""
    payload = json.loads(RISK_COMPARISON_PATH.read_text())
    assert {row["seed"] for row in payload["rows"]} == {42, 43, 44}
    assert {row["hermite_degree"] for row in payload["rows"]} == {2}
    for row in payload["rows"]:
        assert row["largest_within_seed_price_gap"] < 1e-4
