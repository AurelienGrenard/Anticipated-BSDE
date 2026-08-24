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
ORACLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "initial_margin"
    / "results"
    / "deterministic_oracle.json"
)


def test_retained_proxy_reproduces_article_atm_call() -> None:
    """Check the finest non-linear proxy agrees with the article's reported value."""
    payload = json.loads(RESULT_PATH.read_text())
    finest = payload["summary"][-1]
    assert finest["time_steps"] == 200
    price = finest["nonlinear_proxy_control_variate_price_mean"]
    assert abs(price - 2.1959) < 1e-3


def test_direct_price_is_stable_under_refinement() -> None:
    """Check the direct MKABSDE estimate has stabilized on the two finest grids."""
    payload = json.loads(RESULT_PATH.read_text())
    coarse, fine = payload["summary"][-2:]
    difference = abs(
        coarse["direct_mkabsde_control_variate_price_mean"]
        - fine["direct_mkabsde_control_variate_price_mean"]
    )
    assert difference < 5e-5


def test_retained_linear_proxy_reproduces_article_atm_butterfly() -> None:
    """Check the finest linear butterfly proxy agrees with the article table."""
    payload = json.loads(BUTTERFLY_RESULT_PATH.read_text())
    finest = payload["summary"][-1]
    price = finest["linear_proxy_control_variate_price_mean"]
    assert abs(price - 0.3112) < 2e-4


def test_direct_butterfly_price_is_stable_under_refinement() -> None:
    """Check the direct butterfly estimate is stable on the two finest grids."""
    payload = json.loads(BUTTERFLY_RESULT_PATH.read_text())
    coarse, fine = payload["summary"][-2:]
    difference = abs(
        coarse["direct_mkabsde_control_variate_price_mean"]
        - fine["direct_mkabsde_control_variate_price_mean"]
    )
    assert difference < 1e-5


def test_direct_regressions_match_the_deterministic_oracle() -> None:
    """Check the retained direct prices agree with same-grid non-MC references."""
    oracle_payload = json.loads(ORACLE_PATH.read_text())
    oracle = {
        (row["product"], row["time_steps"]): row["price"]
        for row in oracle_payload["rows"]
    }
    for product, path, tolerance in (
        ("call", RESULT_PATH, 5e-5),
        ("butterfly", BUTTERFLY_RESULT_PATH, 1e-5),
    ):
        payload = json.loads(path.read_text())
        for row in payload["summary"]:
            estimate = row["direct_mkabsde_control_variate_price_mean"]
            assert abs(estimate - oracle[(product, row["time_steps"])]) < tolerance
