"""Certification checks for the retained ATM-call numerical experiment."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "atm_call_experiment.json"
)


def test_retained_proxy_reproduces_article_atm_call() -> None:
    """Check the finest non-linear proxy price agrees with the article benchmark."""
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
