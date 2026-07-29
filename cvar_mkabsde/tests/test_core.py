"""Focused identities for the CVaR MKABSDE implementation."""

from __future__ import annotations

import math

import torch

from cvar_mkabsde.config import ExperimentConfig
from cvar_mkabsde.hermite import basis
from cvar_mkabsde.problem import (
    black_scholes_product_delta,
    black_scholes_product_value,
    butterfly_payoff,
    gaussian_cvar_constant,
)
from cvar_mkabsde.solvers import _tail_quadrature, weighted_upper_cvar


def test_delay_is_on_grid() -> None:
    """Check the article margin period maps to one step on its 50-step grid."""
    assert ExperimentConfig(time_steps=50).delay_steps == 1


def test_hermite_basis_is_normalized() -> None:
    """Check low-degree normalized probabilists' Hermite formulas."""
    x = torch.tensor([2.0], dtype=torch.float64)
    values = basis(x, 3)[0]
    expected = torch.tensor(
        [
            1.0,
            2.0,
            (4.0 - 1.0) / math.sqrt(2.0),
            (8.0 - 6.0) / math.sqrt(6.0),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(values, expected)


def test_gaussian_cvar_constant() -> None:
    """Check the standard-normal 99% CVaR constant used by the proxy drivers."""
    assert abs(gaussian_cvar_constant(0.99) - 2.6652142203) < 1e-9


def test_tail_quadrature_recovers_gaussian_cvar() -> None:
    """Check lower-tail quadrature reproduces CVaR of the symmetric Gaussian loss."""
    config = ExperimentConfig(cvar_quadrature_nodes=48)
    gaussian, weights = _tail_quadrature(
        config, torch.device("cpu"), torch.float64
    )
    estimate = float((-gaussian * weights).sum())
    assert abs(estimate - gaussian_cvar_constant(config.alpha)) < 2e-3


def test_butterfly_payoff_changes_slope() -> None:
    """Check the triangular payoff rises then falls around the middle strike."""
    spots = torch.arange(17.0, 24.0, dtype=torch.float64)
    expected = torch.tensor([0, 0, 1, 2, 1, 0, 0], dtype=torch.float64)
    torch.testing.assert_close(butterfly_payoff(spots, 20.0, 2.0), expected)


def test_weighted_cvar_selects_the_worst_probability_mass() -> None:
    """Check sorting selects the upper half of a simple discrete loss law."""
    losses = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    weights = torch.full((4,), 0.25, dtype=torch.float64)
    torch.testing.assert_close(
        weighted_upper_cvar(losses, weights, 0.5),
        torch.tensor([2.5], dtype=torch.float64),
    )


def test_article_butterfly_black_scholes_controls() -> None:
    """Check the analytic ATM butterfly controls match the article table."""
    config = ExperimentConfig(product="butterfly")
    spot = torch.tensor([config.spot], dtype=torch.float64)
    price = float(black_scholes_product_value(0.0, spot, config))
    delta = float(black_scholes_product_delta(0.0, spot, config))
    assert abs(price - 0.3090) < 5e-4
    assert abs(delta - 0.0021) < 5e-4
