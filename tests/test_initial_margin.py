"""Focused identities for the CVaR MKABSDE implementation."""

from __future__ import annotations

import math

import torch

from initial_margin.config import ExperimentConfig
from initial_margin.problem import (
    black_scholes_product_delta,
    black_scholes_product_value,
    butterfly_payoff,
    gaussian_cvar_constant,
)
from initial_margin.risk import (
    RiskEstimatorConfig,
    empirical_upper_cvar,
    gaussian_tail_quadrature,
    weighted_upper_cvar,
)
from initial_margin.simulation import simulate_paths
from initial_margin.solvers import solve_direct, solve_proxy


def test_delay_is_on_grid() -> None:
    """Check the article margin period maps to one step on its 50-step grid."""
    assert ExperimentConfig(time_steps=50).delay_steps == 1


def test_gaussian_cvar_constant() -> None:
    """Check the standard-normal 99% CVaR constant used by the proxy drivers."""
    assert abs(gaussian_cvar_constant(0.99) - 2.6652142203) < 1e-9


def test_tail_quadrature_recovers_gaussian_cvar() -> None:
    """Check lower-tail quadrature reproduces CVaR of the symmetric Gaussian loss."""
    config = ExperimentConfig(cvar_quadrature_nodes=48)
    gaussian, weights = gaussian_tail_quadrature(
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


def test_empirical_cvar_uses_a_fractional_final_observation() -> None:
    """Check a non-integer tail count receives the exact fractional last loss."""
    losses = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    expected = torch.tensor((3.0 + 0.2 * 2.0) / 1.2, dtype=torch.float64)
    torch.testing.assert_close(empirical_upper_cvar(losses, 0.7), expected)


def test_article_butterfly_black_scholes_controls() -> None:
    """Check the analytic ATM butterfly controls match the article table."""
    config = ExperimentConfig(product="butterfly")
    spot = torch.tensor([config.spot], dtype=torch.float64)
    price = float(black_scholes_product_value(0.0, spot, config))
    delta = float(black_scholes_product_delta(0.0, spot, config))
    assert abs(price - 0.3090) < 5e-4
    assert abs(delta - 0.0021) < 5e-4


def test_difference_proxy_is_exactly_zero_without_funding_spread() -> None:
    """Check the variance-reduced proxy restores BS when the IM driver vanishes."""
    config = ExperimentConfig(
        product="butterfly",
        funding_spread=0.0,
        paths=1_000,
        time_steps=50,
    )
    brownian, increments, spots = simulate_paths(config, torch.device("cpu"))
    result = solve_proxy(
        config,
        brownian,
        increments,
        spots,
        linear=False,
    )
    spot0 = torch.tensor([config.spot], dtype=torch.float64)
    expected = black_scholes_product_value(0.0, spot0, config)
    torch.testing.assert_close(result.y_values[0, :1], expected)
    for model in result.y_models:
        torch.testing.assert_close(
            model.coefficients,
            torch.zeros_like(model.coefficients),
            atol=1e-14,
            rtol=0.0,
        )


def test_all_risk_estimators_run_inside_the_same_direct_solver() -> None:
    """Exercise quadrature, nested MC, and VaR/ES learning through one interface."""
    config = ExperimentConfig(
        product="call",
        paths=500,
        time_steps=50,
        hermite_degree=2,
    )
    brownian, increments, spots = simulate_paths(config, torch.device("cpu"))
    settings = (
        RiskEstimatorConfig(
            method="quadrature",
            quadrature_nodes=8,
            state_batch_size=500,
        ),
        RiskEstimatorConfig(
            method="nested_mc",
            nested_inner_samples=256,
            nested_training_states=128,
            state_batch_size=64,
        ),
        RiskEstimatorConfig(
            method="learning",
            learning_width=4,
            learning_hidden_layers=1,
            learning_batch_size=256,
            learning_initial_updates=16,
            learning_updates_per_date=2,
        ),
    )
    for risk_settings in settings:
        result = solve_direct(
            config,
            brownian,
            increments,
            spots,
            risk_settings=risk_settings,
        )
        assert result.risk_method == risk_settings.method
        assert len(result.risk_diagnostics) == config.time_steps
        assert math.isfinite(float(result.y_values[0, 0]))
