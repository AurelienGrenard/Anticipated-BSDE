"""Article payoffs and analytic Black-Scholes controls."""

from __future__ import annotations

import math

import torch

from .config import ExperimentConfig


def normal_cdf(value: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard Gaussian cumulative distribution function."""
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def call_payoff(spot: torch.Tensor, strike: float) -> torch.Tensor:
    """Evaluate the European call payoff."""
    return torch.relu(spot - strike)


def butterfly_payoff(
    spot: torch.Tensor, strike: float, half_width: float
) -> torch.Tensor:
    """Evaluate the symmetric call butterfly with strikes K-w, K, and K+w."""
    return (
        call_payoff(spot, strike - half_width)
        - 2.0 * call_payoff(spot, strike)
        + call_payoff(spot, strike + half_width)
    )


def product_payoff(spot: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    """Dispatch the terminal payoff selected by the experiment configuration."""
    if config.product == "call":
        return call_payoff(spot, config.strike)
    return butterfly_payoff(
        spot, config.strike, config.butterfly_half_width
    )


def _call_value_at_strike(
    time: float, spot: torch.Tensor, strike: float, config: ExperimentConfig
) -> torch.Tensor:
    """Evaluate one Black-Scholes call while allowing a strike override."""
    tau = config.maturity - time
    if tau <= 0.0:
        return call_payoff(spot, strike)
    sqrt_tau = math.sqrt(tau)
    d1 = (
        torch.log(spot / strike)
        + (config.rate + 0.5 * config.volatility**2) * tau
    ) / (config.volatility * sqrt_tau)
    d2 = d1 - config.volatility * sqrt_tau
    return spot * normal_cdf(d1) - strike * math.exp(
        -config.rate * tau
    ) * normal_cdf(d2)


def black_scholes_call(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return the no-initial-margin Black-Scholes call value."""
    return _call_value_at_strike(time, spot, config.strike, config)


def _call_z_at_strike(
    time: float, spot: torch.Tensor, strike: float, config: ExperimentConfig
) -> torch.Tensor:
    """Evaluate sigma times spot times the delta of one call."""
    tau = config.maturity - time
    if tau <= 0.0:
        return config.volatility * spot * (spot > strike).to(spot.dtype)
    d1 = (
        torch.log(spot / strike)
        + (config.rate + 0.5 * config.volatility**2) * tau
    ) / (config.volatility * math.sqrt(tau))
    return config.volatility * spot * normal_cdf(d1)


def black_scholes_z(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return Z = sigma*S*delta for the no-margin Black-Scholes call."""
    return _call_z_at_strike(time, spot, config.strike, config)


def black_scholes_delta(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return the no-margin Black-Scholes call delta."""
    return black_scholes_z(time, spot, config) / (config.volatility * spot)


def black_scholes_product_value(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return the analytic no-margin value of the selected product."""
    if config.product == "call":
        return black_scholes_call(time, spot, config)
    width = config.butterfly_half_width
    return (
        _call_value_at_strike(time, spot, config.strike - width, config)
        - 2.0 * _call_value_at_strike(time, spot, config.strike, config)
        + _call_value_at_strike(time, spot, config.strike + width, config)
    )


def black_scholes_product_z(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return analytic Z for the selected product, including signed butterfly Z."""
    if config.product == "call":
        return black_scholes_z(time, spot, config)
    width = config.butterfly_half_width
    return (
        _call_z_at_strike(time, spot, config.strike - width, config)
        - 2.0 * _call_z_at_strike(time, spot, config.strike, config)
        + _call_z_at_strike(time, spot, config.strike + width, config)
    )


def black_scholes_product_delta(
    time: float, spot: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Return analytic delta for the selected product."""
    return black_scholes_product_z(time, spot, config) / (
        config.volatility * spot
    )


def gaussian_cvar_constant(alpha: float) -> float:
    """Return CVaR_alpha of a standard normal variable."""
    probability = torch.tensor(alpha, dtype=torch.float64)
    quantile = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    density = torch.exp(-0.5 * quantile.square()) / math.sqrt(2.0 * math.pi)
    return float(density / (1.0 - alpha))
