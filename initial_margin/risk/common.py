"""Shared loss construction and weighted-tail primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from hermite import HermiteModel, predict_hermite

from ..config import ExperimentConfig
from ..problem import black_scholes_product_value, product_payoff


@dataclass(frozen=True)
class RiskEstimate:
    """Return conditional gap-CVaR values and compact diagnostics."""

    values: torch.Tensor
    diagnostics: dict[str, float | int]


class ConditionalRiskEstimator(Protocol):
    """Define the estimator interface consumed by the direct MKABSDE solver."""

    name: str

    def estimate(
        self,
        y_models: list[HermiteModel],
        y_values: torch.Tensor,
        step: int,
        brownian_now: torch.Tensor,
        spot_now: torch.Tensor,
        state: torch.Tensor,
        current_bs: torch.Tensor,
        future_spot_path: torch.Tensor,
        config: ExperimentConfig,
    ) -> RiskEstimate:
        """Estimate the conditional CVaR of current BS value minus future value."""
        ...


def evaluate_future_value(
    y_models: list[HermiteModel],
    future_step: int,
    future_brownian: torch.Tensor,
    future_spot: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Reconstruct the learned total future value from its BS correction."""
    if future_step >= config.time_steps:
        return product_payoff(future_spot, config)
    state = future_brownian / math.sqrt(future_step * config.dt)
    correction = predict_hermite(y_models[future_step], state)
    return correction + black_scholes_product_value(
        future_step * config.dt,
        future_spot,
        config,
    )


def project_future_value(
    value: torch.Tensor,
    future_spot: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Enforce the elementary payoff bounds used to stabilize remote states."""
    if config.product == "call":
        return torch.minimum(value.clamp_min(0.0), 1.05 * future_spot)
    return value.clamp(min=0.0, max=1.05 * config.butterfly_half_width)


def weighted_upper_cvar(
    losses: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Average exactly the largest weighted mass ``1-alpha`` of each loss law."""
    sorted_losses, order = torch.sort(losses, dim=1, descending=True)
    sorted_weights = weights.expand_as(losses).gather(1, order)
    cumulative = torch.cumsum(sorted_weights, dim=1)
    previous = cumulative - sorted_weights
    tail_mass = 1.0 - alpha
    used = torch.minimum(
        sorted_weights,
        torch.clamp(tail_mass - previous, min=0.0),
    )
    return (sorted_losses * used).sum(dim=1) / tail_mass


def empirical_upper_cvar(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute row-wise empirical CVaR with the exact fractional last sample."""
    squeeze = losses.ndim == 1
    rows = losses.unsqueeze(0) if squeeze else losses
    tail_count = (1.0 - alpha) * rows.shape[1]
    selected_count = math.ceil(tail_count)
    upper = torch.topk(
        rows,
        selected_count,
        dim=1,
        largest=True,
        sorted=True,
    ).values
    full_count = math.floor(tail_count)
    numerator = upper[:, :full_count].sum(dim=1)
    remainder = tail_count - full_count
    if remainder > 0.0:
        numerator = numerator + remainder * upper[:, full_count]
    result = numerator / tail_count
    return result.squeeze(0) if squeeze else result
