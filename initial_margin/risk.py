"""Conditional CVaR quadrature for monotone and non-monotone products."""

from __future__ import annotations

import math

import numpy as np
import torch

from hermite import HermiteModel, predict_hermite

from .config import ExperimentConfig
from .problem import black_scholes_product_value, product_payoff


def _evaluate_future_y(
    y_models: list[HermiteModel],
    future_step: int,
    future_brownian: torch.Tensor,
    future_spot: torch.Tensor,
    config: ExperimentConfig,
    *,
    difference_to_black_scholes: bool,
) -> torch.Tensor:
    """Evaluate a learned future value or reconstruct it from its BS difference."""
    if future_step >= config.time_steps:
        return product_payoff(future_spot, config)
    state = future_brownian / math.sqrt(future_step * config.dt)
    learned_value = predict_hermite(y_models[future_step], state)
    if not difference_to_black_scholes:
        return learned_value
    return learned_value + black_scholes_product_value(
        future_step * config.dt,
        future_spot,
        config,
    )


def gaussian_tail_quadrature(
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nodes for the lower Gaussian ``1-alpha`` probability tail."""
    nodes, weights = np.polynomial.legendre.leggauss(
        config.cvar_quadrature_nodes
    )
    tail_probability = 1.0 - config.alpha
    uniforms = 0.5 * tail_probability * (nodes + 1.0)
    gaussian = math.sqrt(2.0) * torch.erfinv(
        2.0 * torch.as_tensor(uniforms, device=device, dtype=dtype) - 1.0
    )
    normalized_weights = torch.as_tensor(
        0.5 * weights,
        device=device,
        dtype=dtype,
    )
    return gaussian, normalized_weights


def gaussian_law_quadrature(
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic nodes and weights for the complete Gaussian law."""
    nodes, weights = np.polynomial.legendre.leggauss(
        config.cvar_quadrature_nodes
    )
    uniforms = 0.5 * (nodes + 1.0)
    gaussian = math.sqrt(2.0) * torch.erfinv(
        2.0 * torch.as_tensor(uniforms, device=device, dtype=dtype) - 1.0
    )
    normalized_weights = torch.as_tensor(
        0.5 * weights,
        device=device,
        dtype=dtype,
    )
    return gaussian, normalized_weights


def weighted_upper_cvar(
    losses: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Average the worst ``1-alpha`` mass of weighted discrete loss laws."""
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


def conditional_future_loss_cvar(
    y_models: list[HermiteModel],
    step: int,
    brownian_now: torch.Tensor,
    spot_now: torch.Tensor,
    config: ExperimentConfig,
    *,
    difference_to_black_scholes: bool = False,
) -> torch.Tensor:
    """Approximate conditional future-loss CVaR at every current state.

    When ``difference_to_black_scholes`` is true, the learned models contain
    only ``V - V_BS``. The analytic Black--Scholes value is then restored
    inside the CVaR target, before sorting or selecting the loss tail.
    """
    future_step = min(step + config.delay_steps, config.time_steps)
    horizon = (future_step - step) * config.dt

    if config.product == "butterfly":
        gaussian, weights = gaussian_law_quadrature(
            config,
            brownian_now.device,
            brownian_now.dtype,
        )
        outputs = []
        for start in range(0, brownian_now.shape[0], config.cvar_batch_size):
            stop = min(start + config.cvar_batch_size, brownian_now.shape[0])
            brownian_batch = brownian_now[start:stop]
            spot_batch = spot_now[start:stop]
            future_brownian = (
                brownian_batch[:, None] + math.sqrt(horizon) * gaussian[None, :]
            )
            future_spot = spot_batch[:, None] * torch.exp(
                (config.drift - 0.5 * config.volatility**2) * horizon
                + config.volatility * math.sqrt(horizon) * gaussian[None, :]
            )
            future_y = _evaluate_future_y(
                y_models,
                future_step,
                future_brownian.reshape(-1),
                future_spot.reshape(-1),
                config,
                difference_to_black_scholes=difference_to_black_scholes,
            ).reshape(stop - start, -1)
            # A butterfly is bounded but not monotone, so both Gaussian tails
            # may enter the worst loss set. Bounds are enforced before sorting.
            future_y = future_y.clamp(
                min=0.0,
                max=1.05 * config.butterfly_half_width,
            )
            outputs.append(weighted_upper_cvar(-future_y, weights, config.alpha))
        return torch.cat(outputs)

    gaussian, weights = gaussian_tail_quadrature(
        config,
        brownian_now.device,
        brownian_now.dtype,
    )
    future_brownian = (
        brownian_now[:, None] + math.sqrt(horizon) * gaussian[None, :]
    )
    future_spot = spot_now[:, None] * torch.exp(
        (config.drift - 0.5 * config.volatility**2) * horizon
        + config.volatility * math.sqrt(horizon) * gaussian[None, :]
    )
    future_y = _evaluate_future_y(
        y_models,
        future_step,
        future_brownian.reshape(-1),
        future_spot.reshape(-1),
        config,
        difference_to_black_scholes=difference_to_black_scholes,
    ).reshape(brownian_now.shape[0], -1)
    # The true call value is non-negative, increasing in spot, and bounded by
    # spot up to the small funding correction. This projection prevents remote
    # Hermite oscillations from corrupting the targeted one-percent tail.
    future_y = torch.minimum(future_y.clamp_min(0.0), 1.05 * future_spot)
    future_y = torch.cummax(future_y, dim=1).values
    return (-future_y * weights[None, :]).sum(dim=1)
