"""Backward solvers for the direct MKABSDE and its two proxy BSDEs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .hermite import HermiteModel, fit, predict
from .problem import (
    black_scholes_product_z,
    gaussian_cvar_constant,
    product_payoff,
)
from .simulation import normalized_state, simulate_paths


@dataclass
class SolverResult:
    """Contain local Y/Z models and values on the training trajectories."""

    name: str
    y_models: list[HermiteModel]
    z_models: list[HermiteModel]
    y_values: torch.Tensor
    z_values: torch.Tensor


def _evaluate_future_y(
    y_models: list[HermiteModel],
    future_step: int,
    future_brownian: torch.Tensor,
    future_spot: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Evaluate a learned future value or the terminal payoff."""
    if future_step >= config.time_steps:
        return product_payoff(future_spot, config)
    state = future_brownian / math.sqrt(future_step * config.dt)
    return predict(y_models[future_step], state)


def _tail_quadrature(
    config: ExperimentConfig, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build nodes for an expectation over the lower Gaussian 1-alpha tail."""
    nodes, weights = np.polynomial.legendre.leggauss(
        config.cvar_quadrature_nodes
    )
    tail_probability = 1.0 - config.alpha
    uniforms = 0.5 * tail_probability * (nodes + 1.0)
    normalized_weights = 0.5 * weights
    probabilities = torch.as_tensor(uniforms, device=device, dtype=dtype)
    gaussian = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    return gaussian, torch.as_tensor(
        normalized_weights, device=device, dtype=dtype
    )


def _normal_quadrature(
    config: ExperimentConfig, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic nodes and weights for the complete Gaussian law."""
    nodes, weights = np.polynomial.legendre.leggauss(
        config.cvar_quadrature_nodes
    )
    probabilities = 0.5 * (nodes + 1.0)
    gaussian = math.sqrt(2.0) * torch.erfinv(
        2.0 * torch.as_tensor(probabilities, device=device, dtype=dtype) - 1.0
    )
    return gaussian, torch.as_tensor(
        0.5 * weights, device=device, dtype=dtype
    )


def weighted_upper_cvar(
    losses: torch.Tensor, weights: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Average the worst 1-alpha mass after sorting arbitrary weighted losses."""
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
) -> torch.Tensor:
    """Compute conditional future-loss CVaR, with no monotonicity assumption."""
    future_step = min(step + config.delay_steps, config.time_steps)
    tau = (future_step - step) * config.dt
    if config.product == "butterfly":
        gaussian, weights = _normal_quadrature(
            config, brownian_now.device, brownian_now.dtype
        )
        outputs = []
        for start in range(0, brownian_now.shape[0], config.cvar_batch_size):
            stop = min(start + config.cvar_batch_size, brownian_now.shape[0])
            brownian_batch = brownian_now[start:stop]
            spot_batch = spot_now[start:stop]
            future_brownian = brownian_batch[:, None] + math.sqrt(tau) * gaussian[None, :]
            future_spot = spot_batch[:, None] * torch.exp(
                (config.drift - 0.5 * config.volatility**2) * tau
                + config.volatility * math.sqrt(tau) * gaussian[None, :]
            )
            future_y = _evaluate_future_y(
                y_models,
                future_step,
                future_brownian.reshape(-1),
                future_spot.reshape(-1),
                config,
            ).reshape(stop - start, -1)
            # A butterfly is non-negative and bounded by its half-width, but it
            # is not monotone: both Gaussian tails may belong to the worst loss
            # set. We only enforce bounds and then sort the complete law.
            future_y = future_y.clamp(
                min=0.0, max=1.05 * config.butterfly_half_width
            )
            outputs.append(weighted_upper_cvar(-future_y, weights, config.alpha))
        return torch.cat(outputs)

    gaussian, weights = _tail_quadrature(
        config, brownian_now.device, brownian_now.dtype
    )
    future_brownian = brownian_now[:, None] + math.sqrt(tau) * gaussian[None, :]
    future_spot = spot_now[:, None] * torch.exp(
        (config.drift - 0.5 * config.volatility**2) * tau
        + config.volatility * math.sqrt(tau) * gaussian[None, :]
    )
    future_y = _evaluate_future_y(
        y_models,
        future_step,
        future_brownian.reshape(-1),
        future_spot.reshape(-1),
        config,
    ).reshape(brownian_now.shape[0], -1)
    # Hermite polynomials are unconstrained and may oscillate in the extreme 1%
    # tail queried by CVaR. The true call value is non-negative, increasing in
    # spot, and bounded by spot up to the small IM funding correction. Projecting
    # only these tail evaluations enforces the financial shape used to identify
    # the lower Gaussian tail as the loss tail.
    future_y = torch.minimum(future_y.clamp_min(0.0), 1.05 * future_spot)
    future_y = torch.cummax(future_y, dim=1).values
    return (-future_y * weights[None, :]).sum(dim=1)


def _common_regressions(
    next_y: torch.Tensor,
    increment: torch.Tensor,
    state: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[HermiteModel, HermiteModel, torch.Tensor, torch.Tensor]:
    """Fit the shared Y-tilde and Z conditional regressions."""
    ytilde_model = fit(state, next_y, config.hermite_degree, config.ridge)
    ytilde = predict(ytilde_model, state)
    z_target = (next_y - ytilde) * increment / config.dt
    z_model = fit(state, z_target, config.hermite_degree, config.ridge)
    return ytilde_model, z_model, ytilde, predict(z_model, state)


def solve_direct(
    config: ExperimentConfig,
    brownian: torch.Tensor,
    increments: torch.Tensor,
    spots: torch.Tensor,
) -> SolverResult:
    """Solve the genuine CVaR MKABSDE with an exact scalar implicit update."""
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, ytilde, z = _common_regressions(
            y_values[:, step + 1], increments[:, step], state, config
        )
        cvar_target = conditional_future_loss_cvar(
            y_models,
            step,
            brownian[:, step],
            spots[:, step],
            config,
        )
        cvar_model = fit(
            state, cvar_target, config.hermite_degree, config.ridge
        )
        future_loss_cvar = predict(cvar_model, state)
        denominator = 1.0 - config.dt * (
            config.funding_spread - config.rate
        )
        y_target = (
            ytilde
            + config.dt
            * (
                theta * z
                + config.funding_spread * future_loss_cvar
            )
        ) / denominator
        y_model = fit(state, y_target, config.hermite_degree, config.ridge)
        y_models[step] = y_model
        z_models[step] = z_model
        y_values[:, step] = predict(y_model, state)
        z_values[:, step] = z
    return SolverResult("direct_mkabsde", y_models, z_models, y_values, z_values)


def solve_proxy(
    config: ExperimentConfig,
    brownian: torch.Tensor,
    increments: torch.Tensor,
    spots: torch.Tensor,
    *,
    linear: bool,
) -> SolverResult:
    """Solve the non-linear endogenous-Z or linear Black-Scholes-Z proxy."""
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility
    cvar_constant = gaussian_cvar_constant(config.alpha)

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, ytilde, z = _common_regressions(
            y_values[:, step + 1], increments[:, step], state, config
        )
        time = step * config.dt
        remaining_margin = min(config.margin_period, config.maturity - time)
        source_z = (
            black_scholes_product_z(time, spots[:, step], config)
            if linear
            else z
        )
        source = (
            config.funding_spread
            * cvar_constant
            * math.sqrt(remaining_margin)
            * source_z.abs()
        )
        y_target = (ytilde + config.dt * (theta * z + source)) / (
            1.0 + config.rate * config.dt
        )
        y_model = fit(state, y_target, config.hermite_degree, config.ridge)
        y_models[step] = y_model
        z_models[step] = z_model
        y_values[:, step] = predict(y_model, state)
        z_values[:, step] = z
    name = "linear_proxy" if linear else "nonlinear_proxy"
    return SolverResult(name, y_models, z_models, y_values, z_values)


def solve_no_im(
    config: ExperimentConfig,
    brownian: torch.Tensor,
    increments: torch.Tensor,
    spots: torch.Tensor,
) -> SolverResult:
    """Solve the no-initial-margin BSDE as a numerical bias control."""
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, ytilde, z = _common_regressions(
            y_values[:, step + 1], increments[:, step], state, config
        )
        y_target = (ytilde + config.dt * theta * z) / (
            1.0 + config.rate * config.dt
        )
        y_model = fit(state, y_target, config.hermite_degree, config.ridge)
        y_models[step] = y_model
        z_models[step] = z_model
        y_values[:, step] = predict(y_model, state)
        z_values[:, step] = z
    return SolverResult("no_im_control", y_models, z_models, y_values, z_values)


def solve_all(
    config: ExperimentConfig, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[SolverResult]]:
    """Simulate one common experiment and solve all three equations with CRNs."""
    device = torch.device(device)
    brownian, increments, spots = simulate_paths(config, device)
    direct = solve_direct(config, brownian, increments, spots)
    nonlinear = solve_proxy(
        config, brownian, increments, spots, linear=False
    )
    linear = solve_proxy(config, brownian, increments, spots, linear=True)
    no_im = solve_no_im(config, brownian, increments, spots)
    return brownian, increments, spots, [direct, nonlinear, linear, no_im]
