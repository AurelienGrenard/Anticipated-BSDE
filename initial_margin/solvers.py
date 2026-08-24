"""Backward solvers for the direct MKABSDE and its two proxy BSDEs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hermite import (
    HermiteModel,
    fit_hermite,
    predict_hermite,
)

from .config import ExperimentConfig
from .problem import (
    black_scholes_product_value,
    black_scholes_product_z,
    gaussian_cvar_constant,
    product_payoff,
)
from .risk import conditional_future_loss_cvar
from .simulation import normalized_state, simulate_paths


@dataclass
class SolverResult:
    """Contain local Y/Z models and values on the training trajectories.

    For variance-reduced solvers, the models represent differences from the
    analytic Black--Scholes solution while the stored trajectory values are
    always the reconstructed total values.
    """

    name: str
    y_models: list[HermiteModel]
    z_models: list[HermiteModel]
    y_values: torch.Tensor
    z_values: torch.Tensor
    difference_to_black_scholes: bool = False


def _common_regressions(
    next_y: torch.Tensor,
    increment: torch.Tensor,
    state: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[HermiteModel, HermiteModel, torch.Tensor, torch.Tensor]:
    """Fit the shared Y-tilde and Z conditional regressions."""
    ytilde_model = fit_hermite(state, next_y, config.hermite_degree, config.ridge)
    ytilde = predict_hermite(ytilde_model, state)
    z_target = (next_y - ytilde) * increment / config.dt
    z_model = fit_hermite(state, z_target, config.hermite_degree, config.ridge)
    return ytilde_model, z_model, ytilde, predict_hermite(z_model, state)


def solve_direct(
    config: ExperimentConfig,
    brownian: torch.Tensor,
    increments: torch.Tensor,
    spots: torch.Tensor,
) -> SolverResult:
    """Solve the genuine CVaR MKABSDE as a difference from Black--Scholes."""
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    next_difference = torch.zeros_like(spots[:, -1])
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, difference_tilde, difference_z = _common_regressions(
            next_difference, increments[:, step], state, config
        )
        cvar_target = conditional_future_loss_cvar(
            y_models,
            step,
            brownian[:, step],
            spots[:, step],
            config,
            difference_to_black_scholes=True,
        )
        cvar_model = fit_hermite(
            state, cvar_target, config.hermite_degree, config.ridge
        )
        future_loss_cvar = predict_hermite(cvar_model, state)
        denominator = 1.0 - config.dt * (
            config.funding_spread - config.rate
        )
        current_bs = black_scholes_product_value(
            step * config.dt,
            spots[:, step],
            config,
        )
        difference_target = (
            difference_tilde
            + config.dt
            * (
                theta * difference_z
                + config.funding_spread
                * (current_bs + future_loss_cvar)
            )
        ) / denominator
        y_model = fit_hermite(
            state,
            difference_target,
            config.hermite_degree,
            config.ridge,
        )
        y_models[step] = y_model
        z_models[step] = z_model
        next_difference = predict_hermite(y_model, state)
        y_values[:, step] = current_bs + next_difference
        z_values[:, step] = black_scholes_product_z(
            step * config.dt,
            spots[:, step],
            config,
        ) + difference_z
    return SolverResult(
        "direct_mkabsde",
        y_models,
        z_models,
        y_values,
        z_values,
        difference_to_black_scholes=True,
    )


def solve_proxy(
    config: ExperimentConfig,
    brownian: torch.Tensor,
    increments: torch.Tensor,
    spots: torch.Tensor,
    *,
    linear: bool,
) -> SolverResult:
    """Solve either proxy as a variance-reduced Black--Scholes difference."""
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    next_difference = torch.zeros_like(spots[:, -1])
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility
    cvar_constant = gaussian_cvar_constant(config.alpha)

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, difference_tilde, difference_z = _common_regressions(
            next_difference, increments[:, step], state, config
        )
        time = step * config.dt
        remaining_margin = min(config.margin_period, config.maturity - time)
        black_scholes_z = black_scholes_product_z(
            time, spots[:, step], config
        )
        source_z = (
            black_scholes_z
            if linear
            else difference_z + black_scholes_z
        )
        source = (
            config.funding_spread
            * cvar_constant
            * math.sqrt(remaining_margin)
            * source_z.abs()
        )
        difference_target = (
            difference_tilde
            + config.dt * (theta * difference_z + source)
        ) / (1.0 + config.rate * config.dt)
        y_model = fit_hermite(
            state,
            difference_target,
            config.hermite_degree,
            config.ridge,
        )
        y_models[step] = y_model
        z_models[step] = z_model
        next_difference = predict_hermite(y_model, state)
        y_values[:, step] = black_scholes_product_value(
            time, spots[:, step], config
        ) + next_difference
        z_values[:, step] = black_scholes_z + difference_z
    name = "linear_proxy" if linear else "nonlinear_proxy"
    return SolverResult(
        name,
        y_models,
        z_models,
        y_values,
        z_values,
        difference_to_black_scholes=True,
    )


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
        y_model = fit_hermite(state, y_target, config.hermite_degree, config.ridge)
        y_models[step] = y_model
        z_models[step] = z_model
        y_values[:, step] = predict_hermite(y_model, state)
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
