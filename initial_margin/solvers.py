"""Backward solvers for the direct MKABSDE and its two proxy BSDEs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
from .risk import (
    ConditionalRiskEstimator,
    RiskEstimatorConfig,
    make_risk_estimator,
)
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
    risk_method: str | None = None
    risk_diagnostics: list[dict[str, float | int]] = field(default_factory=list)


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
    *,
    risk_settings: RiskEstimatorConfig | None = None,
    risk_estimator: ConditionalRiskEstimator | None = None,
) -> SolverResult:
    """Solve the genuine MKABSDE with one replaceable conditional-risk engine."""
    if risk_settings is not None and risk_estimator is not None:
        raise ValueError("pass risk_settings or risk_estimator, not both")
    settings = risk_settings or RiskEstimatorConfig(
        method="quadrature",
        quadrature_nodes=config.cvar_quadrature_nodes,
        state_batch_size=config.cvar_batch_size,
    )
    estimator = risk_estimator or make_risk_estimator(
        settings,
        config,
        brownian.device,
    )
    steps = config.time_steps
    y_values = torch.empty_like(spots)
    z_values = torch.zeros_like(spots)
    y_values[:, -1] = product_payoff(spots[:, -1], config)
    next_difference = torch.zeros_like(spots[:, -1])
    y_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    z_models: list[HermiteModel] = [None] * steps  # type: ignore[list-item]
    theta = (config.rate - config.drift) / config.volatility
    denominator = 1.0 - config.dt * (config.funding_spread - config.rate)
    diagnostics: list[dict[str, float | int]] = []

    for step in range(steps - 1, -1, -1):
        state = normalized_state(brownian, step, config.dt)
        _, z_model, difference_tilde, difference_z = _common_regressions(
            next_difference, increments[:, step], state, config
        )
        current_bs = black_scholes_product_value(
            step * config.dt,
            spots[:, step],
            config,
        )
        future_step = min(step + config.delay_steps, config.time_steps)
        risk = estimator.estimate(
            y_models,
            y_values,
            step,
            brownian[:, step],
            spots[:, step],
            state,
            current_bs,
            spots[:, future_step],
            config,
        )
        diagnostics.append(
            {
                "step": step,
                "time": step * config.dt,
                **risk.diagnostics,
            }
        )
        difference_target = (
            difference_tilde
            + config.dt
            * (
                theta * difference_z
                + config.funding_spread * risk.values
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
        risk_method=estimator.name,
        risk_diagnostics=diagnostics,
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
    config: ExperimentConfig,
    device: torch.device | str = "cpu",
    *,
    risk_settings: RiskEstimatorConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[SolverResult]]:
    """Simulate one common experiment and solve all three equations with CRNs."""
    device = torch.device(device)
    brownian, increments, spots = simulate_paths(config, device)
    direct = solve_direct(
        config,
        brownian,
        increments,
        spots,
        risk_settings=risk_settings,
    )
    nonlinear = solve_proxy(
        config, brownian, increments, spots, linear=False
    )
    linear = solve_proxy(config, brownian, increments, spots, linear=True)
    no_im = solve_no_im(config, brownian, increments, spots)
    return brownian, increments, spots, [direct, nonlinear, linear, no_im]
