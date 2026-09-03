"""Price, MVA, delta, proxy-distance, and article-reference diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch

from hermite import predict_hermite

from .config import ExperimentConfig
from .problem import (
    black_scholes_product_delta,
    black_scholes_product_value,
)
from .solvers import SolverResult


# Values reported by Agarwal et al. for the no-IM price and the two proxy
# equations. They are presentation references, never solver inputs.
ARTICLE_REFERENCE_VALUES: dict[str, dict[str, float]] = {
    "call": {
        "no_im_price": 2.1741,
        "nonlinear_proxy_price": 2.1959,
        "linear_proxy_price": 2.1971,
        "nonlinear_proxy_delta": 0.5852,
    },
    "butterfly": {
        "no_im_price": 0.3090,
        "nonlinear_proxy_price": 0.3112,
        "linear_proxy_price": 0.3112,
        "nonlinear_proxy_delta": 0.0021,
    },
}


def evaluate_solution(
    result: SolverResult, config: ExperimentConfig
) -> dict[str, float]:
    """Extract the deterministic time-zero price and hedge ratio."""
    state0 = torch.zeros(
        1,
        dtype=result.y_models[0].coefficients.dtype,
        device=result.y_models[0].coefficients.device,
    )
    # Every path has the same state at time zero, so the stored reconstructed
    # values are the most direct representation for both full-value and
    # Black--Scholes-difference solvers.
    price = float(result.y_values[0, 0])
    z0 = float(result.z_values[0, 0])
    grid = torch.linspace(
        -4.0,
        4.0,
        401,
        dtype=state0.dtype,
        device=state0.device,
    )
    increments = []
    for step, model in enumerate(result.y_models):
        values = predict_hermite(model, grid)
        if result.models_represent_mva:
            time = step * config.dt
            if step == 0:
                spots = torch.full_like(grid, config.spot)
            else:
                spots = config.spot * torch.exp(
                    (config.drift - 0.5 * config.volatility**2) * time
                    + config.volatility * math.sqrt(time) * grid
                )
            values = values + black_scholes_product_value(
                time, spots, config
            )
        increments.append(torch.diff(values))
    state_increments = torch.cat(increments)
    return {
        "price": price,
        "z0": z0,
        "delta": z0 / (config.volatility * config.spot),
        "minimum_y_state_increment": float(state_increments.min()),
        "maximum_y_state_increment": float(state_increments.max()),
    }


def compare_results(
    results: list[SolverResult], config: ExperimentConfig
) -> dict[str, Any]:
    """Compare the three solutions on common paths and report article controls."""
    metrics = {result.name: evaluate_solution(result, config) for result in results}
    by_name = {result.name: result for result in results}
    direct = by_name["direct_mkabsde"]
    for proxy_name in ("nonlinear_proxy", "linear_proxy"):
        proxy = by_name[proxy_name]
        metrics[f"direct_vs_{proxy_name}"] = {
            "y_grid_mse_sup": float(
                (direct.y_values - proxy.y_values)
                .double()
                .square()
                .mean(dim=0)
                .max()
            ),
            "z_grid_mse_integrated": float(
                config.dt
                * (direct.z_values[:, :-1] - proxy.z_values[:, :-1])
                .double()
                .square()
                .mean()
                * config.time_steps
            ),
        }
    no_im_price = metrics["no_im_control"]["price"]
    spot0 = torch.tensor([config.spot], dtype=direct.y_values.dtype)
    analytic_price = float(black_scholes_product_value(0.0, spot0, config))
    analytic_delta = float(black_scholes_product_delta(0.0, spot0, config))
    for solver_name in ("direct_mkabsde", "nonlinear_proxy", "linear_proxy"):
        metrics[solver_name]["mva"] = (
            metrics[solver_name]["price"] - analytic_price
        )
        metrics[solver_name]["mva_percent_of_black_scholes"] = (
            100.0 * metrics[solver_name]["mva"] / analytic_price
        )
    reference = ARTICLE_REFERENCE_VALUES[config.product]
    article = {
        "article_price_atm": reference["no_im_price"],
        "article_nonlinear_price_atm": reference["nonlinear_proxy_price"],
        "article_nonlinear_delta_atm": reference["nonlinear_proxy_delta"],
    }
    metrics["black_scholes_no_im"] = {
        "price": analytic_price,
        "delta": analytic_delta,
        **article,
    }
    metrics["black_scholes_no_im"]["numerical_price_bias"] = (
        no_im_price - analytic_price
    )
    return metrics
