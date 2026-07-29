"""Price, delta, proxy-distance, and article-benchmark diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from .config import ExperimentConfig
from .hermite import predict
from .problem import (
    black_scholes_product_delta,
    black_scholes_product_value,
)
from .solvers import SolverResult


def evaluate_solution(
    result: SolverResult, config: ExperimentConfig
) -> dict[str, float]:
    """Extract the deterministic time-zero price and hedge ratio."""
    state0 = torch.zeros(
        1,
        dtype=result.y_models[0].coefficients.dtype,
        device=result.y_models[0].coefficients.device,
    )
    price = float(predict(result.y_models[0], state0))
    z0 = float(predict(result.z_models[0], state0))
    grid = torch.linspace(
        -4.0,
        4.0,
        401,
        dtype=state0.dtype,
        device=state0.device,
    )
    increments = torch.cat(
        [torch.diff(predict(model, grid)) for model in result.y_models]
    )
    return {
        "price": price,
        "z0": z0,
        "delta": z0 / (config.volatility * config.spot),
        "minimum_y_state_increment": float(increments.min()),
        "maximum_y_state_increment": float(increments.max()),
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
    no_im_delta = metrics["no_im_control"]["delta"]
    spot0 = torch.tensor([config.spot], dtype=direct.y_values.dtype)
    analytic_price = float(black_scholes_product_value(0.0, spot0, config))
    analytic_delta = float(black_scholes_product_delta(0.0, spot0, config))
    for solver_name in ("direct_mkabsde", "nonlinear_proxy", "linear_proxy"):
        metrics[solver_name]["price_correction_vs_numerical_no_im"] = (
            metrics[solver_name]["price"] - no_im_price
        )
        metrics[solver_name]["control_variate_price"] = (
            analytic_price
            + metrics[solver_name]["price_correction_vs_numerical_no_im"]
        )
        metrics[solver_name]["control_variate_delta"] = (
            analytic_delta + metrics[solver_name]["delta"] - no_im_delta
        )
    article = (
        {
            "article_price_atm": 2.1741,
            "article_nonlinear_price_atm": 2.1959,
            "article_nonlinear_delta_atm": 0.5852,
        }
        if config.product == "call"
        else {
            "article_price_atm": 0.3090,
            "article_nonlinear_price_atm": 0.3112,
            "article_nonlinear_delta_atm": 0.0021,
        }
    )
    metrics["black_scholes_no_im"] = {
        "price": analytic_price,
        "delta": analytic_delta,
        **article,
    }
    metrics["black_scholes_no_im"]["numerical_price_bias"] = (
        no_im_price - analytic_price
    )
    return metrics
