"""Deterministic Gauss--Legendre conditional-CVaR estimator."""

from __future__ import annotations

import math

import numpy as np
import torch

from hermite import HermiteModel, fit_hermite, predict_hermite

from ..config import ExperimentConfig
from .common import (
    RiskEstimate,
    evaluate_future_value,
    project_future_value,
    weighted_upper_cvar,
)
from .config import RiskEstimatorConfig


def _quadrature_rule(
    nodes_count: int,
    alpha: float,
    complete_law: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform Gauss--Legendre nodes into a Gaussian law or lower tail."""
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    upper_uniform = 1.0 if complete_law else 1.0 - alpha
    uniforms = 0.5 * upper_uniform * (nodes + 1.0)
    gaussian = math.sqrt(2.0) * torch.erfinv(
        2.0 * torch.as_tensor(uniforms, device=device, dtype=dtype) - 1.0
    )
    normalized_weights = torch.as_tensor(
        0.5 * weights,
        device=device,
        dtype=dtype,
    )
    return gaussian, normalized_weights


def gaussian_tail_quadrature(
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalized lower-tail rule used by the monotone call."""
    return _quadrature_rule(
        config.cvar_quadrature_nodes,
        config.alpha,
        False,
        device,
        dtype,
    )


def gaussian_law_quadrature(
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nodes and probability weights for the complete Gaussian law."""
    return _quadrature_rule(
        config.cvar_quadrature_nodes,
        config.alpha,
        True,
        device,
        dtype,
    )


class QuadratureRiskEstimator:
    """Regress deterministic CVaR labels obtained by Gauss--Legendre quadrature."""

    name = "quadrature"

    def __init__(self, settings: RiskEstimatorConfig) -> None:
        """Store the converged rule size and state batching configuration."""
        self.settings = settings

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
        """Integrate the call tail directly or sort the full butterfly law."""
        del y_values, future_spot_path
        future_step = min(step + config.delay_steps, config.time_steps)
        horizon = (future_step - step) * config.dt
        complete_law = config.product == "butterfly"
        gaussian, weights = _quadrature_rule(
            self.settings.quadrature_nodes,
            config.alpha,
            complete_law,
            brownian_now.device,
            brownian_now.dtype,
        )
        targets = []
        for start in range(0, state.numel(), self.settings.state_batch_size):
            stop = min(start + self.settings.state_batch_size, state.numel())
            future_brownian = brownian_now[start:stop, None] + math.sqrt(
                horizon
            ) * gaussian[None, :]
            future_spot = spot_now[start:stop, None] * torch.exp(
                (config.drift - 0.5 * config.volatility**2) * horizon
                + config.volatility * math.sqrt(horizon) * gaussian[None, :]
            )
            future_value = evaluate_future_value(
                y_models,
                future_step,
                future_brownian.reshape(-1),
                future_spot.reshape(-1),
                config,
            ).reshape(stop - start, -1)
            future_value = project_future_value(future_value, future_spot, config)
            if config.product == "call":
                future_value = torch.cummax(future_value, dim=1).values
                gap_losses = current_bs[start:stop, None] - future_value
                targets.append((gap_losses * weights[None, :]).sum(dim=1))
            else:
                gap_losses = current_bs[start:stop, None] - future_value
                targets.append(
                    weighted_upper_cvar(gap_losses, weights, config.alpha)
                )
        target = torch.cat(targets)
        if step == 0:
            prediction = target[:1].expand_as(state)
        else:
            model = fit_hermite(
                state,
                target,
                config.hermite_degree,
                config.ridge,
            )
            prediction = predict_hermite(model, state)
        return RiskEstimate(
            prediction,
            {
                "nodes": self.settings.quadrature_nodes,
                "target_mean": float(target.mean()),
                "target_std": float(target.std()),
            },
        )
