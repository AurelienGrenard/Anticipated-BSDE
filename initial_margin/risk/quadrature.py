"""Full-law Gauss--Legendre conditional CVaR, common to both payoffs."""

from __future__ import annotations

import math

import numpy as np
import torch

from hermite import HermiteModel, fit_hermite, predict_hermite

from ..config import ExperimentConfig
from .common import RiskEstimate, evaluate_future_value, weighted_upper_cvar
from .config import RiskEstimatorConfig


def _quadrature_rule(
    nodes_count: int, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the full unit-interval Legendre rule to Gaussian nodes and probabilities."""
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    uniforms = torch.as_tensor(0.5 * (nodes + 1.0), device=device, dtype=dtype)
    gaussian = math.sqrt(2.0) * torch.erfinv(2.0 * uniforms - 1.0)
    return gaussian, torch.as_tensor(0.5 * weights, device=device, dtype=dtype)


def gaussian_law_quadrature(
    config: ExperimentConfig, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the complete Gaussian-law rule specified by the experiment."""
    return _quadrature_rule(config.cvar_quadrature_nodes, device, dtype)


class QuadratureRiskEstimator:
    """Regress BS-plus-CVaR targets from the weighted conditional loss distribution."""

    name = "quadrature"

    def __init__(self, settings: RiskEstimatorConfig) -> None:
        """Store settings, cache rules by device/dtype, and bound temporary array sizes."""
        self.settings = settings
        self.batch_size = min(
            settings.state_batch_size, max(1, 4_000_000 // settings.quadrature_nodes)
        )
        self.rules: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def estimate(
        self, y_models: list[HermiteModel], y_values: torch.Tensor, step: int,
        brownian_now: torch.Tensor, spot_now: torch.Tensor, state: torch.Tensor,
        current_bs: torch.Tensor, future_spot_path: torch.Tensor,
        config: ExperimentConfig,
    ) -> RiskEstimate:
        """Sort raw future losses with their weights, then fit the conditional risk term."""
        del y_values, future_spot_path
        future_step = min(step + config.delay_steps, config.time_steps)
        horizon = (future_step - step) * config.dt
        key = (brownian_now.device, brownian_now.dtype)
        if key not in self.rules:
            self.rules[key] = _quadrature_rule(self.settings.quadrature_nodes, *key)
        gaussian, weights = self.rules[key]
        # All paths start at the same state: its conditional target is evaluated once.
        count = 1 if step == 0 else state.numel()
        targets = []
        for start in range(0, count, self.batch_size):
            stop = min(start + self.batch_size, count)
            future_brownian = brownian_now[start:stop, None] + math.sqrt(horizon) * gaussian[None, :]
            future_spot = spot_now[start:stop, None] * torch.exp(
                (config.drift - 0.5 * config.volatility**2) * horizon
                + config.volatility * math.sqrt(horizon) * gaussian[None, :]
            )
            future_value = evaluate_future_value(
                y_models, future_step, future_brownian.reshape(-1),
                future_spot.reshape(-1), config,
            ).reshape(stop - start, -1)
            # Cash invariance gives BS_now + CVaR(-V_future), the MVA driver's target.
            losses = current_bs[start:stop, None] - future_value
            targets.append(weighted_upper_cvar(losses, weights, config.alpha))
        target = torch.cat(targets)
        if step == 0:
            prediction = target.expand_as(state)
        else:
            model = fit_hermite(state, target, config.hermite_degree, config.ridge)
            prediction = predict_hermite(model, state)
        return RiskEstimate(prediction, {
            "nodes": self.settings.quadrature_nodes,
            "target_mean": float(target.mean()),
            "target_std": float(target.std()) if count > 1 else 0.0,
        })
