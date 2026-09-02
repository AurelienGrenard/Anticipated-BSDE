"""Nested conditional Monte Carlo CVaR estimator."""

from __future__ import annotations

import math

import torch

from hermite import HermiteModel, fit_hermite, predict_hermite

from ..config import ExperimentConfig
from .common import (
    RiskEstimate,
    empirical_upper_cvar,
    evaluate_future_value,
    project_future_value,
)
from .config import RiskEstimatorConfig


class NestedMonteCarloRiskEstimator:
    """Estimate conditional CVaR from independent inner Gaussian simulations."""

    name = "nested_mc"

    def __init__(self, settings: RiskEstimatorConfig, seed: int) -> None:
        """Store the calibrated sampling controls and an independent base seed."""
        self.settings = settings
        self.seed = seed

    def _training_indices(self, count: int, device: torch.device) -> torch.Tensor:
        """Choose a deterministic representative subset of outer-path states."""
        selected = min(count, self.settings.nested_training_states)
        if selected == count:
            return torch.arange(count, device=device)
        return torch.linspace(0, count - 1, selected, device=device).long()

    def _innovations(
        self,
        rows: int,
        step: int,
        batch_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate reproducible independent and optionally antithetic innovations."""
        generator = torch.Generator(device=device).manual_seed(
            self.seed + 1_000_003 * (step + 1) + 9_973 * batch_index
        )
        samples = self.settings.nested_inner_samples
        if self.settings.nested_antithetic:
            half = samples // 2
            base = torch.randn(
                rows,
                half,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            return torch.cat((base, -base), dim=1)
        return torch.randn(
            rows,
            samples,
            device=device,
            dtype=dtype,
            generator=generator,
        )

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
        """Simulate future losses, average their upper tail, and regress the labels."""
        del y_values, future_spot_path
        future_step = min(step + config.delay_steps, config.time_steps)
        horizon = (future_step - step) * config.dt
        indices = (
            torch.zeros(1, device=state.device, dtype=torch.long)
            if step == 0
            else self._training_indices(state.numel(), state.device)
        )
        training_state = state[indices]
        targets = []
        for batch_index, start in enumerate(
            range(0, indices.numel(), self.settings.state_batch_size)
        ):
            batch_indices = indices[
                start : start + self.settings.state_batch_size
            ]
            innovation = self._innovations(
                batch_indices.numel(),
                step,
                batch_index,
                state.device,
                state.dtype,
            )
            future_brownian = brownian_now[batch_indices, None] + math.sqrt(
                horizon
            ) * innovation
            future_spot = spot_now[batch_indices, None] * torch.exp(
                (config.drift - 0.5 * config.volatility**2) * horizon
                + config.volatility * math.sqrt(horizon) * innovation
            )
            future_value = evaluate_future_value(
                y_models,
                future_step,
                future_brownian.reshape(-1),
                future_spot.reshape(-1),
                config,
            ).reshape(batch_indices.numel(), -1)
            future_value = project_future_value(future_value, future_spot, config)
            gap_losses = current_bs[batch_indices, None] - future_value
            targets.append(empirical_upper_cvar(gap_losses, config.alpha))
        target = torch.cat(targets)
        if step == 0:
            prediction = target[:1].expand_as(state)
        else:
            model = fit_hermite(
                training_state,
                target,
                config.hermite_degree,
                config.ridge,
            )
            prediction = predict_hermite(model, state)
        return RiskEstimate(
            prediction,
            {
                "training_states": indices.numel(),
                "inner_samples": self.settings.nested_inner_samples,
                "target_mean": float(target.mean()),
                "target_std": float(target.std()) if target.numel() > 1 else 0.0,
            },
        )
