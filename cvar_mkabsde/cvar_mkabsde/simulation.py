"""Exact geometric-Brownian simulation and normalized Markov states."""

from __future__ import annotations

import math

import torch

from .config import ExperimentConfig


def simulate_paths(
    config: ExperimentConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate Brownian paths and exact GBM spots on the configured grid."""
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    generator = torch.Generator(device=device).manual_seed(config.seed)
    increments = torch.randn(
        config.paths,
        config.time_steps,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * math.sqrt(config.dt)
    brownian = torch.zeros(
        config.paths,
        config.time_steps + 1,
        device=device,
        dtype=dtype,
    )
    brownian[:, 1:] = increments.cumsum(dim=1)
    times = torch.arange(
        config.time_steps + 1, device=device, dtype=dtype
    ) * config.dt
    spots = config.spot * torch.exp(
        (config.drift - 0.5 * config.volatility**2) * times[None, :]
        + config.volatility * brownian
    )
    return brownian, increments, spots


def normalized_state(
    brownian: torch.Tensor, step: int, dt: float
) -> torch.Tensor:
    """Return W_t/sqrt(t), with the unique deterministic state at time zero."""
    if step == 0:
        return torch.zeros_like(brownian[:, 0])
    return brownian[:, step] / math.sqrt(step * dt)
