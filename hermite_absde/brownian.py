"""Uniform time-grid construction and Brownian-path simulation."""

from __future__ import annotations

import torch

from .config import ExperimentConfig


def make_time_grid(
    config: ExperimentConfig, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build the uniform grid on the full interval [0, T + delta]."""
    return torch.linspace(
        0.0,
        config.terminal_time + config.anticipation_horizon,
        config.total_steps + 1,
        device=device,
        dtype=dtype,
    )


def simulate_brownian(
    paths: int,
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate Brownian paths and retain their independent increments."""
    dt = config.dt
    increments = torch.randn(
        paths,
        config.total_steps,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * dt**0.5
    brownian = torch.zeros(
        paths, config.total_steps + 1, device=device, dtype=dtype
    )
    brownian[:, 1:] = increments.cumsum(dim=1)
    return brownian, increments
