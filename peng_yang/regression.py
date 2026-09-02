"""Configurable Hermite basis and the local linear regressors built on it."""

from __future__ import annotations

import torch
from torch import nn

from hermite import hermite_basis


class HermiteRegressor(nn.Module):
    """Regress one scalar on normalized Hermite polynomials through a degree."""

    def __init__(self, degree: int) -> None:
        """Create a bias-free linear layer for the requested Hermite basis."""
        super().__init__()
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.degree = degree
        self.linear = nn.Linear(degree + 1, 1, bias=False)

    def basis(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate normalized Hermite features, with constants only at t=0."""
        scaled_time = inputs[:, :1]
        scaled_state = inputs[:, 1:2]
        time_fraction = ((scaled_time + 1.0) / 2.0).clamp_min(0.0)
        positive_time = time_fraction > torch.finfo(inputs.dtype).eps
        state = torch.where(
            positive_time,
            scaled_state
            / time_fraction.clamp_min(torch.finfo(inputs.dtype).eps).sqrt(),
            torch.zeros_like(scaled_state),
        )
        basis = hermite_basis(state.squeeze(1), degree=self.degree)
        h0 = basis[:, :1]
        nonconstant = basis[:, 1:]
        nonconstant = torch.where(
            positive_time, nonconstant, torch.zeros_like(nonconstant)
        )
        return torch.cat((h0, nonconstant), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted scalar regression."""
        return self.linear(self.basis(inputs))


class LocalRegressors(nn.ModuleDict):
    """Four Hermite regressors at every time step."""

    names = ("Y", "Ytilde", "Z", "A")

    def __init__(self, time_steps: int, degree: int) -> None:
        """Create four independent regressors at each time step."""
        super().__init__(
            {
                self.key(name, step): HermiteRegressor(degree)
                for step in range(time_steps)
                for name in self.names
            }
        )
        self.time_steps = time_steps
        self.degree = degree

    @staticmethod
    def key(name: str, step: int) -> str:
        """Build the stable module key for one quantity and date."""
        return f"{name}_{step:04d}"

    def at_step(self, name: str, step: int) -> HermiteRegressor:
        """Return the regressor associated with one quantity and date."""
        return self[self.key(name, step)]


def features(time: torch.Tensor, state: torch.Tensor, terminal_time: float) -> torch.Tensor:
    """Return normalized time and Brownian state features."""
    scaled_time = 2.0 * time / terminal_time - 1.0
    scaled_state = state / terminal_time**0.5
    return torch.stack((scaled_time, scaled_state), dim=-1)
