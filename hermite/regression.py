"""Normalized Hermite features and stable scalar ridge regressions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HermiteModel:
    """Store the coefficients of one local scalar Hermite regression."""

    coefficients: torch.Tensor

    @property
    def degree(self) -> int:
        """Return the largest Hermite degree represented by the model."""
        return self.coefficients.numel() - 1


def hermite_basis(state: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate normalized probabilists' Hermite polynomials up to ``degree``.

    The returned columns are orthonormal under a standard Gaussian law. The
    recurrence avoids hard-coding a particular degree and is shared by both
    numerical experiments.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    columns = [torch.ones_like(state)]
    if degree == 0:
        return columns[0].unsqueeze(-1)
    columns.append(state)
    for order in range(1, degree):
        next_column = (
            state * columns[order] - math.sqrt(order) * columns[order - 1]
        ) / math.sqrt(order + 1)
        columns.append(next_column)
    return torch.stack(columns, dim=-1)


def ridge_coefficients(
    design: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Return scalar ridge coefficients using FP64 normal equations."""
    if design.ndim != 2:
        raise ValueError("design must be a two-dimensional matrix")
    response = target.reshape(-1).double()
    if design.shape[0] != response.shape[0]:
        raise ValueError("design and target must contain the same observations")
    design64 = design.double()
    gram = design64.T @ design64
    penalty = ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(
        gram + penalty,
        design64.T @ response,
    )
    return coefficients.to(dtype=design.dtype)


def fit_hermite(
    state: torch.Tensor,
    target: torch.Tensor,
    degree: int,
    ridge: float,
) -> HermiteModel:
    """Fit one local conditional expectation on the Hermite basis."""
    coefficients = ridge_coefficients(
        hermite_basis(state, degree),
        target,
        ridge,
    )
    return HermiteModel(coefficients)


def predict_hermite(model: HermiteModel, state: torch.Tensor) -> torch.Tensor:
    """Evaluate a fitted Hermite regression at normalized states."""
    return hermite_basis(state, model.degree) @ model.coefficients
