"""Normalized Hermite basis and stable local ridge regressions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HermiteModel:
    """Store the coefficients of one local conditional regression."""

    coefficients: torch.Tensor


def basis(state: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate normalized probabilists' Hermite polynomials through a recurrence."""
    columns = [torch.ones_like(state)]
    if degree == 0:
        return columns[0][:, None]
    columns.append(state)
    for order in range(1, degree):
        next_column = (
            state * columns[order] - math.sqrt(order) * columns[order - 1]
        ) / math.sqrt(order + 1)
        columns.append(next_column)
    return torch.stack(columns, dim=1)


def fit(
    state: torch.Tensor, target: torch.Tensor, degree: int, ridge: float
) -> HermiteModel:
    """Fit one conditional expectation by an FP64 ridge solve."""
    design = basis(state, degree).double()
    response = target.double()
    gram = design.T @ design
    penalty = ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(gram + penalty, design.T @ response)
    return HermiteModel(coefficients.to(state.dtype))


def predict(model: HermiteModel, state: torch.Tensor) -> torch.Tensor:
    """Evaluate a fitted local Hermite model."""
    return basis(state, model.coefficients.numel() - 1) @ model.coefficients
