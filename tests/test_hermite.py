"""Unit tests for the numerical primitives shared by both experiments."""

from __future__ import annotations

import math

import torch

from hermite import fit_hermite, hermite_basis, predict_hermite


def test_normalized_hermite_formulas() -> None:
    """Check the four normalized probabilists' Hermite polynomials used here."""
    state = torch.tensor([2.0], dtype=torch.float64)
    values = hermite_basis(state, degree=3)[0]
    expected = torch.tensor(
        [
            1.0,
            2.0,
            (4.0 - 1.0) / math.sqrt(2.0),
            (8.0 - 6.0) / math.sqrt(6.0),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(values, expected)


def test_ridge_regression_recovers_an_exact_hermite_expansion() -> None:
    """Check a noiseless polynomial target is recovered by the common fitter."""
    state = torch.linspace(-2.5, 2.5, 101, dtype=torch.float64)
    expected = torch.tensor([1.5, -0.7, 0.25, 0.1], dtype=torch.float64)
    target = hermite_basis(state, degree=3) @ expected
    model = fit_hermite(state, target, degree=3, ridge=0.0)
    torch.testing.assert_close(model.coefficients, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(predict_hermite(model, state), target)


def test_negative_degree_is_rejected() -> None:
    """Check invalid polynomial degrees fail before a regression is attempted."""
    state = torch.zeros(2)
    try:
        hermite_basis(state, degree=-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("a negative Hermite degree must be rejected")
