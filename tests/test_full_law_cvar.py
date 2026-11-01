"""Distribution-level checks for the common conditional-CVaR construction."""

import numpy as np
import pytest
import torch

from initial_margin.config import ExperimentConfig
from initial_margin.deterministic_reference import DeterministicReference, _interpolate
from initial_margin.risk import nested, quadrature
from initial_margin.risk.common import weighted_upper_cvar
from initial_margin.risk.config import RiskEstimatorConfig, paper_risk_config
from initial_margin.risk.learning import LearningDiagnostics, LearningRiskEstimator


def test_weighted_tail_equals_variational_minimum() -> None:
    """Check RU minimization over loss atoms, including a fractional final weight."""
    losses = torch.tensor([[3., -2., 1., 8.], [1., 4., 4., -1.]], dtype=torch.float64)
    weights = torch.tensor([.1, .3, .4, .2], dtype=torch.float64)
    alpha = .75
    objective = losses + (
        (losses[:, None, :] - losses[:, :, None]).clamp_min(0) * weights
    ).sum(dim=2) / (1-alpha)
    torch.testing.assert_close(
        weighted_upper_cvar(losses, weights, alpha), objective.min(dim=1).values
    )


def test_weighted_cvar_cash_invariance() -> None:
    """Verify adding the known current BS value before computing CVaR is equivalent."""
    losses = torch.tensor([[3., -2., 1., 8.], [1., 4., 4., -1.]], dtype=torch.float64)
    weights = torch.tensor([.1, .3, .4, .2], dtype=torch.float64)
    shift = torch.tensor([2., 7.], dtype=torch.float64)
    torch.testing.assert_close(
        weighted_upper_cvar(losses+shift[:, None], weights, .99),
        shift+weighted_upper_cvar(losses, weights, .99),
    )


@pytest.mark.parametrize("product", ["call", "butterfly"])
def test_raw_future_values_determine_the_loss_law(product, monkeypatch) -> None:
    """Preserve every supplied value, including negative and nonmonotone examples."""
    config = ExperimentConfig(product=product, time_steps=50, paths=3)
    engine = quadrature.QuadratureRiskEstimator(RiskEstimatorConfig(quadrature_nodes=4))
    future = torch.tensor([-1., 2., -3., 1.], dtype=torch.float64)
    monkeypatch.setattr(quadrature, "evaluate_future_value", lambda *args: future)
    zero = torch.zeros(3, dtype=torch.float64)
    actual = engine.estimate([], None, 0, zero, zero+20, zero, zero+.5, None, config)
    _, weights = quadrature._quadrature_rule(4, torch.device("cpu"), torch.float64)
    expected = weighted_upper_cvar((.5-future)[None, :], weights, config.alpha)
    torch.testing.assert_close(actual.values, expected.expand(3))


@pytest.mark.parametrize("product", ["call", "butterfly"])
def test_grid_reference_uses_the_same_raw_weighted_law(product) -> None:
    """Check the reference computes the CVaR of its interpolated future values."""
    config = ExperimentConfig(product=product, time_steps=50)
    engine = DeterministicReference(config, space_steps=100, cvar_nodes=32)
    future = np.sin(4*engine.grid)-.5
    logs = engine._transition_logs(.02, engine.cvar_gaussian)
    values = _interpolate(engine.grid, future, logs)
    expected = weighted_upper_cvar(torch.from_numpy(-values),
                                   torch.from_numpy(engine.cvar_weights), config.alpha)
    np.testing.assert_allclose(engine._future_loss_cvar(future, .02), expected.numpy(), atol=1e-12)


def test_paper_settings_are_shared_by_both_payoffs() -> None:
    """Prevent drift between experiment defaults and calibrated risk settings."""
    for product in ("call", "butterfly"):
        assert paper_risk_config("quadrature", product).quadrature_nodes == 256
        assert ExperimentConfig(product=product).cvar_quadrature_nodes == 256


@pytest.mark.parametrize("product", ["call", "butterfly"])
def test_nested_estimator_preserves_raw_future_values(product, monkeypatch) -> None:
    """Check nested sampling uses the supplied loss law without modifying its atoms."""
    config = ExperimentConfig(product=product, time_steps=50, paths=3)
    settings = RiskEstimatorConfig(method="nested_mc", nested_inner_samples=4)
    engine = nested.NestedMonteCarloRiskEstimator(settings, seed=42)
    future = torch.tensor([-1., 2., -3., 1.], dtype=torch.float64)
    monkeypatch.setattr(nested, "evaluate_future_value", lambda *args: future)
    zero = torch.zeros(3, dtype=torch.float64)
    actual = engine.estimate([], None, 0, zero, zero+20, zero, zero+.5, None, config)
    torch.testing.assert_close(actual.values, zero+3.5)


@pytest.mark.parametrize("product", ["call", "butterfly"])
def test_learning_targets_preserve_raw_future_values(product, monkeypatch) -> None:
    """Check the learner receives the realized raw future-value losses at each state."""
    config = ExperimentConfig(product=product, time_steps=50, paths=3)
    settings = RiskEstimatorConfig(method="learning", learning_width=4, learning_hidden_layers=1)
    engine = LearningRiskEstimator(settings, config.alpha, torch.device("cpu"), seed=42)
    monkeypatch.setattr(engine.learner, "fit_predict",
                        lambda state, target, step: (target, LearningDiagnostics(0, 0., 0., 1.)))
    future = torch.tensor([-1., 2., -3.], dtype=torch.float64)
    values = torch.zeros((3, 51), dtype=torch.float64)
    values[:, 2] = future
    zero = torch.zeros(3, dtype=torch.float64)
    actual = engine.estimate([], values, 1, zero, zero+20, zero, zero+.5, None, config)
    torch.testing.assert_close(actual.values, .5-future)
