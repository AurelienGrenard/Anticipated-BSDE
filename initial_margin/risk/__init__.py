"""Interchangeable conditional-CVaR estimators for the direct MKABSDE."""

from __future__ import annotations

import torch

from ..config import ExperimentConfig
from .common import (
    ConditionalRiskEstimator,
    RiskEstimate,
    empirical_upper_cvar,
    weighted_upper_cvar,
)
from .config import RiskEstimatorConfig, RiskMethod, paper_risk_config
from .learning import LearningRiskEstimator
from .nested import NestedMonteCarloRiskEstimator
from .quadrature import (
    QuadratureRiskEstimator,
    gaussian_law_quadrature,
    gaussian_tail_quadrature,
)


def make_risk_estimator(
    settings: RiskEstimatorConfig,
    config: ExperimentConfig,
    device: torch.device,
) -> ConditionalRiskEstimator:
    """Instantiate the requested estimator with method-independent seeding."""
    if settings.method == "quadrature":
        return QuadratureRiskEstimator(settings)
    if settings.method == "nested_mc":
        return NestedMonteCarloRiskEstimator(settings, config.seed + 200_000)
    return LearningRiskEstimator(
        settings,
        config.alpha,
        device,
        config.seed + 400_000,
    )


__all__ = [
    "ConditionalRiskEstimator",
    "LearningRiskEstimator",
    "NestedMonteCarloRiskEstimator",
    "QuadratureRiskEstimator",
    "RiskEstimate",
    "RiskEstimatorConfig",
    "RiskMethod",
    "empirical_upper_cvar",
    "gaussian_law_quadrature",
    "gaussian_tail_quadrature",
    "make_risk_estimator",
    "paper_risk_config",
    "weighted_upper_cvar",
]
