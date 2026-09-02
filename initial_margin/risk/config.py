"""Validated settings for the three conditional-CVaR estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RiskMethod = Literal["quadrature", "nested_mc", "learning"]


@dataclass(frozen=True)
class RiskEstimatorConfig:
    """Collect the numerical controls of one conditional-risk estimator."""

    method: RiskMethod = "quadrature"
    quadrature_nodes: int = 64
    state_batch_size: int = 2_048
    nested_inner_samples: int = 32_768
    nested_training_states: int = 2_048
    nested_antithetic: bool = True
    learning_width: int = 32
    learning_hidden_layers: int = 3
    learning_batch_size: int = 131_072
    learning_initial_updates: int = 2_048
    learning_updates_per_date: int = 128
    learning_rate: float = 1e-3
    learning_es_ridge: float = 1e-5
    learning_dtype: str = "float32"

    def __post_init__(self) -> None:
        """Reject settings that cannot define the requested estimator."""
        if self.method not in {"quadrature", "nested_mc", "learning"}:
            raise ValueError(f"unknown conditional-risk method {self.method!r}")
        positive = {
            "quadrature_nodes": self.quadrature_nodes,
            "state_batch_size": self.state_batch_size,
            "nested_inner_samples": self.nested_inner_samples,
            "nested_training_states": self.nested_training_states,
            "learning_width": self.learning_width,
            "learning_hidden_layers": self.learning_hidden_layers,
            "learning_batch_size": self.learning_batch_size,
            "learning_initial_updates": self.learning_initial_updates,
            "learning_updates_per_date": self.learning_updates_per_date,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.nested_antithetic and self.nested_inner_samples % 2:
            raise ValueError("antithetic nested samples require an even count")
        if self.learning_rate <= 0.0 or self.learning_es_ridge < 0.0:
            raise ValueError("learning rate must be positive and ridge non-negative")
        if self.learning_dtype not in {"float32", "float64"}:
            raise ValueError("learning_dtype must be float32 or float64")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the settings."""
        return asdict(self)


def paper_risk_config(method: RiskMethod, product: str) -> RiskEstimatorConfig:
    """Return the calibrated method and product settings used in the paper audit."""
    if method == "quadrature":
        return RiskEstimatorConfig(
            method=method,
            quadrature_nodes=24 if product == "call" else 64,
            state_batch_size=10_000,
        )
    if method == "nested_mc":
        return RiskEstimatorConfig(
            method=method,
            nested_inner_samples=32_768,
            nested_training_states=2_048,
            state_batch_size=32,
            nested_antithetic=True,
        )
    return RiskEstimatorConfig(method=method)
