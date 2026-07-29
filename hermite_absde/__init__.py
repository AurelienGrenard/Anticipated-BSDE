"""Public interface of the minimal local Hermite-3 ABSDE solver."""

from .config import ExperimentConfig
from .convergence import (
    ErrorMetrics,
    EvaluationResult,
    ReconstructionMetrics,
    evaluate,
    evaluate_piecewise_reconstruction,
)
from .problems import ExampleDefinition, get_example
from .solver import TrainingResult, train_backward

__all__ = [
    "ErrorMetrics",
    "EvaluationResult",
    "ExampleDefinition",
    "ExperimentConfig",
    "ReconstructionMetrics",
    "TrainingResult",
    "evaluate",
    "evaluate_piecewise_reconstruction",
    "get_example",
    "train_backward",
]
