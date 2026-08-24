"""Central experiment parameters and time-grid consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Collect one Peng--Yang experiment and its reproducibility settings."""

    example: str
    terminal_time: float = 1.0
    anticipation_horizon: float = 0.2
    time_steps: int = 20
    train_paths: int = 1_000_000
    ridge: float = 1e-10
    seed: int = 42
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Validate that the anticipation delay lies on the time grid."""
        delay_steps = self.anticipation_horizon / self.dt
        if abs(delay_steps - round(delay_steps)) > 1e-10:
            raise ValueError("anticipation_horizon/dt must be an integer")

    @property
    def dt(self) -> float:
        """Return the uniform time step."""
        return self.terminal_time / self.time_steps

    @property
    def delay_steps(self) -> int:
        """Return the anticipation delay as a number of grid steps."""
        return round(self.anticipation_horizon / self.dt)

    @property
    def total_steps(self) -> int:
        """Return the number of steps on [0, T + delta]."""
        return self.time_steps + self.delay_steps

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the configuration."""
        return asdict(self)


def torch_dtype(name: str):
    """Translate the configured floating-point name into a PyTorch dtype."""
    import torch

    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")
