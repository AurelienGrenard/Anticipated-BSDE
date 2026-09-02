"""Financial and numerical parameters for the CVaR MKABSDE experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Collect one article case and validate its delay-compatible time grid."""

    product: str = "call"
    spot: float = 20.0
    strike: float = 20.0
    butterfly_half_width: float = 2.0
    maturity: float = 1.0
    rate: float = 0.02
    drift: float = 0.02
    volatility: float = 0.25
    funding_spread: float = 0.02
    alpha: float = 0.99
    margin_period: float = 0.02
    time_steps: int = 50
    paths: int = 200_000
    seed: int = 42
    hermite_degree: int = 2
    ridge: float = 1e-10
    cvar_quadrature_nodes: int = 24
    cvar_batch_size: int = 10_000
    dtype: str = "float64"

    def __post_init__(self) -> None:
        """Reject inconsistent financial inputs and off-grid margin periods."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if min(self.spot, self.strike, self.maturity, self.volatility) <= 0.0:
            raise ValueError("spot, strike, maturity, and volatility must be positive")
        if self.product not in {"call", "butterfly"}:
            raise ValueError("product must be 'call' or 'butterfly'")
        if self.butterfly_half_width <= 0.0:
            raise ValueError("butterfly_half_width must be positive")
        if self.cvar_quadrature_nodes < 2 or self.cvar_batch_size < 1:
            raise ValueError("quadrature nodes and batch size must be positive")
        if self.hermite_degree < 0:
            raise ValueError("hermite_degree must be non-negative")
        delay = self.margin_period / self.dt
        if abs(delay - round(delay)) > 1e-10:
            raise ValueError("margin_period / dt must be an integer")

    @property
    def dt(self) -> float:
        """Return the uniform numerical time step."""
        return self.maturity / self.time_steps

    @property
    def delay_steps(self) -> int:
        """Return the margin period as an integer number of time steps."""
        return round(self.margin_period / self.dt)

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)
