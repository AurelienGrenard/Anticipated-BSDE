"""Deterministic one-dimensional reference for the direct CVaR MKABSDE.

This solver uses exactly the same time-discrete difference scheme and the same
Gauss--Legendre CVaR construction as the regression solver. Only conditional
functions are evaluated differently: on a log-spot grid by interpolation and
Gauss--Hermite integration, instead of Monte Carlo and Hermite regression.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri

from .config import ExperimentConfig


def _call_payoff(spot: np.ndarray, strike: float) -> np.ndarray:
    """Evaluate a call payoff on a NumPy array."""
    return np.maximum(spot - strike, 0.0)


def _call_value(
    time: float,
    spot: np.ndarray,
    strike: float,
    config: ExperimentConfig,
) -> np.ndarray:
    """Evaluate one Black--Scholes call while allowing a strike override."""
    tau = config.maturity - time
    if tau <= 0.0:
        return _call_payoff(spot, strike)
    d1 = (
        np.log(spot / strike)
        + (config.rate + 0.5 * config.volatility**2) * tau
    ) / (config.volatility * math.sqrt(tau))
    d2 = d1 - config.volatility * math.sqrt(tau)
    return spot * ndtr(d1) - strike * math.exp(-config.rate * tau) * ndtr(d2)


def _black_scholes_value(
    time: float,
    spot: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Evaluate the no-IM Black--Scholes value of the configured product."""
    if config.product == "call":
        return _call_value(time, spot, config.strike, config)
    width = config.butterfly_half_width
    return (
        _call_value(time, spot, config.strike - width, config)
        - 2.0 * _call_value(time, spot, config.strike, config)
        + _call_value(time, spot, config.strike + width, config)
    )


def _interpolate(
    grid: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one grid function on an arbitrary query array."""
    flat = np.interp(query.reshape(-1), grid, values)
    return flat.reshape(query.shape)


def _weighted_lower_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    tail_mass: float,
) -> np.ndarray:
    """Average the smallest requested probability mass of each discrete law."""
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    expanded_weights = np.broadcast_to(weights, values.shape)
    sorted_weights = np.take_along_axis(expanded_weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    previous = cumulative - sorted_weights
    used = np.minimum(
        sorted_weights,
        np.maximum(tail_mass - previous, 0.0),
    )
    return np.sum(sorted_values * used, axis=1) / tail_mass


class DeterministicReference:
    """Solve the direct equation on a fixed log-spot grid without regression."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        space_steps: int = 6_000,
        expectation_nodes: int = 24,
        cvar_nodes: int | None = None,
    ) -> None:
        """Build the spatial grid and reusable Gaussian quadrature rules."""
        self.config = config
        self.grid = np.linspace(
            math.log(config.strike / 80.0),
            math.log(16.0 * config.strike),
            space_steps + 1,
        )
        self.spots = np.exp(self.grid)

        hermite_nodes, hermite_weights = hermgauss(expectation_nodes)
        self.expectation_gaussian = math.sqrt(2.0) * hermite_nodes
        self.expectation_weights = hermite_weights / math.sqrt(math.pi)

        node_count = cvar_nodes or (64 if config.product == "call" else 256)
        legendre_nodes, legendre_weights = leggauss(node_count)
        if config.product == "call":
            tail_mass = 1.0 - config.alpha
            uniforms = 0.5 * tail_mass * (legendre_nodes + 1.0)
        else:
            uniforms = 0.5 * (legendre_nodes + 1.0)
        self.cvar_gaussian = ndtri(uniforms)
        self.cvar_weights = 0.5 * legendre_weights

    def _transition_logs(
        self,
        horizon: float,
        gaussian: np.ndarray,
    ) -> np.ndarray:
        """Return conditional future log-spots for every grid point and node."""
        drift = self.config.drift - 0.5 * self.config.volatility**2
        return (
            self.grid[:, None]
            + drift * horizon
            + self.config.volatility
            * math.sqrt(horizon)
            * gaussian[None, :]
        )

    def _conditional_moments(
        self,
        future_mva: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one-step conditional means and Brownian covariances."""
        future_logs = self._transition_logs(
            self.config.dt,
            self.expectation_gaussian,
        )
        evaluated = _interpolate(
            self.grid,
            future_mva,
            future_logs,
        )
        conditional = evaluated @ self.expectation_weights
        z = (
            evaluated
            @ (self.expectation_weights * self.expectation_gaussian)
            / math.sqrt(self.config.dt)
        )
        return conditional, z

    def _future_loss_cvar(
        self,
        future_total: np.ndarray,
        horizon: float,
    ) -> np.ndarray:
        """Compute conditional CVaR of minus the supplied future total value."""
        future_logs = self._transition_logs(horizon, self.cvar_gaussian)
        evaluated = _interpolate(self.grid, future_total, future_logs)
        if self.config.product == "call":
            evaluated = np.maximum.accumulate(evaluated, axis=1)
            return -(evaluated @ self.cvar_weights)
        evaluated = np.clip(
            evaluated,
            0.0,
            1.05 * self.config.butterfly_half_width,
        )
        return -_weighted_lower_tail_mean(
            evaluated,
            self.cvar_weights,
            1.0 - self.config.alpha,
        )

    def solve(self) -> dict[str, float | int | str]:
        """Run the backward difference scheme and return its time-zero value."""
        mvas = [
            np.empty_like(self.grid)
            for _ in range(self.config.time_steps + 1)
        ]
        mvas[-1] = np.zeros_like(self.grid)
        theta = (
            self.config.rate - self.config.drift
        ) / self.config.volatility
        denominator = 1.0 - self.config.dt * (
            self.config.funding_spread - self.config.rate
        )

        for step in range(self.config.time_steps - 1, -1, -1):
            conditional, mva_z = self._conditional_moments(
                mvas[step + 1]
            )
            future_step = min(
                step + self.config.delay_steps,
                self.config.time_steps,
            )
            future_time = future_step * self.config.dt
            future_total = _black_scholes_value(
                future_time,
                self.spots,
                self.config,
            ) + mvas[future_step]
            horizon = (future_step - step) * self.config.dt
            future_loss_cvar = self._future_loss_cvar(
                future_total,
                horizon,
            )
            current_bs = _black_scholes_value(
                step * self.config.dt,
                self.spots,
                self.config,
            )
            mvas[step] = (
                conditional
                + self.config.dt
                * (
                    theta * mva_z
                    + self.config.funding_spread
                    * (current_bs + future_loss_cvar)
                )
            ) / denominator

        base = float(
            _black_scholes_value(
                0.0,
                np.array([self.config.spot]),
                self.config,
            )[0]
        )
        mva = float(
            np.interp(
                math.log(self.config.spot),
                self.grid,
                mvas[0],
            )
        )
        return {
            "product": self.config.product,
            "time_steps": self.config.time_steps,
            "h": self.config.dt,
            "black_scholes_price": base,
            "mva": mva,
            "mva_percent_of_black_scholes": 100.0 * mva / base,
            "price": base + mva,
        }


def parse_args() -> argparse.Namespace:
    """Parse the deterministic validation grids."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the same-grid deterministic reference for the direct "
            "one-dimensional MKABSDE."
        )
    )
    parser.add_argument(
        "--product",
        choices=("all", "call", "butterfly"),
        default="all",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--space-steps", type=int, default=6_000)
    parser.add_argument("--expectation-nodes", type=int, default=24)
    parser.add_argument("--cvar-nodes", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON file receiving all validation rows",
    )
    return parser.parse_args()


def main() -> None:
    """Print one JSON validation row for every requested product and time grid."""
    args = parse_args()
    rows = []
    products = (
        ("call", "butterfly")
        if args.product == "all"
        else (args.product,)
    )
    for product in products:
        base = ExperimentConfig(product=product)
        for steps in args.steps:
            config = replace(base, time_steps=steps)
            result = DeterministicReference(
                config,
                space_steps=args.space_steps,
                expectation_nodes=args.expectation_nodes,
                cvar_nodes=args.cvar_nodes,
            ).solve()
            rows.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"rows": rows}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
