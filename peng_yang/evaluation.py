"""Independent grid-date and piecewise-reconstruction convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from .config import ExperimentConfig
from .problems import ExampleDefinition
from .regression import LocalRegressors, features
from .simulation import make_time_grid, simulate_brownian
from .config import torch_dtype


@dataclass(frozen=True)
class ErrorMetrics:
    """Store grid-date errors used for implementation diagnostics."""

    y_mse_sup: float
    z_mse_integrated: float
    y_rmse_global: float
    z_rmse_global: float
    y_mae_global: float
    z_mae_global: float
    y0_absolute_error: float
    z0_absolute_error: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable metric dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Bundle metrics with optional predictions retained for plotting."""

    metrics: ErrorMetrics
    time_grid: torch.Tensor
    y_mse_by_time: torch.Tensor
    z_mse_by_time: torch.Tensor
    y_prediction: torch.Tensor | None = None
    z_prediction: torch.Tensor | None = None
    y_exact: torch.Tensor | None = None
    z_exact: torch.Tensor | None = None


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Store the two squared errors appearing in the convergence theorem."""

    y_mse_sup: float
    z_mse_integrated: float


def predict_on_paths(
    models: LocalRegressors,
    example: ExampleDefinition,
    config: ExperimentConfig,
    brownian: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the trained Y and Z regressors along Brownian paths."""
    y = torch.empty_like(brownian)
    z = torch.empty_like(brownian)
    with torch.no_grad():
        for step in range(config.time_steps):
            state = brownian[:, step]
            time = grid[step].expand_as(state)
            inputs = features(time, state, config.terminal_time)
            y_model = models.at_step("Y", step)
            z_model = models.at_step("Z", step)
            y[:, step] = y_model(inputs).squeeze(-1)
            z[:, step] = z_model(inputs).squeeze(-1)
        terminal_slice = slice(config.time_steps, config.total_steps + 1)
        terminal_time = grid[terminal_slice].unsqueeze(0)
        terminal_state = brownian[:, terminal_slice]
        y[:, terminal_slice] = example.terminal_y(
            terminal_time, terminal_state, config.terminal_time
        )
        z[:, terminal_slice] = example.terminal_z(
            terminal_time, terminal_state, config.terminal_time
        )
    return y, z


def evaluate(
    models: LocalRegressors,
    example: ExampleDefinition,
    config: ExperimentConfig,
    device: torch.device | str,
    *,
    test_paths: int = 100_000,
    seed: int = 1_000_003,
    keep_paths: bool = False,
) -> EvaluationResult:
    """Compute independent grid-date errors against the closed-form solution."""
    device = torch.device(device)
    dtype = torch_dtype(config.dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    grid = make_time_grid(config, device, dtype)
    brownian, _ = simulate_brownian(test_paths, config, device, dtype, generator)
    y_prediction, z_prediction = predict_on_paths(
        models, example, config, brownian, grid
    )
    time = grid.unsqueeze(0)
    y_exact = example.exact_y(time, brownian, config.terminal_time)
    z_exact = example.exact_z(time, brownian, config.terminal_time)
    if z_exact.shape != brownian.shape:
        z_exact = z_exact.expand_as(brownian)

    y_error = y_prediction - y_exact
    z_error = z_prediction - z_exact
    y_mse_by_time = y_error.square().mean(dim=0)
    z_mse_by_time = z_error.square().mean(dim=0)
    # Left-point quadrature is the discrete counterpart of the theorem's Z norm.
    z_mse_integrated = config.dt * z_mse_by_time[:-1].sum()
    y_mse_sup = y_mse_by_time.max()
    metrics = ErrorMetrics(
        y_mse_sup=float(y_mse_sup),
        z_mse_integrated=float(z_mse_integrated),
        y_rmse_global=float(y_error.square().mean().sqrt()),
        z_rmse_global=float(z_error.square().mean().sqrt()),
        y_mae_global=float(y_error.abs().mean()),
        z_mae_global=float(z_error.abs().mean()),
        y0_absolute_error=float(y_error[:, 0].mean().abs()),
        z0_absolute_error=float(z_error[:, 0].mean().abs()),
    )
    cpu = torch.device("cpu")
    return EvaluationResult(
        metrics=metrics,
        time_grid=grid.detach().to(cpu),
        y_mse_by_time=y_mse_by_time.detach().to(cpu),
        z_mse_by_time=z_mse_by_time.detach().to(cpu),
        y_prediction=y_prediction.detach().to(cpu) if keep_paths else None,
        z_prediction=z_prediction.detach().to(cpu) if keep_paths else None,
        y_exact=y_exact.detach().to(cpu) if keep_paths else None,
        z_exact=z_exact.detach().to(cpu) if keep_paths else None,
    )


def evaluate_piecewise_reconstruction(
    models: LocalRegressors,
    example: ExampleDefinition,
    config: ExperimentConfig,
    device: torch.device | str,
    *,
    test_paths: int = 200_000,
    subdivisions: int = 8,
    seed: int = 2_000_003,
    path_batch_size: int = 25_000,
) -> ReconstructionMetrics:
    """Audit the implementable left-constant reconstruction between grid dates.

    The trained regressors provide values at grid dates. This function freezes
    their state and value over each interval, yielding the standard piecewise
    reconstruction used to visualize time convergence of discrete outputs.
    """
    device = torch.device(device)
    dtype = torch_dtype(config.dtype)
    fine_dt = config.dt / subdivisions
    fine_steps = config.total_steps * subdivisions
    y_sums = torch.zeros(
        config.time_steps * subdivisions + 1,
        device=device,
        dtype=torch.float64,
    )
    z_sum = torch.zeros((), device=device, dtype=torch.float64)
    processed = 0
    generator = torch.Generator(device=device).manual_seed(seed)

    while processed < test_paths:
        batch = min(path_batch_size, test_paths - processed)
        increments = torch.randn(
            batch, fine_steps, device=device, dtype=dtype, generator=generator
        ) * fine_dt**0.5
        brownian = torch.zeros(batch, fine_steps + 1, device=device, dtype=dtype)
        brownian[:, 1:] = increments.cumsum(dim=1)
        with torch.no_grad():
            for step in range(config.time_steps):
                left_index = step * subdivisions
                left_state = brownian[:, left_index]
                left_time = left_state.new_full(left_state.shape, step * config.dt)
                inputs = features(left_time, left_state, config.terminal_time)
                y_left = models.at_step("Y", step)(inputs).squeeze(-1)
                z_left = models.at_step("Z", step)(inputs).squeeze(-1)
                indices = torch.arange(
                    left_index,
                    left_index + subdivisions,
                    device=device,
                )
                time = indices.to(dtype).mul(fine_dt).unsqueeze(0).expand(batch, -1)
                state = brownian[:, indices]
                exact_y = example.exact_y(time, state, config.terminal_time)
                exact_z = example.exact_z(time, state, config.terminal_time)
                y_errors = (y_left.unsqueeze(1) - exact_y).double().square()
                z_errors = (z_left.unsqueeze(1) - exact_z).double().square()
                y_sums[left_index : left_index + subdivisions] += y_errors.sum(dim=0)
                z_sum += z_errors.sum() * fine_dt
            # At T the prescribed process is known exactly, hence zero error.
            y_sums[-1] += 0.0
        processed += batch

    y_mse_sup = y_sums.div(test_paths).max()
    z_mse_integrated = z_sum / test_paths
    return ReconstructionMetrics(
        y_mse_sup=float(y_mse_sup),
        z_mse_integrated=float(z_mse_integrated),
    )
