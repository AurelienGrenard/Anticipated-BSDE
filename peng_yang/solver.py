"""Backward ABSDE solver using local Hermite ridge regressions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from hermite import ridge_coefficients

from .config import ExperimentConfig, torch_dtype
from .problems import ExampleDefinition
from .regression import HermiteRegressor, LocalRegressors, features
from .simulation import make_time_grid, simulate_brownian


@dataclass
class TrainingResult:
    """Contain the fitted local regressors and training metadata."""

    models: LocalRegressors
    config: ExperimentConfig
    wall_seconds: float


def fit_ridge(
    model: HermiteRegressor,
    inputs: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> None:
    """Fit every active Hermite coefficient with a stable FP64 solve."""
    design = model.basis(inputs)
    coefficients = ridge_coefficients(design, target, ridge)
    model.linear.weight.copy_(coefficients.to(model.linear.weight).unsqueeze(0))
    model.eval()


def future_value(
    models: LocalRegressors,
    example: ExampleDefinition,
    config: ExperimentConfig,
    grid: torch.Tensor,
    state: torch.Tensor,
    index: int,
    name: str,
) -> torch.Tensor:
    """Evaluate a frozen future Y or Z model, or the terminal oracle after T."""
    time = grid[index].expand_as(state)
    if index >= config.time_steps:
        terminal = example.terminal_y if name == "Y" else example.terminal_z
        return terminal(time, state, config.terminal_time)
    return models.at_step(name, index)(
        features(time, state, config.terminal_time)
    ).squeeze(-1)


def train_backward(
    config: ExperimentConfig,
    example: ExampleDefinition,
    device: torch.device | str,
    *,
    progress: bool = True,
) -> TrainingResult:
    """Train Y-tilde, Z, A, and Y from the terminal date back to zero."""
    device = torch.device(device)
    dtype = torch_dtype(config.dtype)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    grid = make_time_grid(config, device, dtype)
    brownian, increments = simulate_brownian(
        config.train_paths, config, device, dtype, generator
    )
    models = LocalRegressors(
        config.time_steps,
        config.hermite_degree,
    ).to(device=device, dtype=dtype)
    started = perf_counter()

    with torch.no_grad():
        for step in range(config.time_steps - 1, -1, -1):
            state = brownian[:, step]
            time = grid[step].expand_as(state)
            inputs = features(time, state, config.terminal_time)
            next_y = future_value(
                models,
                example,
                config,
                grid,
                brownian[:, step + 1],
                step + 1,
                "Y",
            )

            ytilde = models.at_step("Ytilde", step)
            fit_ridge(ytilde, inputs, next_y, config.ridge)
            baseline = ytilde(inputs).squeeze(-1)

            z = models.at_step("Z", step)
            z_target = (next_y - baseline) * increments[:, step] / config.dt
            fit_ridge(z, inputs, z_target, config.ridge)

            future_index = step + config.delay_steps
            future_state = brownian[:, future_index]
            if example.anticipated_kind == "future_y":
                anticipated = future_value(
                    models,
                    example,
                    config,
                    grid,
                    future_state,
                    future_index,
                    "Y",
                )
            else:
                current_z = z(inputs).squeeze(-1)
                future_z = future_value(
                    models,
                    example,
                    config,
                    grid,
                    future_state,
                    future_index,
                    "Z",
                )
                anticipated = (future_z - current_z).abs()

            a = models.at_step("A", step)
            fit_ridge(a, inputs, anticipated, config.ridge)
            y_target = baseline + config.dt * example.driver_from_a(
                time, a(inputs).squeeze(-1), config.anticipation_horizon
            )
            fit_ridge(models.at_step("Y", step), inputs, y_target, config.ridge)
            if progress:
                print(f"Completed backward step {step:04d}", flush=True)

    return TrainingResult(models, config, perf_counter() - started)
