"""Persistence and backward-compatible loading of compact solver checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, torch_dtype
from .regression import LocalRegressors
from .solver import TrainingResult


def save_result(result: TrainingResult, path: str | Path) -> None:
    """Save a trained local model and its reproducibility configuration."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "absde-hermite-v2",
            "config": result.config.as_dict(),
            "state_dict": result.models.state_dict(),
            "wall_seconds": result.wall_seconds,
        },
        target,
    )


def _config(data: dict[str, Any], checkpoint_format: str | None) -> ExperimentConfig:
    """Read either the compact format or an earlier Hermite checkpoint format."""
    if "network" not in data:
        values = dict(data)
        if "hermite_degree" not in values and checkpoint_format != "absde-hermite-v2":
            values["hermite_degree"] = 3
        return ExperimentConfig(**values)
    training = data["training"]
    return ExperimentConfig(
        example=data["example"],
        terminal_time=data["terminal_time"],
        anticipation_horizon=data["anticipation_horizon"],
        time_steps=data["time_steps"],
        train_paths=data["train_paths"],
        hermite_degree=3,
        ridge=training["ridge"],
        seed=training["seed"],
        dtype=data["dtype"],
    )


def load_result(path: str | Path, device: torch.device | str) -> TrainingResult:
    """Load a trained local Hermite model on the requested device."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = _config(checkpoint["config"], checkpoint.get("format"))
    models = LocalRegressors(config.time_steps, config.hermite_degree).to(
        device=device, dtype=torch_dtype(config.dtype)
    )
    state = {
        key.replace(".network.", "."): value
        for key, value in checkpoint["state_dict"].items()
    }
    models.load_state_dict(state)
    models.eval()
    return TrainingResult(models, config, checkpoint["wall_seconds"])
