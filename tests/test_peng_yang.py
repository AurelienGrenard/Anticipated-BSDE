"""Focused checks for the Peng--Yang backward scheme and its artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from peng_yang.checkpoints import load_result, save_result
from peng_yang.config import ExperimentConfig
from peng_yang.evaluation import evaluate
from peng_yang.problems import get_example
from peng_yang.regression import HermiteRegressor
from peng_yang.solver import train_backward


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_time_zero_uses_only_the_constant_feature() -> None:
    """Check the deterministic initial state cannot activate higher polynomials."""
    inputs = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(HermiteRegressor.basis(inputs), expected)


def test_small_training_checkpoint_round_trip(tmp_path: Path) -> None:
    """Train a small benchmark and check its serialized regressors load exactly."""
    config = ExperimentConfig(example="4.3", time_steps=5, train_paths=2_000)
    example = get_example(config.example)
    result = train_backward(config, example, "cpu", progress=False)
    checkpoint = tmp_path / "example_4_3.pt"
    save_result(result, checkpoint)
    loaded = load_result(checkpoint, "cpu")
    for name, value in result.models.state_dict().items():
        torch.testing.assert_close(loaded.models.state_dict()[name], value)
    metrics = evaluate(
        loaded.models,
        example,
        loaded.config,
        "cpu",
        test_paths=1_000,
    ).metrics
    assert math.isfinite(metrics.y_mse_sup)
    assert math.isfinite(metrics.z_mse_integrated)


def test_retained_summaries_use_the_published_grids() -> None:
    """Check every committed convergence summary covers the six paper grids."""
    expected_steps = [5, 10, 20, 40, 80, 160]
    for slug in ("example_4_3", "example_5_3_1", "example_5_3_2"):
        path = PROJECT_ROOT / "peng_yang" / "results" / (
            f"{slug}_reconstruction_summary.json"
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert [row["N"] for row in rows] == expected_steps
