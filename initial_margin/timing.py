"""Synchronized wall and GPU timing helpers for reproducibility campaigns."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class Timing:
    """Store synchronized wall time and device execution time in seconds."""

    wall_seconds: float
    gpu_seconds: float


def timed_call(function: Callable[[], T], device: torch.device) -> tuple[T, Timing]:
    """Run one callable and measure it with CUDA events when applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    wall_start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize(device)
        gpu_seconds = start.elapsed_time(end) / 1000.0
    else:
        gpu_seconds = 0.0
    return result, Timing(time.perf_counter() - wall_start, gpu_seconds)
