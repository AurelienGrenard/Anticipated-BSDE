"""Mathematical definitions and exact evaluation oracles for the three ABSDEs.

The terminal functions are part of the problem data and are available to training.
The exact solution functions are evaluation-only and are never used by the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

Tensor = torch.Tensor
AnticipatedKind = Literal["future_y", "future_z_increment"]


@dataclass(frozen=True)
class ExampleDefinition:
    key: str
    title: str
    anticipated_kind: AnticipatedKind
    terminal_y: Callable[[Tensor, Tensor, float], Tensor]
    terminal_z: Callable[[Tensor, Tensor, float], Tensor]
    driver_from_a: Callable[[Tensor, Tensor, float], Tensor]
    exact_y: Callable[[Tensor, Tensor, float], Tensor]
    exact_z: Callable[[Tensor, Tensor, float], Tensor]


def _43_y(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Y process and terminal condition for Example 4.3."""
    del terminal_time
    return t * w


def _43_z(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Z process and terminal condition for Example 4.3."""
    del w, terminal_time
    return t


def _43_driver(t: Tensor, anticipated: Tensor, delta: float) -> Tensor:
    """Evaluate the anticipated driver of Example 4.3."""
    return -anticipated / (t + delta)


def _531_y(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Y process and terminal condition for Example 5.3 (I)."""
    return w.square() - terminal_time - (terminal_time - t)


def _531_z(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Z process and terminal condition for Example 5.3 (I)."""
    del t, terminal_time
    return 2.0 * w


def _532_y(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Y process and terminal condition for Example 5.3 (II)."""
    return 4.0 * w.square() - 4.0 * (terminal_time - t)


def _532_z(t: Tensor, w: Tensor, terminal_time: float) -> Tensor:
    """Return the Z process and terminal condition for Example 5.3 (II)."""
    del t, terminal_time
    return 8.0 * w


def _z_increment_driver(t: Tensor, anticipated: Tensor, delta: float) -> Tensor:
    """Evaluate the anticipated Z-increment driver of the 5.3 examples."""
    del t
    coefficient = torch.sqrt(anticipated.new_tensor(torch.pi / (2.0 * delta)))
    return -coefficient * anticipated


_EXAMPLES = {
    "4.3": ExampleDefinition(
        key="4.3",
        title="Peng and Yang Example 4.3",
        anticipated_kind="future_y",
        terminal_y=_43_y,
        terminal_z=_43_z,
        driver_from_a=_43_driver,
        exact_y=_43_y,
        exact_z=_43_z,
    ),
    "5.3.1": ExampleDefinition(
        key="5.3.1",
        title="Peng and Yang Example 5.3 (I)",
        anticipated_kind="future_z_increment",
        # This is the continuous convention used by the original notebook. The
        # article draft displays a different terminal line which would jump at T.
        terminal_y=_531_y,
        terminal_z=_531_z,
        driver_from_a=_z_increment_driver,
        exact_y=_531_y,
        exact_z=_531_z,
    ),
    "5.3.2": ExampleDefinition(
        key="5.3.2",
        title="Peng and Yang Example 5.3 (II)",
        anticipated_kind="future_z_increment",
        terminal_y=_532_y,
        terminal_z=_532_z,
        driver_from_a=_z_increment_driver,
        exact_y=_532_y,
        exact_z=_532_z,
    ),
}


def get_example(key: str) -> ExampleDefinition:
    """Return one of the three benchmark definitions by key."""
    try:
        return _EXAMPLES[key]
    except KeyError as exc:
        raise ValueError(f"unknown ABSDE example {key!r}") from exc
