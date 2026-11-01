"""Shared source and software provenance for every initial-margin campaign."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]


def provenance(device: torch.device) -> dict[str, Any]:
    """Record exact solver hashes, package versions, revision and execution device."""
    sources = [path for folder in ("hermite", "initial_margin")
               for path in sorted((ROOT / folder).rglob("*.py"))]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "git_revision": revision,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
