"""Integrity checks for the numerical artifacts retained with the paper."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "artifacts.sha256"


def test_retained_artifact_checksums() -> None:
    """Check every retained checkpoint, result, figure, and notebook hash."""
    entries = [line.split(maxsplit=1) for line in MANIFEST.read_text().splitlines()]
    assert len(entries) == 20
    for expected, relative_name in entries:
        artifact = ROOT / relative_name
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == expected, f"checksum mismatch for {relative_name}"
