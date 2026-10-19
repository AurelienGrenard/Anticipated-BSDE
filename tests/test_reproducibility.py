"""Integrity checks for the numerical artifacts retained with the paper."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "artifacts.sha256"


def test_retained_artifact_checksums() -> None:
    """Check every retained checkpoint, result, figure, and notebook hash."""
    entries = [line.split(maxsplit=1) for line in MANIFEST.read_text().splitlines()]
    names = {name for _, name in entries}
    assert len(names) == len(entries)
    retained = {
        str(path.relative_to(ROOT))
        for folder in (ROOT / "initial_margin" / "results").glob("margin_period*")
        if folder.is_dir()
        for path in folder.iterdir() if path.is_file()
    }
    assert retained
    assert retained <= names
    for expected, relative_name in entries:
        artifact = ROOT / relative_name
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == expected, f"checksum mismatch for {relative_name}"
