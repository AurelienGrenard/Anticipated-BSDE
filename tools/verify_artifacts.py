"""Verify the SHA-256 manifest of every retained numerical artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "artifacts.sha256"


def _digest(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _entries() -> list[tuple[str, Path]]:
    """Parse nonempty manifest rows into expected hashes and repository paths."""
    entries = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split(maxsplit=1)
            entries.append((expected, ROOT / relative))
    return entries


def main() -> None:
    """Reject missing or modified artifacts and print one compact success line."""
    failures = []
    entries = _entries()
    for expected, path in entries:
        if not path.is_file():
            failures.append(f"missing: {path.relative_to(ROOT)}")
        elif _digest(path) != expected:
            failures.append(f"checksum mismatch: {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(entries)} retained artifacts from {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
