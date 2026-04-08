"""Helpers for locating packaged example data."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import tempfile

_EXAMPLES_PACKAGE = "smckit.data.examples"


def example_path(name: str) -> Path:
    """Return a filesystem path for a packaged example fixture.

    Parameters
    ----------
    name : str
        Relative example path inside ``smckit.data.examples``.
    """
    resource = resources.files(_EXAMPLES_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown packaged example: {name}")

    cache_root = Path(tempfile.gettempdir()) / "smckit-examples"
    target = cache_root / name
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = resource.read_bytes()
    if not target.exists() or target.read_bytes() != payload:
        target.write_bytes(payload)

    return target


def example_prefix(prefix: str, suffixes: tuple[str, ...]) -> Path:
    """Materialize a group of packaged files that share a common root."""
    for suffix in suffixes:
        example_path(prefix + suffix)
    return Path(str(example_path(prefix + suffixes[0]))[: -len(suffixes[0])])
