"""Versioned, JSON-serializable run provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = "1.0"


def package_version() -> str:
    """Return the installed package version without importing ``smckit``."""
    try:
        return importlib.metadata.version("smckit")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(paths: Iterable[str | Path] | None) -> dict[str, str]:
    """Hash existing input files, preserving their supplied path labels."""
    hashes: dict[str, str] = {}
    for raw_path in paths or ():
        path = Path(raw_path)
        if path.is_file():
            hashes[str(raw_path)] = sha256_file(path)
    return hashes


def build_provenance(
    *,
    method: str,
    implementation_requested: str,
    implementation_used: str,
    arguments: dict[str, Any] | None = None,
    inputs: Iterable[str | Path] | None = None,
    seed: int | None = None,
    runtime_seconds: float | None = None,
    warnings: list[str] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable provenance envelope stored on every result."""
    if implementation_requested == "auto":
        if implementation_used == "native":
            from smckit._capabilities import native_supports

            selection_reason = (
                "native capability promoted"
                if native_supports(method)
                else "native capability not promoted; upstream could not serve request"
            )
        else:
            selection_reason = "native capability not promoted; upstream ready"
    else:
        selection_reason = f"explicit {implementation_used} request"

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "method": method,
        "package_version": package_version(),
        "implementation_requested": implementation_requested,
        "implementation_used": implementation_used,
        "selection_reason": selection_reason,
        "arguments": dict(arguments or {}),
        "input_sha256": input_hashes(inputs),
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "warnings": list(warnings or []),
        "artifacts": list(artifacts or []),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "upstream": upstream,
    }


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "build_provenance",
    "input_hashes",
    "package_version",
    "sha256_file",
]
