"""Helpers for selecting native vs upstream algorithm implementations."""

from __future__ import annotations

import warnings
from typing import Any

import smckit.upstream as upstream
from smckit._method_status import method_status

VALID_IMPLEMENTATIONS = {"auto", "native", "upstream"}


class NativeTrustWarning(UserWarning):
    """Warning emitted when a native method is runnable but not docs-trusted yet."""


def normalize_implementation(
    implementation: str,
    *,
    backend: str | None = None,
) -> str:
    """Normalize the public implementation selector.

    ``backend`` is kept as a compatibility alias for older callers that used
    it to select native vs upstream algorithm provenance.
    """
    if backend is not None:
        if implementation != "auto" and implementation != backend:
            raise ValueError(
                "implementation and backend selectors disagree; pass only "
                "implementation or use matching values."
            )
        warnings.warn(
            "'backend' is deprecated for algorithm selection; use "
            "'implementation' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        implementation = backend

    if implementation not in VALID_IMPLEMENTATIONS:
        raise ValueError("implementation must be one of: auto, native, upstream")
    return implementation


def choose_implementation(
    implementation: str,
    *,
    upstream_available: bool,
) -> str:
    """Resolve ``auto`` to the concrete implementation that will run."""
    if implementation == "auto":
        return "upstream" if upstream_available else "native"
    return implementation


def require_upstream_available(method_name: str) -> None:
    """Raise a consistent error for methods without an upstream bridge."""
    tool_status = upstream.method_status(method_name)
    if tool_status is None:
        raise NotImplementedError(
            f"Upstream implementation for smckit.tl.{method_name} is not available yet."
        )

    missing = ", ".join(tool_status["missing"]) if tool_status["missing"] else "unknown reason"
    if tool_status["public_upstream"]:
        raise RuntimeError(
            f"Upstream implementation for smckit.tl.{method_name} is not ready: {missing}. "
            f"Run smckit.upstream.status('{tool_status['tool']}') for details.\n"
            f"{tool_status.get('install_help', '')}".rstrip()
        )

    raise NotImplementedError(
        f"Upstream implementation for smckit.tl.{method_name} is not available yet: {missing}."
    )


def annotate_result(
    result: dict[str, Any],
    *,
    implementation_requested: str,
    implementation_used: str,
    upstream_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach standardized implementation metadata to a result payload."""
    result["implementation_requested"] = implementation_requested
    result["implementation"] = implementation_used
    if upstream_metadata is not None:
        base = result.get("upstream", {})
        if isinstance(base, dict):
            merged = dict(base)
            merged.update(upstream_metadata)
            result["upstream"] = merged
        else:
            result["upstream"] = upstream_metadata
    return result


def warn_if_native_not_trusted(method_name: str, implementation_used: str) -> None:
    """Warn when a runnable native path is not trusted for docs defaults."""
    if implementation_used != "native":
        return

    status = method_status(method_name)
    if status.get("native_trusted_for_docs", True):
        return

    message = str(status.get("native_warning", "")).strip()
    tool_status = upstream.method_status(method_name)
    if tool_status is not None:
        install_text = str(tool_status.get("install_help", "")).strip()
        if install_text:
            message = f"{message}\n{install_text}" if message else install_text

    if message:
        warnings.warn(message, NativeTrustWarning, stacklevel=3)


def method_upstream_available(method_name: str) -> bool:
    """Return whether the upstream path for a method is ready to run."""
    status = upstream.method_status(method_name)
    return False if status is None else bool(status["ready"])


def standard_upstream_metadata(
    method_name: str,
    *,
    effective_args: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardized upstream provenance metadata for a method."""
    status = upstream.method_status(method_name) or {}
    metadata = {
        "tool": status.get("tool", method_name),
        "version": status.get("version", "unknown"),
        "vendor_path": status.get("vendor_path"),
        "cache_path": status.get("cache_path"),
        "runtime": status.get("runtime"),
    }
    if effective_args is not None:
        metadata["effective_args"] = effective_args
    if extra:
        metadata.update(extra)
    return metadata
