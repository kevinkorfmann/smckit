"""Upstream tool registry, readiness probes, and local bootstrap helpers."""

from __future__ import annotations

from typing import Any

from smckit.upstream._registry import (
    bootstrap_tool,
    get_tool,
    get_tool_for_method,
    repo_root,
    tool_names,
)


def status(tool: str | None = None) -> dict[str, Any]:
    """Return upstream readiness status for one tool or for the whole registry."""
    if tool is not None:
        return get_tool(tool).status()
    return {name: get_tool(name).status() for name in tool_names()}


def bootstrap(tool: str | None = None) -> dict[str, Any]:
    """Bootstrap one tool or every bootstrap-capable upstream tool."""
    if tool is not None:
        return bootstrap_tool(tool)

    results: dict[str, Any] = {}
    for name in tool_names():
        try:
            results[name] = bootstrap_tool(name)
        except RuntimeError as exc:
            results[name] = {
                **get_tool(name).status(),
                "bootstrap_error": str(exc),
            }
    return results


def is_ready(tool: str) -> bool:
    """Return whether a tool's public upstream path is ready to run."""
    return bool(get_tool(tool).status()["ready"])


def method_status(method_name: str) -> dict[str, Any] | None:
    """Return upstream status for the registry entry matching a method name."""
    spec = get_tool_for_method(method_name)
    return None if spec is None else spec.status()


__all__ = [
    "bootstrap",
    "get_tool",
    "is_ready",
    "method_status",
    "repo_root",
    "status",
]
