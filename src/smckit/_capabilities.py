"""Machine-readable method capability and promotion metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from smckit._method_status import method_status, method_statuses

STANDARD_CAPABILITY = "standard"


def _native_capabilities(entry: dict[str, Any]) -> dict[str, Any]:
    promoted = bool(entry.get("native_default_eligible", False))
    return {
        "available": entry.get("native") == "✓",
        "trusted": bool(entry.get("native_trusted_for_docs", False)),
        "default_eligible": promoted,
        "promoted": [STANDARD_CAPABILITY] if promoted else [],
        "warning": entry.get("native_warning"),
        "agreement": entry.get("tracked_agreement"),
    }


def capabilities(method_name: str | None = None) -> dict[str, Any]:
    """Return method implementation and promotion capabilities.

    Parameters
    ----------
    method_name
        Public method name. When omitted, return the complete registry keyed
        by method name.
    """
    from smckit import upstream

    entries = [method_status(method_name)] if method_name is not None else method_statuses()
    result: dict[str, Any] = {}
    for raw_entry in entries:
        entry = deepcopy(raw_entry)
        name = str(entry["method"])
        upstream_status = upstream.method_status(name)
        result[name] = {
            "method": name,
            "display_name": entry["display_name"],
            "schema_version": 1,
            "native": _native_capabilities(entry),
            "upstream": upstream_status,
            "notes": entry.get("notes", ""),
        }
    if method_name is not None:
        return result[method_name]
    return result


def native_supports(
    method_name: str,
    requested: set[str] | None = None,
) -> bool:
    """Return whether native is promoted for every requested capability."""
    requested = requested or {STANDARD_CAPABILITY}
    promoted = set(capabilities(method_name)["native"]["promoted"])
    return requested <= promoted


__all__ = ["STANDARD_CAPABILITY", "capabilities", "native_supports"]
