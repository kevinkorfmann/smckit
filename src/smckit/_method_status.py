"""Shared method trust/status metadata used by runtime and docs."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def method_statuses() -> list[dict[str, Any]]:
    path = resources.files("smckit.data").joinpath("method_status.json")
    return json.loads(path.read_text(encoding="utf-8"))


def method_status(method_name: str) -> dict[str, Any]:
    for entry in method_statuses():
        if entry["method"] == method_name:
            return entry
    raise KeyError(f"Unknown method status entry: {method_name}")
