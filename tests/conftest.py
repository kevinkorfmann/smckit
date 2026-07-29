"""Repository-wide test-tier classification."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach stable tier markers from the test directory layout."""
    for item in items:
        parts = set(Path(str(item.path)).parts)
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)
