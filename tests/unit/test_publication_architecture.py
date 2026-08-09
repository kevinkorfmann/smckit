"""Tests for the reproducible conceptual architecture figure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "plot_architecture.py"
SPEC = importlib.util.spec_from_file_location("smckit_publication_architecture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_architecture_figure_renders_vector_and_raster_outputs(tmp_path) -> None:
    outputs = MODULE.plot_architecture(tmp_path / "figure2", raster_dpi=300)

    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".tiff"}
    assert all(path.stat().st_size > 5_000 for path in outputs)
    svg = (tmp_path / "figure2.svg").read_text()
    assert "UPSTREAM PRESERVATION LANE" in svg
    assert "auto → native" in svg
    assert "PSMC+" in svg


def test_architecture_figure_rejects_low_resolution(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 300 dpi"):
        MODULE.plot_architecture(tmp_path / "figure2", raster_dpi=299)
