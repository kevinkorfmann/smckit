"""Slow MSMC-IM fitting workflows kept outside the unit tier."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from smckit.tl import msmc_im

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "vendor" / "MSMC-IM" / "example" / "Yoruba_French.8haps.combined.msmc2.final.txt"
pytestmark = pytest.mark.slow


def test_msmc_im_auto_prefers_promoted_native_and_handles_relative_input_paths() -> None:
    data = msmc_im("vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt")
    result = data.results["msmc_im"]

    assert result["implementation"] == "native"
    assert result["implementation_requested"] == "auto"
    assert np.all(np.isfinite(result["left_boundary"]))
    assert set(result["split_time_quantiles"]) == {0.25, 0.5, 0.75}


def test_msmc_im_exposes_raw_and_thresholded_migration_rates() -> None:
    result = msmc_im(INPUT, implementation="native").results["msmc_im"]

    assert "m_thresholded" in result
    assert np.all(result["m"] >= result["m_thresholded"])
    assert np.any(result["m"] > result["m_thresholded"])


def test_msmc_im_does_not_emit_matrix_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msmc_im(INPUT, implementation="native")

    pending = [
        warning
        for warning in caught
        if issubclass(warning.category, PendingDeprecationWarning)
        and "matrix subclass" in str(warning.message)
    ]
    assert pending == []
