"""Tests for publication-oriented MSMC-IM diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from smckit.pl import msmc_im_summary, save_msmc_im_figure


@pytest.fixture
def result() -> dict:
    return {
        "left_boundary": np.array([10, 100, 1_000, 10_000], dtype=float),
        "N1": np.array([10_000, 12_000, 20_000, 25_000], dtype=float),
        "N2": np.array([8_000, 9_000, 18_000, 24_000], dtype=float),
        "m": np.array([1e-4, 5e-5, 1e-5, 1e-30], dtype=float),
        "m_thresholded": np.array([1e-4, 5e-5, 1e-5, 1e-30], dtype=float),
        "M": np.array([0.1, 0.3, 0.7, 1.0], dtype=float),
        "split_time_quantiles": {0.25: 70.0, 0.5: 550.0, 0.75: 3_000.0},
    }


def test_summary_uses_aligned_log_time_and_separate_units(result) -> None:
    axes = msmc_im_summary(result, population_labels=("YRI", "CEU"))
    assert len(axes) == 3
    assert axes[0].get_yscale() == "log"
    assert axes[1].get_yscale() == "log"
    assert axes[2].get_xscale() == "log"
    assert len(axes[2].collections) == 3
    assert [line.get_label() for line in axes[0].lines] == ["YRI", "CEU"]
    plt.close(axes[0].figure)


def test_summary_vector_export(result, tmp_path) -> None:
    axes = msmc_im_summary(result)
    output = save_msmc_im_figure(axes[0].figure, tmp_path / "msmc-im.svg")
    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(axes[0].figure)


def test_summary_rejects_mismatched_series(result) -> None:
    result["N1"] = result["N1"][:-1]
    with pytest.raises(ValueError, match="shape"):
        msmc_im_summary(result)
