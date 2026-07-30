"""Tests for publication-oriented PHLASH visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from smckit.pl import phlash_demographic_history, save_phlash_figure


@pytest.fixture
def phlash_result() -> dict:
    time = np.geomspace(10, 100_000, 50)
    posterior = np.vstack(
        [
            10_000 + 2_000 * np.sin(np.log(time) + phase)
            for phase in np.linspace(0, 1, 12)
        ]
    )
    return {
        "time": time,
        "ne": np.median(posterior, axis=0),
        "posterior_ne": posterior,
        "credible_interval": {
            "level": 0.95,
            "lower": np.quantile(posterior, 0.025, axis=0),
            "upper": np.quantile(posterior, 0.975, axis=0),
        },
    }


def test_phlash_plot_has_interval_and_log_axes(phlash_result) -> None:
    ax = phlash_demographic_history(phlash_result, posterior_samples=4)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert len(ax.collections) == 1
    assert len(ax.lines) == 5
    plt.close(ax.figure)


def test_phlash_plot_validates_draw_count(phlash_result) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        phlash_demographic_history(phlash_result, posterior_samples=13)


def test_phlash_vector_export(phlash_result, tmp_path) -> None:
    ax = phlash_demographic_history(phlash_result)
    output = save_phlash_figure(ax.figure, tmp_path / "phlash.pdf")
    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(ax.figure)
