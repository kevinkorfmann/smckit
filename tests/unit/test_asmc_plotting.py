"""Tests for publication-oriented ASMC visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from smckit.pl import (
    asmc_posterior_heatmap,
    asmc_recent_coalescence_density,
    save_asmc_figure,
)


@pytest.fixture
def asmc_result() -> dict:
    rng = np.random.default_rng(12)
    posterior = rng.random((200, 5))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return {
        "expected_times": np.array([10.0, 50.0, 200.0, 1_000.0, 5_000.0]),
        "sum_of_posteriors": posterior * 6,
        "sum_of_posteriors_major_minor": {
            "00": posterior,
            "01": posterior * 2,
            "11": posterior * 3,
        },
    }


def test_posterior_heatmap_uses_log_time_and_colorbar(asmc_result):
    ax = asmc_posterior_heatmap(asmc_result, genotype="01", max_time=1_000)
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Genomic position (bp)"
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)


def test_recent_coalescence_density_uses_windows(asmc_result):
    ax = asmc_recent_coalescence_density(
        asmc_result,
        max_time=200,
        window_sites=20,
        step_sites=10,
    )
    assert len(ax.lines[0].get_xdata()) == 19
    assert np.all(ax.lines[0].get_ydata() >= 0)
    plt.close(ax.figure)


def test_vector_export(asmc_result, tmp_path):
    ax = asmc_posterior_heatmap(asmc_result)
    output = save_asmc_figure(ax.figure, tmp_path / "asmc-posterior.pdf")
    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(ax.figure)


def test_invalid_export_extension(asmc_result, tmp_path):
    ax = asmc_posterior_heatmap(asmc_result)
    with pytest.raises(ValueError, match="PDF"):
        save_asmc_figure(ax.figure, tmp_path / "figure.jpg")
    plt.close(ax.figure)
