"""Tests for publication-oriented SMC++ figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from smckit.pl import (
    save_smcpp_figure,
    smcpp_cross_validation_scores,
    smcpp_demographic_history,
)


@pytest.fixture
def smcpp_result() -> dict:
    return {
        "time": np.geomspace(1e-4, 5.0, 8),
        "time_years": np.geomspace(100, 1_000_000, 8),
        "ne": np.geomspace(8_000, 30_000, 8),
        "cross_validation": {
            "selected_regularization": 4.0,
            "candidates": [
                {
                    "regularization": 2.0,
                    "heldout_log_likelihood": -120.0,
                    "folds": [
                        {"heldout_log_likelihood": -58.0},
                        {"heldout_log_likelihood": -62.0},
                    ],
                },
                {
                    "regularization": 4.0,
                    "heldout_log_likelihood": -110.0,
                    "folds": [
                        {"heldout_log_likelihood": -54.0},
                        {"heldout_log_likelihood": -56.0},
                    ],
                },
            ],
        },
    }


def test_smcpp_demography_has_log_axes_and_knots(smcpp_result) -> None:
    ax = smcpp_demographic_history(smcpp_result, show_knots=True)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    plt.close(ax.figure)


def test_smcpp_cross_validation_shows_fold_points_and_selection(smcpp_result) -> None:
    ax = smcpp_cross_validation_scores(smcpp_result)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 3
    assert ax.get_xlabel() == "Regularization penalty"
    plt.close(ax.figure)


def test_smcpp_vector_export(smcpp_result, tmp_path) -> None:
    ax = smcpp_demographic_history(smcpp_result)
    output = save_smcpp_figure(ax.figure, tmp_path / "smcpp.svg")
    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(ax.figure)


def test_smcpp_plot_rejects_invalid_time_unit(smcpp_result) -> None:
    with pytest.raises(ValueError, match="time_unit"):
        smcpp_demographic_history(smcpp_result, time_unit="centuries")
