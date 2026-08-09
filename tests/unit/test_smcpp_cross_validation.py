"""Tests for native SMC++ contig-level cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.tl import smcpp_cross_validate


def _cross_validation_data() -> SmcData:
    records = []
    for index, derived in enumerate((1, 2, 1)):
        records.append(
            {
                "name": f"chr{index + 1}",
                "observations": [
                    (100, 0, 0),
                    (1, 1, derived),
                    (100, 0, 0),
                ],
            }
        )
    return SmcData(
        uns={
            "records": records,
            "n_undist": 3,
            "n_distinguished": 2,
            "n_populations": 1,
        }
    )


def test_cross_validation_scores_folds_and_refits_best_candidate() -> None:
    data = smcpp_cross_validate(
        _cross_validation_data(),
        regularization_candidates=[0.0, 2.0],
        folds=2,
        seed=37,
        n_intervals=2,
        max_iterations=0,
    )
    result = data.results["smcpp"]
    cross_validation = result["cross_validation"]

    assert cross_validation["folds"] == 2
    assert cross_validation["selected_regularization"] in {0.0, 2.0}
    assert result["regularization"] == cross_validation["selected_regularization"]
    assert len(cross_validation["candidates"]) == 2
    assert all(
        np.isfinite(candidate["heldout_log_likelihood"])
        for candidate in cross_validation["candidates"]
    )
    assert all(len(candidate["folds"]) == 2 for candidate in cross_validation["candidates"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"folds": 4}, "folds"),
        ({"regularization_candidates": []}, "regularization_candidates"),
        ({"regularization_candidates": [1.0, 1.0]}, "unique"),
    ],
)
def test_cross_validation_validates_controls(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        smcpp_cross_validate(_cross_validation_data(), **kwargs)
