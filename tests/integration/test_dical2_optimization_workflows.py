"""Slow end-to-end diCal2 composite-likelihood optimization workflows."""

from __future__ import annotations

import numpy as np
import pytest

from smckit.io import read_dical2
from smckit.tl import dical2

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("seed", "n_haplotypes", "length", "n_intervals", "n_iterations", "max_t", "mode"),
    [
        (7, 4, 200, 4, 2, 2.0, "pcl"),
        (8, 5, 150, 3, 2, 1.5, "pac"),
        (9, 4, 150, 3, 1, 1.5, "lol"),
    ],
)
def test_dical2_composite_optimization_workflow(
    seed,
    n_haplotypes,
    length,
    n_intervals,
    n_iterations,
    max_t,
    mode,
):
    rng = np.random.default_rng(seed)
    sequences = (rng.random((n_haplotypes, length)) < 0.02).astype(np.int8)
    data = read_dical2(sequences=sequences, theta=0.001, rho=0.0005)

    result = dical2(
        data,
        n_intervals=n_intervals,
        n_em_iterations=n_iterations,
        max_t=max_t,
        composite_mode=mode,
    ).results["dical2"]

    assert "ne" in result
    assert np.isfinite(result["log_likelihood"])
    assert np.all(np.asarray(result["ne"]) > 0)
    assert np.all(np.isfinite(result["ne"]))
