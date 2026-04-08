"""Integration coverage for the upstream-backed SMC++ wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.tl._smcpp import _resolve_upstream_smcpp_python, smcpp


pytestmark = pytest.mark.skipif(
    _resolve_upstream_smcpp_python() is None,
    reason="Upstream SMC++ side environment not available",
)


def test_smcpp_upstream_backend_runs_end_to_end() -> None:
    observations = []
    for i in range(120):
        observations.append((9999, 0, 0))
        observations.append((1, i % 2, (i % 5) + 1))

    data = SmcData(
        uns={
            "records": [{"name": "synthetic", "observations": observations}],
            "n_undist": 5,
        },
    )

    result = smcpp(
        data,
        n_intervals=4,
        max_iterations=1,
        regularization=10.0,
        seed=42,
        backend="upstream",
    )

    res = result.results["smcpp"]
    assert res["backend"] == "upstream"
    assert res["ne"].shape == (4,)
    assert np.all(res["ne"] > 0)
    assert np.isfinite(res["log_likelihood"])
    assert set(res["upstream"]) >= {"alpha", "model", "hidden_states", "stepwise_ne"}
