"""Integration coverage for the upstream-backed PSMC wrapper."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import smckit
from smckit.io import read_psmcfa
from smckit.tl import psmc


pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("cc") is None,
    reason="Upstream PSMC build toolchain is not available",
)


def test_psmc_upstream_backend_runs_end_to_end() -> None:
    smckit.upstream.bootstrap("psmc")

    data = read_psmcfa("tests/data/NA12878_chr22.psmcfa")
    result = psmc(
        data,
        pattern="1+1+1",
        n_iterations=1,
        max_t=5.0,
        tr_ratio=5.0,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
    )

    res = result.results["psmc"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.all(res["ne"] > 0)
    assert np.isfinite(res["theta"])
    assert np.isfinite(res["rho"])
    assert set(res["upstream"]) >= {"tool", "binary", "input_path", "effective_args"}
