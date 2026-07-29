"""Integration coverage for the upstream-backed MSMC2 wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io import read_multihetsep
from smckit.tl import msmc2

INPUT = Path("data/msmc2_test.multihetsep")


pytestmark = pytest.mark.skipif(
    not INPUT.exists(),
    reason="MSMC2 input fixture not available",
)


def test_msmc2_upstream_backend_runs_end_to_end() -> None:
    smckit.upstream.bootstrap("msmc2")
    data = read_multihetsep(INPUT)
    result = msmc2(
        data,
        n_iterations=1,
        time_pattern="1*2+2*1",
        implementation="upstream",
    )
    res = result.results["msmc2"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.all(np.isfinite(res["lambda"]))
    assert np.all(res["lambda"] > 0)
    assert set(res["upstream"]) >= {"tool", "binary", "final_path", "effective_args"}


def test_msmc2_upstream_preserves_skip_ambiguous_and_artifacts(tmp_path) -> None:
    smckit.upstream.bootstrap("msmc2")
    prefix = tmp_path / "upstream"
    result = msmc2(
        read_multihetsep(INPUT, pair_indices=[(0, 1)], skip_ambiguous=True),
        n_iterations=0,
        time_pattern="1*2+2*1",
        stride_width=200,
        time_factor=0.5,
        n_threads=1,
        output_prefix=prefix,
        implementation="upstream",
    )

    res = result.results["msmc2"]
    assert res["skip_ambiguous"] is True
    assert res["upstream"]["effective_args"]["skipAmbiguous"] is True
    assert {artifact["kind"] for artifact in res["provenance"]["artifacts"]} >= {
        "msmc2-final.txt",
        "msmc2-log",
        "msmc2-loop.txt",
    }
