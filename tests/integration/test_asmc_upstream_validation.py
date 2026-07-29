"""Integration coverage for the upstream-backed ASMC wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io._asmc import read_asmc
from smckit.tl import asmc

DATA_DIR = Path("vendor/ASMC/ASMC_data")
EXAMPLE_ROOT = DATA_DIR / "examples" / "asmc" / "exampleFile.n300.array"
DQ_FILE = DATA_DIR / "decoding_quantities" / "30-100-2000_CEU.decodingQuantities.gz"


pytestmark = pytest.mark.skipif(
    not (DQ_FILE.exists() and Path(str(EXAMPLE_ROOT) + ".hap.gz").exists()),
    reason="ASMC upstream fixtures not available",
)


def test_asmc_upstream_backend_runs_end_to_end() -> None:
    try:
        smckit.upstream.bootstrap("asmc")
    except Exception as exc:
        pytest.skip(f"ASMC upstream bootstrap unavailable: {exc}")
    data = read_asmc(str(EXAMPLE_ROOT), str(DQ_FILE))
    result = asmc(
        data,
        implementation="upstream",
    )
    res = result.results["asmc"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert "sum_of_posteriors" in res
    assert np.all(np.isfinite(res["sum_of_posteriors"]))


def test_asmc_upstream_python_binding_preserves_explicit_pair_interface() -> None:
    pytest.importorskip("asmc.asmc")
    data = read_asmc(str(EXAMPLE_ROOT), str(DQ_FILE))
    result = asmc(
        data,
        pairs=[(1, 2), (2, 3), (3, 4)],
        store_per_pair_posterior=True,
        store_per_pair_posterior_mean=True,
        store_per_pair_map=True,
        implementation="upstream",
    )
    res = result.results["asmc"]
    assert res["implementation"] == "upstream"
    assert res["n_pairs_decoded"] == 3
    assert len(res["per_pair_posteriors"]) == 3
    assert res["per_pair_posteriors"][0].shape == (6760, 69)
    assert len(res["per_pair_posterior_means"]) == 3
    assert len(res["per_pair_maps"]) == 3
    assert res["min_posterior_means"].shape == (6760,)
    assert res["argmin_maps"].shape == (6760,)
    assert np.all(np.isfinite(res["sum_of_posteriors"]))
