"""Strict dense-sequence ASMC native-versus-upstream oracle."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from smckit.io._asmc import read_asmc
from smckit.tl import asmc

DATA_DIR = Path("vendor/ASMC/ASMC_data")
SEQUENCE_ROOT = DATA_DIR / "examples" / "asmc" / "exampleFile.n300"
ARRAY_ROOT = DATA_DIR / "examples" / "asmc" / "exampleFile.n300.array"
DQ_FILE = DATA_DIR / "decoding_quantities" / "30-100-2000_CEU.decodingQuantities.gz"

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(
        not (DQ_FILE.exists() and Path(f"{SEQUENCE_ROOT}.hap.gz").exists()),
        reason="ASMC dense sequence oracle fixtures are unavailable",
    ),
]


def _require_upstream_binding() -> None:
    pytest.importorskip("asmc.asmc")


def test_dense_sequence_interval_clears_strict_promotion_gate() -> None:
    _require_upstream_binding()
    data = read_asmc(SEQUENCE_ROOT, DQ_FILE)
    options = {
        "pairs": [(1, 2)],
        "mode": "sequence",
        "from_pos": 20_000,
        "to_pos": 25_000,
        "cm_burn_in": 0.5,
        "store_per_pair_posterior_mean": True,
        "store_per_pair_map": True,
    }
    native = asmc(
        copy.deepcopy(data),
        implementation="native",
        **options,
    ).results["asmc"]
    upstream = asmc(
        copy.deepcopy(data),
        implementation="upstream",
        **options,
    ).results["asmc"]

    native_means = np.asarray(native["per_pair_posterior_means"])
    upstream_means = np.asarray(upstream["per_pair_posterior_means"])
    relative_error = np.abs(native_means - upstream_means) / np.maximum(
        np.abs(upstream_means),
        np.finfo(float).tiny,
    )
    assert float(np.max(relative_error)) <= 1e-3
    assert (
        float(
            np.mean(np.asarray(native["per_pair_maps"]) == np.asarray(upstream["per_pair_maps"]))
        )
        >= 0.999
    )


def test_compressed_sequence_upstream_contract_does_not_duplicate_skip_control() -> None:
    _require_upstream_binding()
    data = read_asmc(ARRAY_ROOT, DQ_FILE)
    result = asmc(
        data,
        pairs=[(1, 2)],
        mode="sequence",
        skip_csfs_distance=float("inf"),
        store_per_pair_posterior_mean=True,
        implementation="upstream",
    ).results["asmc"]
    assert result["n_pairs_decoded"] == 1
    assert np.all(np.isfinite(result["per_pair_posterior_means"][0]))
