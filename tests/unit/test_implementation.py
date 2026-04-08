"""Tests for native/upstream implementation selection."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.tl import asmc, dical2, esmc2, msmc2, msmc_im, psmc, smcpp
from smckit.tl._implementation import (
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
)


def _tiny_psmc_data() -> SmcData:
    seq = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    return SmcData(
        uns={
            "records": [{"codes": seq}],
            "sum_L": int((seq < 2).sum()),
            "sum_n": int((seq == 1).sum()),
        },
    )


def _tiny_smcpp_data() -> SmcData:
    return SmcData(
        uns={
            "records": [{
                "name": "synthetic",
                "observations": [(10, 0, 0), (1, 0, 1), (10, 0, 0), (1, 1, 0)],
            }],
            "n_undist": 5,
        },
    )


def _tiny_esmc2_data() -> SmcData:
    seq = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    data = SmcData()
    data.uns["records"] = [{"codes": seq}]
    data.uns["sum_L"] = int((seq < 2).sum())
    data.uns["sum_n"] = int((seq == 1).sum())
    return data


def test_normalize_implementation_accepts_valid_values() -> None:
    assert normalize_implementation("auto") == "auto"
    assert normalize_implementation("native") == "native"
    assert normalize_implementation("upstream") == "upstream"


def test_normalize_implementation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="implementation must be one of"):
        normalize_implementation("gpu")


def test_normalize_implementation_supports_backend_alias() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = normalize_implementation("auto", backend="native")

    assert resolved == "native"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_choose_implementation_auto_prefers_upstream_when_available() -> None:
    assert choose_implementation("auto", upstream_available=True) == "upstream"
    assert choose_implementation("auto", upstream_available=False) == "native"


def test_dical2_upstream_requires_path_backed_inputs() -> None:
    with pytest.raises(ValueError, match="path-backed inputs"):
        dical2(SmcData(), implementation="upstream")


def test_dical2_auto_uses_native_for_array_inputs() -> None:
    data = SmcData(
        sequences=np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8),
        params={"theta": 0.001, "rho": 0.0005},
        uns={"n_alleles": 2},
    )
    res = dical2(
        data,
        n_intervals=3,
        n_em_iterations=1,
        implementation="auto",
    ).results["dical2"]
    assert res["implementation_requested"] == "auto"
    assert res["implementation"] == "native"


def test_psmc_records_requested_and_used_implementation() -> None:
    res = psmc(_tiny_psmc_data(), pattern="1+1+1", n_iterations=0, implementation="auto").results["psmc"]
    assert res["implementation_requested"] == "auto"
    expected = "upstream" if method_upstream_available("psmc") else "native"
    assert res["implementation"] == expected


def test_smcpp_implementation_alias_and_metadata() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smcpp(
            _tiny_smcpp_data(),
            n_intervals=4,
            max_iterations=1,
            regularization=1.0,
            seed=1,
            backend="native",
        ).results["smcpp"]

    assert res["implementation_requested"] == "native"
    assert res["implementation"] == "native"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_esmc2_implementation_metadata() -> None:
    res = esmc2(
        _tiny_esmc2_data(),
        n_states=4,
        n_iterations=1,
        mu=1e-8,
        implementation="native",
    ).results["esmc2"]

    assert res["implementation_requested"] == "native"
    assert res["implementation"] == "native"
