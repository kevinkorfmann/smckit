"""Tests for MSMC-IM output semantics and oracle-aligned helpers."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from smckit.tl import msmc_im
from smckit.tl._msmc_im import (
    _compute_tmrca_density,
    _correct_ancient_lambdas,
    _cumulative_migration,
    _expand_params,
    _make_Q,
    _make_Qexp,
    _parse_im_pattern,
    _propagate_state_vectors,
    _tmrca_from_msmc,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:the matrix subclass is not the recommended way:PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar multiply:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    ),
]


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "vendor" / "MSMC-IM" / "example" / "Yoruba_French.8haps.combined.msmc2.final.txt"


def _load_vendor_funcs():
    module_path = ROOT / "vendor" / "MSMC-IM" / "MSMC_IM_funcs.py"
    spec = importlib.util.spec_from_file_location("vendor_msmc_im_funcs", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VENDOR = _load_vendor_funcs()


def test_msmc_im_auto_prefers_upstream_and_handles_relative_input_paths() -> None:
    data = msmc_im("vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt")
    res = data.results["msmc_im"]

    assert res["implementation"] == "upstream"
    assert res["implementation_requested"] == "auto"
    assert np.all(np.isfinite(res["left_boundary"]))
    assert set(res["split_time_quantiles"]) == {0.25, 0.5, 0.75}


def test_msmc_im_exposes_raw_and_thresholded_migration_rates() -> None:
    data = msmc_im(INPUT, implementation="native")
    res = data.results["msmc_im"]

    assert "m_thresholded" in res
    assert np.all(res["m"] >= res["m_thresholded"])
    assert np.any(res["m"] > res["m_thresholded"])


def test_msmc_im_does_not_emit_matrix_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msmc_im(INPUT, implementation="native")

    pending = [
        w for w in caught
        if issubclass(w.category, PendingDeprecationWarning)
        and "matrix subclass" in str(w.message)
    ]
    assert pending == []


def test_msmc_im_pattern_and_ancient_lambda_helpers_match_expected_semantics() -> None:
    segs, repeat = _parse_im_pattern("1*2+25*1+1*2+1*3")
    assert segs == [1, 25, 1, 1]
    assert repeat == [2, 1, 2, 3]

    unique = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expanded = _expand_params(unique, [1, 2], [2, 3])
    np.testing.assert_array_equal(expanded, np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]))

    corrected = _correct_ancient_lambdas(
        [0.2, 0.2, 0.2, 0.9, 0.9, 0.9],
        [1, 1, 1],
        [1, 2, 3],
    )
    assert corrected == [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


def test_msmc_im_numeric_helpers_match_vendored_functions() -> None:
    left_boundaries = np.array([0.0, 100.0, 250.0], dtype=np.float64)
    right_boundaries = np.array([100.0, 250.0, 500.0], dtype=np.float64)
    lambdas = np.array([1.0e-4, 2.5e-4, 3.0e-4], dtype=np.float64)
    N1 = np.array([12000.0, 13000.0, 14000.0], dtype=np.float64)
    N2 = np.array([18000.0, 17500.0, 17000.0], dtype=np.float64)
    m = np.array([1.0e-5, 2.0e-5, 3.0e-5], dtype=np.float64)
    x0 = [1, 0, 0, 0, 0]

    np.testing.assert_allclose(
        _tmrca_from_msmc(left_boundaries, left_boundaries, lambdas),
        VENDOR.read_tmrcadist_from_MSMC(
            left_boundaries,
            left_boundaries.tolist(),
            lambdas.tolist(),
        ),
    )

    q_vendor = np.asarray(VENDOR.makeQ(m[1], N1[1], N2[1]), dtype=np.float64)
    q_native = _make_Q(m[1], N1[1], N2[1])
    np.testing.assert_allclose(q_native, q_vendor)

    dt = left_boundaries[2] - left_boundaries[1]
    np.testing.assert_allclose(
        _make_Qexp(q_native, dt),
        np.asarray(
            VENDOR.makeQexp(VENDOR.makeQ(m[1], N1[1], N2[1]), dt),
            dtype=np.float64,
        ),
    )

    vectors_native = _propagate_state_vectors(x0, left_boundaries, N1, N2, m)
    vectors_vendor = VENDOR.makeQpropagator_xvector_Symmlist(
        x0,
        left_boundaries.tolist(),
        N1.tolist(),
        N2.tolist(),
        m.tolist(),
    )
    for ours, ref in zip(vectors_native, vectors_vendor):
        np.testing.assert_allclose(
            np.asarray(ours, dtype=np.float64),
            np.asarray(ref, dtype=np.float64),
        )

    t = 175.0
    np.testing.assert_allclose(
        _compute_tmrca_density(t, vectors_native, left_boundaries, N1, N2),
        VENDOR.computeTMRCA_t0_DynamicN_caltbound_mlist(
            t,
            vectors_vendor,
            left_boundaries.tolist(),
            N1.tolist(),
            N2.tolist(),
        ),
    )

    np.testing.assert_allclose(
        _cumulative_migration(right_boundaries, m),
        VENDOR.cumulative_Symmigproportion(right_boundaries.tolist(), m.tolist()),
    )
