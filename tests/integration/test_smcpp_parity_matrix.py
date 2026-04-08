"""Tracked native-vs-upstream SMC++ parity matrix without msprime dependency."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.io import read_smcpp_input
from smckit.tl._smcpp import (
    _collect_expectation_stats,
    _preprocess_onepop_records,
    _resolve_upstream_smcpp_python,
    _run_upstream_smcpp_fixed_model_stats,
    compute_hmm_params,
    smcpp,
)


pytestmark = pytest.mark.skipif(
    _resolve_upstream_smcpp_python() is None,
    reason="Upstream SMC++ side environment not available",
)


ROOT = Path(__file__).resolve().parents[2]
LARGER_FIXTURE = ROOT / "tests" / "data" / "smcpp_onepop_larger.smc"
N_INTERVALS = 4


def _small_control_data() -> SmcData:
    observations = []
    for i in range(120):
        observations.append((4999, 0, 0))
        observations.append((1, i % 3, (i * 2) % 6))
    return SmcData(
        uns={
            "records": [{"name": "default_onepop", "observations": observations, "n_undist": 5}],
            "n_undist": 5,
        },
    )


def _shared_grid_metrics(native: dict, upstream: dict, *, n_grid: int = 200) -> dict[str, float]:
    native_time = np.asarray(native["time"], dtype=float)
    upstream_time = np.asarray(upstream["time"], dtype=float)
    t_min = max(
        float(np.min(native_time[native_time > 0])),
        float(np.min(upstream_time[upstream_time > 0])),
    )
    t_max = min(float(np.max(native_time)), float(np.max(upstream_time)))
    grid = np.geomspace(t_min, t_max, n_grid)

    def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, query, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    native_ne = _step_eval(native_time, np.asarray(native["ne"], dtype=float), grid)
    upstream_ne = _step_eval(upstream_time, np.asarray(upstream["ne"], dtype=float), grid)
    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    return {
        "log_corr": float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1]),
        "scale_ratio": float(np.median(native_ne) / np.median(upstream_ne)),
        "median_rel": float(np.median(rel)),
        "median_log10_error": float(np.median(np.abs(np.log10(native_ne / upstream_ne)))),
    }


def _run_pair(data: SmcData) -> tuple[dict, dict]:
    params = {
        "n_intervals": N_INTERVALS,
        "regularization": 10.0,
        "max_iterations": 2,
        "seed": 42,
        "generation_time": 1.0,
    }
    native = smcpp(copy.deepcopy(data), implementation="native", **params).results["smcpp"]
    upstream = smcpp(copy.deepcopy(data), implementation="upstream", **params).results["smcpp"]
    return native, upstream


def _native_fixed_model_stats(data: SmcData, upstream: dict) -> dict:
    fit = _preprocess_onepop_records(data.uns["records"], int(data.uns["n_undist"]))
    ne = np.asarray(upstream["ne"], dtype=float)
    n0 = float(upstream["n0"])
    eta = ne / (2.0 * n0)
    t = np.r_[0.0, np.asarray(upstream["time"], dtype=float)]
    hs = np.asarray(upstream["upstream"]["hidden_states"]["pop1"], dtype=float)
    hp = compute_hmm_params(
        eta,
        int(data.uns["n_undist"]),
        t,
        float(upstream["theta"]),
        float(upstream["rho"]),
        n_distinguished=2,
        hidden_states=hs,
        polarization_error=0.5,
        observation_scale=float(upstream["upstream"]["alpha"]),
    )
    stats = _collect_expectation_stats(hp, fit)
    return {
        "gamma0": np.asarray(stats.gamma0, dtype=float),
        "gamma_sums": {k: np.asarray(v, dtype=float) for k, v in stats.gamma_sums.items()},
        "xisum": np.asarray(stats.xisum, dtype=float),
        "log_likelihood": float(stats.log_likelihood),
    }


def _fixed_model_stats_pair(data: SmcData) -> tuple[dict, dict]:
    _, upstream = _run_pair(data)
    upstream_stats = _run_upstream_smcpp_fixed_model_stats(
        copy.deepcopy(data),
        model_dict=copy.deepcopy(upstream["upstream"]["model"]),
        alpha=float(upstream["upstream"]["alpha"]),
        n_intervals=N_INTERVALS,
        mu=1.25e-8,
        recombination_rate=1e-8,
        regularization=10.0,
        seed=42,
    )
    native_stats = _native_fixed_model_stats(data, upstream)
    return native_stats, upstream_stats


def test_smcpp_small_control_fixture_stays_strict() -> None:
    native, upstream = _run_pair(_small_control_data())
    metrics = _shared_grid_metrics(native, upstream)

    assert native["n_distinguished"] == 2
    assert upstream["n_distinguished"] == 2
    assert native["preprocessing"]["applied"] is True
    assert native["observation_scale"] == 100.0
    assert upstream["optimization"]["history"]
    assert metrics["log_corr"] >= 0.999
    assert 0.998 < metrics["scale_ratio"] < 1.002
    assert metrics["median_log10_error"] < 0.001
    assert metrics["median_rel"] < 0.002


def test_smcpp_larger_tracked_fixture_stays_strict() -> None:
    data = read_smcpp_input(LARGER_FIXTURE)
    native, upstream = _run_pair(data)
    metrics = _shared_grid_metrics(native, upstream)

    assert native["implementation"] == "native"
    assert upstream["implementation"] == "upstream"
    assert native["preprocessing"]["applied"] is True
    assert native["observation_scale"] == 100.0
    assert upstream["optimization"]["history"]
    assert metrics["log_corr"] >= 0.999
    assert 0.998 < metrics["scale_ratio"] < 1.002
    assert metrics["median_log10_error"] < 0.001
    assert metrics["median_rel"] < 0.002


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("small_control", _small_control_data()),
        ("larger_tracked_fixture", read_smcpp_input(LARGER_FIXTURE)),
    ],
)
def test_smcpp_fixed_model_stats_match_upstream(label: str, data: SmcData) -> None:
    del label
    native_stats, upstream_stats = _fixed_model_stats_pair(data)

    upstream_gamma0 = np.asarray(upstream_stats["gamma0"], dtype=float).reshape(-1)
    gamma0_rel = np.max(
        np.abs(native_stats["gamma0"] - upstream_gamma0) / np.maximum(np.abs(upstream_gamma0), 1e-12)
    )

    native_xisum = np.asarray(native_stats["xisum"], dtype=float)
    upstream_xisum = np.asarray(upstream_stats["xisum"], dtype=float)
    xisum_rel = np.max(
        np.abs(native_xisum - upstream_xisum) / np.maximum(np.abs(upstream_xisum), 1e-12)
    )

    loglik_abs = abs(float(native_stats["log_likelihood"]) - float(upstream_stats["log_likelihood"]))

    assert gamma0_rel < 1e-3
    assert xisum_rel < 1e-3
    assert loglik_abs < 5e-4
