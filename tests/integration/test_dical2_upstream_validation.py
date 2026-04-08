"""Integration coverage for the upstream-backed diCal2 wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io import read_dical2
from smckit.tl import dical2


ROOT = Path("vendor/diCal2/examples/fromReadme")
EXP_NATIVE_OPTIONS = {
    "interval_type": "logUniform",
    "interval_params": "11,0.01,4",
    "number_iterations_mstep": 2,
    "disableCoordinateWiseMStep": True,
}
IM_NATIVE_OPTIONS = {
    "interval_type": "logUniform",
    "interval_params": "11,0.01,4",
    "number_iterations_mstep": 2,
}
EXP_COMMON = {
    "meta_start_file": str(ROOT / "exp.rand"),
    "meta_num_iterations": 2,
    "meta_keep_best": 1,
    "meta_num_points": 3,
    "loci_per_hmm_step": 3,
    "composite_mode": "lol",
    "n_em_iterations": 2,
    "bounds": "1.00001,1000;0.01,0.06;0.01,0.23;0.02,2;0.5,4",
    "seed": 541816302422,
}
IM_COMMON = {
    "meta_start_file": str(ROOT / "IM.rand"),
    "meta_num_iterations": 2,
    "meta_keep_best": 1,
    "meta_num_points": 3,
    "loci_per_hmm_step": 4,
    "composite_mode": "pcl",
    "n_em_iterations": 2,
    "bounds": "0.01,0.32;0.05,1.0001;0.05,5;0.05,5;0.02,2;0.9,5;0.1,500",
    "seed": 60643714832,
}


pytestmark = pytest.mark.skipif(
    not smckit.upstream.status("dical2")["runtime_ready"],
    reason="Java runtime not available",
)


def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def _read_exp_data():
    return read_dical2(
        sequences=ROOT / "test.vcf",
        param_file=ROOT / "test.param",
        demo_file=ROOT / "exp.demo",
        rates_file=ROOT / "exp.rates",
        config_file=ROOT / "exp.config",
        reference_file=ROOT / "test.fa",
    )


def _read_im_data():
    return read_dical2(
        sequences=ROOT / "test.vcf",
        param_file=ROOT / "test.param",
        demo_file=ROOT / "IM.demo",
        config_file=ROOT / "IM.config",
        reference_file=ROOT / "test.fa",
    )


def _exp_native_options(*, record_meta_trace: bool = False) -> dict[str, object]:
    options = dict(EXP_NATIVE_OPTIONS)
    if record_meta_trace:
        options["record_meta_trace"] = True
    return options


def _run_exp_upstream() -> dict:
    return dical2(
        _read_exp_data(),
        implementation="upstream",
        upstream_options=EXP_NATIVE_OPTIONS,
        **EXP_COMMON,
    ).results["dical2"]


def _run_exp_native(*, record_meta_trace: bool = False) -> dict:
    return dical2(
        _read_exp_data(),
        implementation="native",
        native_options=_exp_native_options(record_meta_trace=record_meta_trace),
        **EXP_COMMON,
    ).results["dical2"]


def test_dical2_upstream_backend_runs_end_to_end() -> None:
    res = _run_exp_upstream()
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.isfinite(res["log_likelihood"])
    assert len(res["em_path"]) >= 1
    assert res["resolved_options"]["interval_type"] == "loguniform"
    assert res["resolved_options"]["number_iterations_mstep"] == 2
    assert np.all(np.asarray(res["time"]) >= 0.0)
    assert np.all(np.asarray(res["ne"]) > 0.0)
    np.testing.assert_allclose(np.asarray(res["best_params"]), np.asarray(res["ordered_params"]))


def test_dical2_upstream_im_backend_runs_end_to_end() -> None:
    res = dical2(
        _read_im_data(),
        implementation="upstream",
        upstream_options=IM_NATIVE_OPTIONS,
        **IM_COMMON,
    ).results["dical2"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.isfinite(res["log_likelihood"])
    assert len(res["em_path"]) >= 1
    assert res["resolved_options"]["interval_type"] == "loguniform"
    assert res["resolved_options"]["number_iterations_mstep"] == 2
    np.testing.assert_allclose(np.asarray(res["best_params"]), np.asarray(res["ordered_params"]))
    assert len(np.asarray(res["time"])) > 0
    assert len(res["structured_ne"]) > 0


def test_dical2_native_loguniform_exp_runs_independently() -> None:
    upstream = _run_exp_upstream()
    native = _run_exp_native()

    np.testing.assert_allclose(
        native["interval_boundaries"],
        np.array(
            [
                0.0,
                0.01,
                0.0172405417,
                0.0297236279,
                0.0512451448,
                0.0883494058,
                0.1523191625,
                0.2626064874,
                0.4527478100,
                0.7805617510,
                1.3457307492,
                2.3201127105,
                4.0,
                1000.0,
            ],
            dtype=np.float64,
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    assert native["resolved_options"]["interval_type"] == upstream["resolved_options"]["interval_type"]
    assert native["resolved_options"]["interval_params"] == upstream["resolved_options"]["interval_params"]
    assert native["resolved_options"]["nm_fraction"] == pytest.approx(0.2)
    assert native["core_type"] == "ode"
    assert native["initialization"] is None
    assert np.isfinite(native["log_likelihood"])
    assert np.isfinite(np.asarray(native["best_params"])).all()
    assert native["n_iterations"] >= 1


def test_dical2_native_loguniform_exp_records_meta_trace() -> None:
    native = _run_exp_native(record_meta_trace=True)

    trace = native["meta_trace"]
    assert trace is not None
    assert len(trace) == EXP_COMMON["meta_num_iterations"]

    generation_zero = trace[0]
    generation_one = trace[1]

    assert len(generation_zero["starting_points"]) == EXP_COMMON["meta_num_points"]
    assert len(generation_zero["runs"]) == EXP_COMMON["meta_num_points"]
    assert len(generation_zero["offspring"]) == EXP_COMMON["meta_num_points"] - EXP_COMMON["meta_keep_best"]
    assert len(generation_zero["next_generation"]) == EXP_COMMON["meta_num_points"]
    np.testing.assert_allclose(
        np.asarray(generation_zero["next_generation"]),
        np.asarray(generation_one["starting_points"]),
    )

    for generation in trace:
        best_from_runs = max(run["log_likelihood"] for run in generation["runs"])
        assert generation["best_log_likelihood"] == pytest.approx(best_from_runs)

    np.testing.assert_allclose(
        np.asarray(generation_one["best_params"]),
        np.asarray(native["best_params"]),
    )
    assert generation_one["best_log_likelihood"] == pytest.approx(native["log_likelihood"])


@pytest.mark.xfail(
    reason="Native diCal2 full exp meta-start search is not yet interchangeable with upstream.",
    strict=False,
)
def test_dical2_native_loguniform_exp_full_search_matches_upstream_fit() -> None:
    upstream = _run_exp_upstream()
    native = _run_exp_native()

    np.testing.assert_allclose(
        np.asarray(native["best_params"]),
        np.asarray(upstream["best_params"]),
        rtol=1e-8,
        atol=1e-8,
    )
    assert native["log_likelihood"] == pytest.approx(upstream["log_likelihood"], abs=1e-8)
    assert float(np.asarray(native["growth_rates"], dtype=float)[0]) == pytest.approx(
        float(np.asarray(upstream["growth_rates"], dtype=float)[0]),
        abs=1e-8,
    )


def test_dical2_native_im_runs_independently() -> None:
    upstream = dical2(
        _read_im_data(),
        implementation="upstream",
        upstream_options=IM_NATIVE_OPTIONS,
        **IM_COMMON,
    ).results["dical2"]
    native = dical2(
        _read_im_data(),
        implementation="native",
        native_options=IM_NATIVE_OPTIONS,
        **IM_COMMON,
    ).results["dical2"]

    assert native["resolved_options"]["interval_type"] == upstream["resolved_options"]["interval_type"]
    assert native["resolved_options"]["interval_params"] == upstream["resolved_options"]["interval_params"]
    assert native["resolved_options"]["nm_fraction"] == pytest.approx(0.2)
    assert native["initialization"] is None
    assert np.isfinite(native["log_likelihood"])
    assert np.isfinite(np.asarray(native["best_params"])).all()
    assert native["n_iterations"] >= 1
    assert len(np.asarray(native["time"])) == len(np.asarray(upstream["time"]))
    assert len(native["structured_ne"]) == len(upstream["structured_ne"])


def test_dical2_native_loguniform_exp_matches_upstream_curve_at_oracle_params() -> None:
    upstream = _run_exp_upstream()

    native_eval_data = _read_exp_data()
    native = dical2(
        native_eval_data,
        implementation="native",
        n_em_iterations=0,
        start_point=upstream["best_params"],
        native_options=_exp_native_options(),
        loci_per_hmm_step=EXP_COMMON["loci_per_hmm_step"],
        composite_mode=EXP_COMMON["composite_mode"],
        bounds=EXP_COMMON["bounds"],
        seed=EXP_COMMON["seed"],
    ).results["dical2"]

    t_min = max(
        float(np.min(np.asarray(native["time"])[np.asarray(native["time"]) > 0])),
        float(np.min(np.asarray(upstream["time"])[np.asarray(upstream["time"]) > 0])),
    )
    t_max = min(float(np.max(np.asarray(native["time"]))), float(np.max(np.asarray(upstream["time"]))))
    grid = np.geomspace(t_min, t_max, 200)

    native_ne = _step_eval(np.asarray(native["time"]), np.asarray(native["ne"]), grid)
    upstream_ne = _step_eval(np.asarray(upstream["time"]), np.asarray(upstream["ne"]), grid)
    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    log_corr = float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1])
    scale_ratio = float(np.median(native_ne) / np.median(upstream_ne))
    median_log10_error = float(np.median(np.abs(np.log10(native_ne / upstream_ne))))

    assert np.asarray(native["best_params"]) == pytest.approx(np.asarray(upstream["best_params"]))
    assert log_corr >= 0.999999
    assert 0.999 < scale_ratio < 1.001
    assert median_log10_error < 1e-6
    assert float(np.median(rel)) < 1e-6


def test_dical2_native_im_matches_upstream_demography_at_oracle_params() -> None:
    upstream = dical2(
        _read_im_data(),
        implementation="upstream",
        upstream_options=IM_NATIVE_OPTIONS,
        **IM_COMMON,
    ).results["dical2"]
    native = dical2(
        _read_im_data(),
        implementation="native",
        start_point=upstream["best_params"],
        n_em_iterations=0,
        native_options=IM_NATIVE_OPTIONS,
        loci_per_hmm_step=IM_COMMON["loci_per_hmm_step"],
        composite_mode=IM_COMMON["composite_mode"],
        bounds=IM_COMMON["bounds"],
        seed=IM_COMMON["seed"],
    ).results["dical2"]

    assert np.asarray(native["best_params"]) == pytest.approx(np.asarray(upstream["best_params"]))
    np.testing.assert_allclose(np.asarray(native["time"]), np.asarray(upstream["time"]))
    np.testing.assert_allclose(np.asarray(native["ordered_params"]), np.asarray(upstream["ordered_params"]))
    assert len(native["structured_ne"]) == len(upstream["structured_ne"])
    for native_epoch, upstream_epoch in zip(native["structured_ne"], upstream["structured_ne"]):
        np.testing.assert_allclose(np.asarray(native_epoch), np.asarray(upstream_epoch))
