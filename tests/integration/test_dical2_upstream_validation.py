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


pytestmark = pytest.mark.skipif(
    not smckit.upstream.status("dical2")["runtime_ready"],
    reason="Java runtime not available",
)


def test_dical2_upstream_backend_runs_end_to_end() -> None:
    data = read_dical2(
        sequences=ROOT / "test.vcf",
        param_file=ROOT / "test.param",
        demo_file=ROOT / "exp.demo",
        rates_file=ROOT / "exp.rates",
        config_file=ROOT / "exp.config",
        reference_file=ROOT / "test.fa",
    )
    result = dical2(
        data,
        implementation="upstream",
        upstream_options=EXP_NATIVE_OPTIONS,
        **EXP_COMMON,
    )
    res = result.results["dical2"]
    assert res["implementation"] == "upstream"
    assert res["backend"] == "upstream"
    assert np.isfinite(res["log_likelihood"])
    assert len(res["em_path"]) >= 1
    assert res["resolved_options"]["interval_type"] == "loguniform"
    assert res["resolved_options"]["number_iterations_mstep"] == 2


def test_dical2_native_loguniform_exp_matches_upstream_at_oracle_params() -> None:
    upstream_data = read_dical2(
        sequences=ROOT / "test.vcf",
        param_file=ROOT / "test.param",
        demo_file=ROOT / "exp.demo",
        rates_file=ROOT / "exp.rates",
        config_file=ROOT / "exp.config",
        reference_file=ROOT / "test.fa",
    )
    upstream = dical2(
        upstream_data,
        implementation="upstream",
        upstream_options=EXP_NATIVE_OPTIONS,
        **EXP_COMMON,
    ).results["dical2"]

    native_eval_data = read_dical2(
        sequences=ROOT / "test.vcf",
        param_file=ROOT / "test.param",
        demo_file=ROOT / "exp.demo",
        rates_file=ROOT / "exp.rates",
        config_file=ROOT / "exp.config",
        reference_file=ROOT / "test.fa",
    )
    native_eval = dical2(
        native_eval_data,
        implementation="native",
        n_em_iterations=0,
        start_point=upstream["best_params"],
        native_options=EXP_NATIVE_OPTIONS,
        loci_per_hmm_step=EXP_COMMON["loci_per_hmm_step"],
        composite_mode=EXP_COMMON["composite_mode"],
        bounds=EXP_COMMON["bounds"],
        seed=EXP_COMMON["seed"],
    ).results["dical2"]

    np.testing.assert_allclose(
        native_eval["interval_boundaries"],
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
    assert native_eval["resolved_options"]["interval_type"] == upstream["resolved_options"]["interval_type"]
    assert native_eval["resolved_options"]["interval_params"] == upstream["resolved_options"]["interval_params"]
    assert native_eval["resolved_options"]["nm_fraction"] == pytest.approx(0.2)
    assert native_eval["core_type"] == "ode"
    assert abs(native_eval["log_likelihood"] - upstream["log_likelihood"]) < 0.25
