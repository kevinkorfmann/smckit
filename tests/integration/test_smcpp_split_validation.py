"""Native-versus-upstream validation for SMC++ clean-split inference."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.tl._smcpp import (
    _expected_joint_sfs_clean_split,
    _history_value,
    _joint_observation_probability,
    _joint_sfs_to_jcsfs,
    _piecewise_model_history,
    _resolve_upstream_smcpp_python,
    _run_upstream_smcpp_joint_csfs_oracle,
    _run_upstream_smcpp_model_stepwise_oracle,
    _run_upstream_smcpp_split,
    _split_distinguished_pair_emission,
    smcpp,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.oracle,
    pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    ),
]


def _model(population: str, eta: list[float]) -> dict:
    return {
        "class": "SMCModel",
        "knots": [0.01, 0.5, 1.5],
        "N0": 10_000.0,
        "spline_class": "Piecewise",
        "y": np.log(eta).tolist(),
        "pid": population,
    }


@pytest.mark.parametrize(
    "spline_class",
    ["Piecewise", "CubicSpline", "PChipSpline", "AkimaSpline", "BSpline"],
)
def test_native_spline_discretization_matches_upstream(spline_class: str) -> None:
    knots = [0.01, 0.05, 0.2, 0.8, 2.0]
    value_count = len(knots) + (2 if spline_class == "BSpline" else 0)
    model = {
        "class": "SMCModel",
        "knots": knots,
        "N0": 10_000.0,
        "spline_class": spline_class,
        "y": np.linspace(-0.4, 0.6, value_count).tolist(),
        "pid": "pop-a",
    }
    time, upstream_eta = _run_upstream_smcpp_model_stepwise_oracle(model)
    change_times, native_eta = _piecewise_model_history(model)
    evaluated = np.asarray(
        [_history_value(change_times, native_eta, np.nextafter(value, -np.inf)) for value in time]
    )

    np.testing.assert_allclose(evaluated, upstream_eta, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize(
    ("n_undistinguished", "n_distinguished", "rtol"),
    [
        # Upstream averages the together-lineage transition over random
        # coalescence times; this tolerance covers its fixed-seed K=10,000
        # Monte Carlo error while the native calculation is deterministic.
        ((3, 2), (2, 0), 1.5e-3),
        ((3, 2), (1, 1), 2e-6),
    ],
)
def test_native_joint_csfs_matches_upstream_oracle(
    n_undistinguished: tuple[int, int],
    n_distinguished: tuple[int, int],
    rtol: float,
) -> None:
    model1 = _model("pop-a", [1.0, 2.0, 0.8])
    model2 = _model("pop-b", [1.5, 0.7, 1.2])
    total_samples = tuple(n_undistinguished[index] + n_distinguished[index] for index in range(2))

    native = _joint_sfs_to_jcsfs(
        _expected_joint_sfs_clean_split(total_samples, model1, model2, 0.3),
        n_distinguished,
    )
    upstream = _run_upstream_smcpp_joint_csfs_oracle(
        model1=model1,
        model2=model2,
        split=0.3,
        n_undistinguished=n_undistinguished,
        n_distinguished=n_distinguished,
        quadrature_points=10_000,
        seed=17,
    )

    np.testing.assert_allclose(native, upstream, rtol=rtol, atol=2e-7)
    np.testing.assert_allclose(native.sum(), upstream.sum(), rtol=rtol, atol=2e-7)


@pytest.mark.parametrize(
    ("distinguished", "observations"),
    [
        (
            (2, 0),
            [
                ((0, 0, 3), (0, 0, 2)),
                ((1, 1, 2), (0, 1, 1)),
                ((-1, 0, 3), (0, 0, 2)),
                ((0, 0, 0), (0, 0, 0)),
                ((1, 0, 0), (0, 0, 0)),
            ],
        ),
        (
            (1, 1),
            [
                ((0, 0, 3), (0, 0, 2)),
                ((1, 1, 2), (0, 1, 1)),
                ((-1, 0, 3), (1, 0, 1)),
                ((0, 0, 0), (0, 0, 0)),
                ((1, 0, 0), (0, 0, 0)),
            ],
        ),
    ],
)
def test_missing_and_downsampled_observation_mapping_matches_upstream(
    distinguished: tuple[int, int],
    observations: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> None:
    full_undistinguished = (3, 2)
    header = {
        "pids": ["pop-a", "pop-b"],
        "dist": [
            [["pop-a-distinguished", index] for index in range(distinguished[0])],
            [["pop-b-distinguished", index] for index in range(distinguished[1])],
        ],
        "undist": [
            [["pop-a", index] for index in range(full_undistinguished[0])],
            [["pop-b", index] for index in range(full_undistinguished[1])],
        ],
    }
    data = SmcData(
        uns={
            "n_populations": 2,
            "populations": ["pop-a", "pop-b"],
            "pids": ["pop-a", "pop-b"],
            "joint_observations": [
                (150_000 if index == 0 else 1, observation)
                for index, observation in enumerate(observations)
            ],
            "n_undist_by_population": list(full_undistinguished),
            "n_distinguished_by_population": list(distinguished),
            "smcpp_header": header,
            "total_sites": 150_000 + len(observations) - 1,
        }
    )
    model1 = {"model": _model("pop-a", [1.0, 2.0, 0.8]), "theta": 2.5e-4, "rho": 2e-4}
    model2 = {"model": _model("pop-b", [1.5, 0.7, 1.2]), "theta": 2.5e-4, "rho": 2e-4}
    upstream = _run_upstream_smcpp_split(
        data,
        marginal_models=(model1, model2),
        generation_time=29.0,
        regularization=0.0,
        max_iterations=1,
        seed=17,
        fixed_split=0.3,
        fixed_log_scale=0.0,
    )

    component = next(iter(upstream["components"].values()))
    emission = np.asarray(component["emission"], dtype=float)[0].reshape(
        distinguished[0] + 1,
        full_undistinguished[0] + 1,
        distinguished[1] + 1,
        full_undistinguished[1] + 1,
    )
    reduced_emission = _split_distinguished_pair_emission(
        model1["model"],
        0.3,
        distinguished,
        theta=2.5e-4,
    )
    for observation in observations:
        key = ",".join(str(value) for population in observation for value in population)
        expected = float(upstream["emission_probabilities"][key][0])
        actual = _joint_observation_probability(
            emission,
            observation,
            distinguished,
            full_undistinguished,
            polarization_error=0.5,
            reduced_emission=reduced_emission,
        )
        assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)


def test_native_split_fit_matches_upstream_coordinate_updates() -> None:
    observations = []
    for index in range(30):
        observations.extend(
            [
                (4_999, ((0, 0, 3), (0, 0, 2))),
                (
                    1,
                    (
                        (index % 3, (index * 2) % 4, 3),
                        (0, index % 3, 2),
                    ),
                ),
            ]
        )
    header = {
        "pids": ["pop-a", "pop-b"],
        "dist": [[["distinguished", 0], ["distinguished", 1]], []],
        "undist": [
            [["pop-a", index] for index in range(3)],
            [["pop-b", index] for index in range(2)],
        ],
    }
    data = SmcData(
        uns={
            "n_populations": 2,
            "populations": ["pop-a", "pop-b"],
            "pids": ["pop-a", "pop-b"],
            "joint_observations": observations,
            "n_undist_by_population": [3, 2],
            "n_distinguished_by_population": [2, 0],
            "smcpp_header": header,
            "total_sites": sum(span for span, _ in observations),
        }
    )
    model1 = {"model": _model("pop-a", [1.0, 2.0, 0.8]), "theta": 2.5e-4, "rho": 2e-4}
    model2 = {"model": _model("pop-b", [1.5, 0.7, 1.2]), "theta": 2.5e-4, "rho": 2e-4}

    native = smcpp(
        copy.deepcopy(data),
        implementation="native",
        split_models=(model1, model2),
        max_iterations=100,
        seed=17,
    ).results["smcpp"]
    upstream = smcpp(
        copy.deepcopy(data),
        implementation="upstream",
        split_models=(model1, model2),
        max_iterations=100,
        seed=17,
    ).results["smcpp"]

    assert native["split"] == pytest.approx(upstream["split"], abs=1e-12)
    for native_model, upstream_model in zip(
        native["population_models"],
        upstream["population_models"],
    ):
        np.testing.assert_allclose(native_model["ne"], upstream_model["ne"], rtol=1e-12)
    per_site_difference = (
        abs(native["log_likelihood"] - upstream["log_likelihood"]) / data.uns["total_sites"]
    )
    assert per_site_difference < 1e-5
