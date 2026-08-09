"""Native PSMC+ parameter layout and fitting-engine tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smckit import io
from smckit.tl import PSMCPlusOptions, psmcplus
from smckit.tl._psmcplus_native import (
    decode_psmcplus_native,
    fit_psmcplus_native,
    parse_psmcplus_segments,
)

ROOT = Path(__file__).resolve().parents[2]
CONSTPOP = ROOT / "vendor/PSMCplus/simulations/constpopsize.mhs"
EXPECTED = ROOT / "tests/data/psmcplus/constpop_D4_1iter.final_parameters.txt"
RATE_MAP_EXPECTED = ROOT / "tests/data/psmcplus/rate_map_D4_1iter.final_parameters.txt"


def test_segment_layout_matches_original_grouping_and_fixed_grammar() -> None:
    layout = parse_psmcplus_segments(
        "4*1,4*0,1*4",
        12,
        [1, 1, 1, 1, 2, 3],
    )
    assert layout.widths == (1, 1, 1, 1, 4, 4)
    assert layout.fixed == (False, False, False, False, True, False)
    np.testing.assert_array_equal(layout.free_initial, [1, 1, 1, 1, 3])
    np.testing.assert_array_equal(
        layout.expand(np.array([5, 6, 7, 8, 9], dtype=float)),
        [5, 6, 7, 8, 2, 2, 2, 2, 9, 9, 9, 9],
    )


@pytest.mark.parametrize(
    ("pattern", "initial", "message"),
    [
        ("2*1", None, "not 4"),
        ("1", None, "token"),
        ("1*-1,3*1", None, "non-negative"),
        ("4*1", [1, 2], "requires 4"),
        ("4*1", [1, 1, 0, 1], "positive"),
    ],
)
def test_invalid_segment_layouts_are_rejected(
    pattern: str,
    initial: list[float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_psmcplus_segments(pattern, 4, initial)


@pytest.mark.slow
def test_multi_file_fixed_model_aggregates_likelihood_with_multiple_cores() -> None:
    one = fit_psmcplus_native(
        [CONSTPOP],
        number_states=4,
        bin_size=100,
        lambda_segments="4*0",
        lambda_initial=[1],
        scaled_recombination_rate=0.0005,
        estimate_rho=False,
        iterations=1,
        likelihood_threshold=0,
        cores=1,
    )
    two = fit_psmcplus_native(
        [CONSTPOP, CONSTPOP],
        number_states=4,
        bin_size=100,
        lambda_segments="4*0",
        lambda_initial=[1],
        scaled_recombination_rate=0.0005,
        estimate_rho=False,
        iterations=1,
        likelihood_threshold=0,
        cores=2,
    )
    assert two.final_log_likelihood == pytest.approx(2 * one.final_log_likelihood, rel=1e-14)
    np.testing.assert_array_equal(two.lambda_values, one.lambda_values)


@pytest.mark.slow
def test_local_mutation_and_recombination_map_fit_matches_upstream() -> None:
    fixtures = ROOT / "tests/data/psmcplus"
    fit = fit_psmcplus_native(
        [fixtures / "preprocessing_masked.mhs"],
        mutation_map_paths=[fixtures / "preprocessing_mutation.bed"],
        recombination_map_paths=[fixtures / "workflow_recombination.bed"],
        recombination_map_downsamples=2,
        number_states=4,
        bin_size=10,
        scaled_recombination_rate=0.0005,
        estimate_rho=False,
        iterations=1,
        likelihood_threshold=0,
    )
    expected = np.loadtxt(RATE_MAP_EXPECTED)
    expected_lambda = expected[:, 2] * fit.theta / 4.0
    # LAPACK/Numba ordering differs by <5e-12 on macOS ARM64.
    assert fit.final_log_likelihood == pytest.approx(-7.420073963148187, abs=1e-11)
    assert fit.likelihood_change == pytest.approx(0.019001157859748652, abs=1e-11)
    np.testing.assert_allclose(fit.lambda_values, expected_lambda, rtol=2e-9, atol=1e-12)
    np.testing.assert_allclose(
        fit.sequences[0].recombination_factors,
        [0.64, 1.55, 1.8],
    )


@pytest.mark.slow
def test_native_extension_estimates_rho_with_local_recombination_map() -> None:
    fixtures = ROOT / "tests/data/psmcplus"
    fit = fit_psmcplus_native(
        [fixtures / "preprocessing_masked.mhs"],
        mutation_map_paths=[fixtures / "preprocessing_mutation.bed"],
        recombination_map_paths=[fixtures / "workflow_recombination.bed"],
        recombination_map_downsamples=2,
        number_states=4,
        bin_size=10,
        scaled_recombination_rate=0.0005,
        estimate_rho=True,
        iterations=1,
        likelihood_threshold=0,
    )
    assert fit.optimization_success == (True,)
    assert np.isfinite(fit.rho) and fit.rho > 0
    assert fit.final_log_likelihood >= fit.likelihoods[-1] - 1e-10


@pytest.mark.slow
def test_rate_map_decode_matches_upstream_and_exposes_corrected_marginal() -> None:
    fixtures = ROOT / "tests/data/psmcplus"
    decode = decode_psmcplus_native(
        fixtures / "preprocessing_masked.mhs",
        mutation_map_path=fixtures / "preprocessing_mutation.bed",
        recombination_map_path=fixtures / "workflow_recombination.bed",
        recombination_map_downsamples=2,
        number_states=4,
        bin_size=10,
        scaled_recombination_rate=0.0005,
        lambda_initial=[1, 1, 1, 1],
        downsample=1,
    )
    with np.load(fixtures / "rate_map_decode_oracle_v1.npz") as oracle:
        # LAPACK/Numba ordering differs by <4e-12 on macOS ARM64.
        assert decode.log_likelihood == pytest.approx(float(oracle["log_likelihood"]), abs=1e-11)
        np.testing.assert_allclose(decode.posterior, oracle["posterior"], atol=4e-14)
        np.testing.assert_allclose(
            decode.marginal_recombination,
            oracle["marginal_recombination"],
            atol=2e-15,
        )
    assert not np.allclose(
        decode.corrected_marginal_recombination,
        decode.marginal_recombination,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.slow
def test_final_time_factor_decode_matches_upstream_transition_grid_contract() -> None:
    fixtures = ROOT / "tests/data/psmcplus"
    decode = decode_psmcplus_native(
        fixtures / "preprocessing_masked.mhs",
        number_states=4,
        bin_size=10,
        scaled_recombination_rate=0.0005,
        lambda_initial=[0.7, 0.9, 1.4, 1.1],
        final_time_factor=3.0,
        downsample=1,
    )

    # Frozen from the pinned upstream commit. Upstream uses the custom grid for
    # emissions/output while retaining its default grid for transitions.
    assert decode.boundaries[-1] == decode.boundaries[-2] * 3.0
    assert decode.log_likelihood == pytest.approx(-6.738403574763081, abs=5e-12)
    np.testing.assert_allclose(
        decode.posterior,
        [
            [
                4.0492251933703016e-07,
                0.0011028984702955156,
                0.6428763799723198,
                0.3560203166348654,
            ],
            [
                3.109287867693253e-07,
                0.0010922581367361457,
                0.6434697719505399,
                0.35543765898393725,
            ],
            [3.198475801174263e-07, 0.0010980960783360874, 0.644295431168896, 0.3546061529051877],
            [8.780260297657805e-07, 0.001166174690514078, 0.6466554470086726, 0.3521775002747836],
            [9.812387871022075e-06, 0.001374079859371954, 0.6486856091575597, 0.34993049859519726],
        ],
        rtol=2e-13,
        atol=2e-14,
    )


@pytest.mark.slow
def test_one_iteration_native_fit_matches_frozen_upstream_result() -> None:
    fit = fit_psmcplus_native(
        [CONSTPOP],
        number_states=4,
        bin_size=100,
        iterations=1,
        likelihood_threshold=0,
    )
    expected = np.loadtxt(EXPECTED)
    expected_text = EXPECTED.read_text(encoding="utf-8")

    def header(label: str) -> float:
        for line in expected_text.splitlines():
            if line.startswith(f"# {label}"):
                return float(line.split("=")[-1].strip())
        raise AssertionError(label)

    assert fit.number_iterations == 1
    assert fit.optimization_success == (True,)
    assert fit.theta == pytest.approx(header("theta=4*N_E*mu"), rel=1e-12)
    assert fit.rho == pytest.approx(header("rho=4*N_E*r"), rel=1e-6)
    assert fit.final_log_likelihood == pytest.approx(
        header("final log likelihood"),
        rel=1e-8,
        abs=1e-8,
    )
    np.testing.assert_allclose(
        0.5 * fit.boundaries[:-1] * fit.theta,
        expected[:, 0],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        4.0 * fit.lambda_values / fit.theta,
        expected[:, 2],
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.slow
def test_native_decode_matches_frozen_upstream_outputs() -> None:
    decode = decode_psmcplus_native(
        CONSTPOP,
        number_states=4,
        bin_size=100,
        lambda_initial=[1, 1, 1, 1],
        downsample=1000,
    )
    oracle_path = ROOT / "tests/data/psmcplus/decode_oracle_v1.npz"
    with np.load(oracle_path) as oracle:
        assert decode.theta == pytest.approx(float(oracle["theta"]), rel=1e-14)
        assert decode.rho == pytest.approx(float(oracle["rho"]), rel=1e-14)
        assert decode.log_likelihood == pytest.approx(
            float(oracle["log_likelihood"]),
            rel=2e-14,
            abs=1e-12,
        )
        np.testing.assert_array_equal(decode.positions, oracle["positions"])
        np.testing.assert_allclose(
            decode.posterior,
            oracle["posterior"],
            rtol=2e-13,
            atol=2e-14,
        )
        np.testing.assert_array_equal(
            decode.marginal_positions,
            oracle["marginal_positions"],
        )
        np.testing.assert_allclose(
            decode.marginal_recombination,
            oracle["marginal_recombination"],
            rtol=2e-13,
            atol=2e-14,
        )


@pytest.mark.slow
def test_public_native_fit_writes_normalized_original_compatible_artifacts(
    tmp_path: Path,
) -> None:
    data = io.read_multihetsep(CONSTPOP)
    prefix = tmp_path / "native/result_"
    psmcplus(
        data,
        options=PSMCPlusOptions(
            number_time_windows=4,
            bin_size=100,
            iterations=1,
            likelihood_threshold=0,
            save_iteration_files=True,
            cores=1,
        ),
        output_prefix=prefix,
        implementation="native",
    )
    result = data.results["psmcplus"]
    assert result["implementation"] == "native"
    assert result["backend"] == "native"
    assert result["n_iterations"] == 1
    assert result["log_likelihood"] == pytest.approx(-3556.760710596285, abs=1e-8)
    assert result["likelihood_trace"].shape == (1,)
    assert result["optimization_success"].tolist() == [True]
    assert Path(f"{prefix}final_parameters.txt").is_file()
    assert Path(f"{prefix}params_iteration1.txt").is_file()
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "final_parameters",
        "iteration_parameters",
    }
    assert all(artifact["persisted"] for artifact in result["artifacts"])
    assert result["provenance"]["upstream"] is None


@pytest.mark.slow
def test_public_native_decode_normalizes_and_persists_outputs(tmp_path: Path) -> None:
    data = io.read_multihetsep(CONSTPOP)
    posterior_path = tmp_path / "native/posterior.txt"
    marginal_path = tmp_path / "native/marginal.txt"
    psmcplus(
        data,
        options=PSMCPlusOptions(
            mode="decode",
            number_time_windows=4,
            bin_size=100,
            lambda_initial=[1, 1, 1, 1],
            decode_downsample=1000,
            cores=1,
        ),
        output_prefix=posterior_path,
        marginal_recombination_path=marginal_path,
        implementation="native",
    )
    result = data.results["psmcplus"]
    assert result["implementation"] == "native"
    assert result["posterior"].shape == (9, 4)
    assert result["marginal_recombination"]["position"].shape == (9992,)
    np.testing.assert_allclose(result["posterior"].sum(axis=1), 1.0)
    assert posterior_path.is_file()
    assert marginal_path.is_file()
