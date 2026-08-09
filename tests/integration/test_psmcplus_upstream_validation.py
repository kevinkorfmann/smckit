"""Live validation of the immutable PSMC+ upstream preservation path."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import smckit

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "vendor/PSMCplus/simulations/constpopsize.mhs"
EXPECTED = REPO_ROOT / "tests/data/psmcplus/constpop_D4_1iter.final_parameters.txt"

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.slow,
    pytest.mark.skipif(
        not smckit.upstream.status("psmcplus")["ready"],
        reason="Pinned PSMC+ source or its Python dependency stack is unavailable",
    ),
]


def _header_scalar(text: str, label: str) -> float:
    match = re.search(rf"^# {re.escape(label)} =? ?([^\s]+)$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Missing PSMC+ output header: {label}")
    return float(match.group(1))


def test_pinned_psmcplus_constant_population_oracle(tmp_path: Path) -> None:
    """The unmodified pinned source reproduces its frozen numeric artifact."""
    output_dir = tmp_path / "upstream-run"
    result = smckit.upstream.run(
        "psmcplus",
        [
            "-in",
            str(INPUT),
            "-D",
            "4",
            "-b",
            "100",
            "-its",
            "1",
            "-thresh",
            "0",
            "-c",
            "1",
            "-o",
            "oracle_",
        ],
        output_dir=output_dir,
        timeout=180,
        env={
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert any("numpy.math" in patch for patch in result.compatibility_patches)
    artifact = output_dir / "oracle_final_parameters.txt"
    assert artifact.is_file()
    assert {item["path"] for item in result.artifacts} == {"oracle_final_parameters.txt"}

    actual_text = artifact.read_text(encoding="utf-8")
    expected_text = EXPECTED.read_text(encoding="utf-8")
    for label in [
        "final log likelihood",
        "final change in log likelihood",
        "theta=4*N_E*mu",
        "rho=4*N_E*r",
    ]:
        assert _header_scalar(actual_text, label) == pytest.approx(
            _header_scalar(expected_text, label),
            rel=1e-8,
            abs=1e-12,
        )
    np.testing.assert_allclose(
        np.loadtxt(artifact),
        np.loadtxt(EXPECTED),
        rtol=1e-8,
        atol=1e-12,
    )


def test_typed_psmcplus_fit_matches_frozen_upstream_oracle(tmp_path: Path) -> None:
    """The typed adapter preserves inference while normalizing its result."""
    data = smckit.io.read_multihetsep(INPUT)
    output_prefix = tmp_path / "typed/oracle_"

    returned = smckit.tl.psmcplus(
        data,
        options=smckit.tl.PSMCPlusOptions(
            number_time_windows=4,
            bin_size=100,
            iterations=1,
            likelihood_threshold=0,
            cores=1,
        ),
        mutation_rate=1e-8,
        generation_time=25,
        output_prefix=output_prefix,
        implementation="upstream",
        timeout=180,
    )

    result = returned.results["psmcplus"]
    expected = np.loadtxt(EXPECTED)
    assert returned is data
    assert result["implementation"] == "upstream"
    assert result["implementation_requested"] == "upstream"
    assert result["mode"] == "fit"
    assert result["log_likelihood"] == pytest.approx(
        _header_scalar(EXPECTED.read_text(encoding="utf-8"), "final log likelihood"),
        rel=1e-8,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        result["scaled_left_time_boundary"],
        expected[:, 0],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["scaled_inverse_population_size"],
        expected[:, 2],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(result["time"], expected[:, 0] / 1e-8 * 25)
    np.testing.assert_allclose(result["ne"], 1.0 / expected[:, 2] / 1e-8)
    assert data.log_likelihood("psmcplus") == pytest.approx(result["log_likelihood"])
    assert data.effective_population_size("psmcplus").shape == (4,)

    persisted = Path(f"{output_prefix}final_parameters.txt")
    assert persisted.is_file()
    assert result["artifacts"][0]["path"] == str(persisted)
    assert result["artifacts"][0]["persisted"] is True
    assert str(INPUT) in result["provenance"]["input_sha256"]
    assert result["provenance"]["runtime_seconds"] > 0


def test_typed_psmcplus_decode_normalizes_live_upstream_output(tmp_path: Path) -> None:
    """Posterior and marginal-recombination output survive typed execution."""
    data = smckit.io.read_multihetsep(INPUT)
    posterior_path = tmp_path / "typed/posterior.txt"
    recombination_path = tmp_path / "typed/marginal_recombination.txt"

    smckit.tl.psmcplus(
        data,
        options=smckit.tl.PSMCPlusOptions(
            mode="decode",
            number_time_windows=4,
            bin_size=100,
            lambda_initial=[1.0, 1.0, 1.0, 1.0],
            decode_downsample=1000,
            cores=1,
        ),
        mutation_rate=1e-8,
        generation_time=25,
        output_prefix=posterior_path,
        marginal_recombination_path=recombination_path,
        implementation="upstream",
        timeout=180,
    )

    result = data.results["psmcplus"]
    assert result["mode"] == "decode"
    assert result["posterior"].shape == (9, 4)
    np.testing.assert_allclose(result["posterior"].sum(axis=1), 1.0)
    assert result["time_boundaries"].shape == (5,)
    assert result["posterior_mean_time"].shape == result["position"].shape
    marginal = result["marginal_recombination"]
    assert marginal["position"].shape == (9992,)
    np.testing.assert_allclose(
        marginal["recombination_probability"] + marginal["no_recombination_probability"],
        1.0,
    )
    assert posterior_path.is_file()
    assert recombination_path.is_file()
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "posterior_decoding",
        "marginal_recombination",
    }
