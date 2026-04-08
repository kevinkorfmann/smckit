"""Integration coverage for the upstream-backed eSMC2 wrapper."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.backends._numba_esmc2 import esmc2_build_hmm
from smckit.io import read_multihetsep, read_psmcfa
from smckit.tl._esmc2 import _expectation_step, _sequences_from_smcdata, esmc2


def _upstream_esmc2_ready() -> bool:
    return shutil.which("Rscript") is not None and Path(".r-lib/eSMC2").exists()


pytestmark = pytest.mark.skipif(
    not _upstream_esmc2_ready(),
    reason="Local upstream eSMC2 R environment is not available",
)


@pytest.fixture()
def upstream_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMCKIT_ESMC2_RSCRIPT", shutil.which("Rscript") or "")


@pytest.fixture()
def clean_pairwise_data() -> SmcData:
    rng = np.random.default_rng(7)
    seq = rng.choice([0, 1], size=800, p=[0.99, 0.01]).astype(np.int8)
    return SmcData(
        uns={
            "records": [{"codes": seq}],
            "sum_L": int((seq < 2).sum()),
            "sum_n": int((seq == 1).sum()),
        }
    )


def _dense_sequence(
    length: int,
    *,
    het_positions: list[int],
    missing_ranges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    seq = np.zeros(length, dtype=np.int8)
    seq[np.asarray(het_positions, dtype=np.int64)] = 1
    for start, end in missing_ranges or []:
        seq[start:end] = 2
    return seq


def _write_psmcfa(path: Path, records: list[np.ndarray]) -> Path:
    mapping = {0: "A", 1: "K", 2: "N"}
    with path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            chars = "".join(mapping[int(code)] for code in np.asarray(record, dtype=np.int8))
            handle.write(f">seq{idx}\n")
            handle.write(f"{chars}\n")
    return path


def _pair_alleles(code: int) -> str:
    if code == 0:
        return "AA"
    if code == 1:
        return "AT"
    if code == 2:
        return "?A"
    raise ValueError(f"Unexpected pairwise code: {code}")


def _pair4_alleles(code_a: int, code_b: int) -> str:
    pair_a = {
        0: ("A", "A"),
        1: ("A", "T"),
        2: ("?", "A"),
    }[int(code_a)]
    pair_b = {
        0: ("C", "C"),
        1: ("C", "G"),
        2: ("?", "C"),
    }[int(code_b)]
    return "".join(pair_a + pair_b)


def _write_multihetsep_pair(
    path: Path,
    seq: np.ndarray,
    *,
    chrom: str = "chr1",
    ambiguous_positions: set[int] | None = None,
) -> Path:
    ambiguous_positions = ambiguous_positions or set()
    with path.open("w", encoding="utf-8") as handle:
        for pos, code in enumerate(np.asarray(seq, dtype=np.int8), start=1):
            allele = "AA,AT" if (pos - 1) in ambiguous_positions else _pair_alleles(int(code))
            handle.write(f"{chrom}\t{pos}\t1\t{allele}\n")
    return path


def _write_multihetsep_multi_pair(
    path: Path,
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    *,
    chrom: str = "chr1",
) -> Path:
    if len(seq_a) != len(seq_b):
        raise ValueError("Multi-pair sequences must share the same length.")
    with path.open("w", encoding="utf-8") as handle:
        for pos, (code_a, code_b) in enumerate(
            zip(np.asarray(seq_a, dtype=np.int8), np.asarray(seq_b, dtype=np.int8)),
            start=1,
        ):
            allele = _pair4_alleles(int(code_a), int(code_b))
            handle.write(f"{chrom}\t{pos}\t1\t{allele}\n")
    return path


def _build_public_family_data(name: str, tmp_path: Path) -> SmcData:
    if name == "psmcfa_clean":
        record = _dense_sequence(
            420,
            het_positions=[19, 44, 78, 120, 166, 210, 255, 310, 360, 401],
        )
        return read_psmcfa(_write_psmcfa(tmp_path / "clean.psmcfa", [record]))
    if name == "psmcfa_missing":
        record = _dense_sequence(
            420,
            het_positions=[19, 44, 78, 120, 166, 210, 255, 310, 360, 401],
            missing_ranges=[(0, 12), (90, 103), (250, 266)],
        )
        return read_psmcfa(_write_psmcfa(tmp_path / "missing.psmcfa", [record]))
    if name == "psmcfa_multi_record":
        record_a = _dense_sequence(
            420,
            het_positions=[19, 44, 78, 120, 166, 210, 255, 310, 360, 401],
            missing_ranges=[(30, 40), (280, 288)],
        )
        record_b = _dense_sequence(
            510,
            het_positions=[25, 60, 101, 144, 188, 230, 272, 315, 360, 405, 451, 489],
            missing_ranges=[(0, 8), (200, 215), (420, 430)],
        )
        return read_psmcfa(_write_psmcfa(tmp_path / "multi.psmcfa", [record_a, record_b]))
    if name == "multihetsep_single_pair":
        seq = _dense_sequence(
            430,
            het_positions=[17, 51, 83, 126, 171, 214, 259, 301, 347, 390],
            missing_ranges=[(100, 111), (280, 292)],
        )
        return read_multihetsep(
            _write_multihetsep_pair(tmp_path / "single.multihetsep", seq),
            pair_indices=[(0, 1)],
        )
    if name == "multihetsep_multi_pair":
        seq_a = _dense_sequence(
            430,
            het_positions=[17, 51, 83, 126, 171, 214, 259, 301, 347, 390],
            missing_ranges=[(100, 111), (280, 292)],
        )
        seq_b = _dense_sequence(
            430,
            het_positions=[11, 48, 90, 132, 176, 218, 262, 306, 351, 399],
            missing_ranges=[(40, 47), (320, 333)],
        )
        return read_multihetsep(
            _write_multihetsep_multi_pair(tmp_path / "pairs.multihetsep", seq_a, seq_b),
            pair_indices=[(0, 1), (2, 3)],
        )
    if name == "multihetsep_multi_file":
        seq_a = _dense_sequence(
            260,
            het_positions=[15, 44, 73, 102, 141, 180, 219, 244],
            missing_ranges=[(0, 6), (120, 128)],
        )
        seq_b = _dense_sequence(
            310,
            het_positions=[18, 57, 96, 135, 174, 213, 252, 291],
            missing_ranges=[(60, 69), (220, 231)],
        )
        chr_a = _write_multihetsep_pair(tmp_path / "chrA.multihetsep", seq_a, chrom="chrA")
        chr_b = _write_multihetsep_pair(tmp_path / "chrB.multihetsep", seq_b, chrom="chrB")
        return read_multihetsep([chr_a, chr_b], pair_indices=[(0, 1)])
    if name == "multihetsep_skip_ambiguous":
        seq = _dense_sequence(
            410,
            het_positions=[14, 52, 91, 129, 168, 206, 245, 283, 322, 360, 398],
            missing_ranges=[(200, 210)],
        )
        return read_multihetsep(
            _write_multihetsep_pair(
                tmp_path / "ambiguous.multihetsep",
                seq,
                ambiguous_positions={33, 155, 275},
            ),
            pair_indices=[(0, 1)],
            skip_ambiguous=True,
        )
    raise ValueError(f"Unknown public family fixture: {name}")


def _max_rel_error(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return float(
        np.max(np.abs(left_arr - right_arr) / np.maximum(np.abs(right_arr), 1e-12))
    )


def _assert_fit_parity(
    native: dict,
    upstream: dict,
    *,
    tc_rel_tol: float,
    xi_rel_tol: float,
    beta_abs_tol: float,
    sigma_abs_tol: float,
    final_ll_abs_tol: float,
) -> None:
    tc_rel = _max_rel_error(np.asarray(native["Tc"]), np.asarray(upstream["Tc"]))
    xi_rel = _max_rel_error(np.asarray(native["Xi"]), np.asarray(upstream["Xi"]))
    final_ll_delta = abs(
        float(native["log_likelihood"])
        - float(upstream["upstream"]["final_sufficient_statistics"]["log_likelihood"])
    )

    assert tc_rel < tc_rel_tol
    assert xi_rel < xi_rel_tol
    assert abs(float(native["beta"]) - float(upstream["beta"])) < beta_abs_tol
    assert abs(float(native["sigma"]) - float(upstream["sigma"])) < sigma_abs_tol
    assert final_ll_delta < final_ll_abs_tol
    assert native["rho"] == pytest.approx(upstream["rho"], rel=1e-7, abs=1e-10)
    assert native["rho_per_sequence"] == pytest.approx(
        upstream["rho_per_sequence"],
        rel=1e-7,
        abs=1e-10,
    )
    assert native["mu"] == pytest.approx(upstream["mu"], rel=1e-7, abs=1e-10)
    assert native["theta"] == pytest.approx(upstream["theta"], rel=1e-7, abs=1e-10)


def test_esmc2_upstream_backend_runs_end_to_end(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    result = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        backend="upstream",
    )

    res = result.results["esmc2"]
    assert res["backend"] == "upstream"
    assert res["Xi"].shape == (6,)
    assert np.all(res["Xi"] > 0)
    assert np.all(np.isfinite(res["ne"]))
    assert np.isfinite(res["log_likelihood"])
    assert set(res["upstream"]) >= {"Tc_returned", "rscript", "sequence_length", "stdout_log"}
    assert res["rho"] == pytest.approx(res["rho_per_sequence"] / (2 * 800), rel=1e-12)


def test_esmc2_upstream_can_capture_sufficient_statistics(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    result = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
        upstream_options={"capture_sufficient_statistics": True},
    )

    stats = result.results["esmc2"]["upstream"]["sufficient_statistics"]
    assert np.asarray(stats["N"]).shape == (6, 6)
    assert np.asarray(stats["M"]).shape == (3, 6)
    assert np.asarray(stats["q"]).shape == (6,)


def test_esmc2_native_hmm_builder_matches_upstream(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    result = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
        upstream_options={"capture_hmm_builder": True},
    ).results["esmc2"]

    oracle = result["upstream"]["hmm_builder"]
    sequences = _sequences_from_smcdata(copy.deepcopy(clean_pairwise_data))
    Q, q, t, Tc, e = esmc2_build_hmm(
        6,
        np.asarray(result["Xi"], dtype=np.float64),
        float(result["beta"]),
        float(result["sigma"]),
        float(result["rho_per_sequence"]),
        float(result["mu"]),
        1.0,
        len(sequences[0]),
    )

    np.testing.assert_allclose(Q, np.asarray(oracle["Q"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(q, np.asarray(oracle["q"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(t, np.asarray(oracle["t"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Tc, np.asarray(oracle["Tc"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(e, np.asarray(oracle["g"]), rtol=1e-10, atol=1e-12)


def test_esmc2_native_final_sufficient_statistics_match_upstream(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    result = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
    ).results["esmc2"]

    oracle = result["upstream"]["final_sufficient_statistics"]
    sequences = _sequences_from_smcdata(copy.deepcopy(clean_pairwise_data))
    native_stats, native_ll, _, _ = _expectation_step(
        sequences=sequences,
        n=6,
        Xi=np.asarray(result["Xi"], dtype=np.float64),
        beta=float(result["beta"]),
        sigma=float(result["sigma"]),
        rho=float(result["rho_per_sequence"]),
        mu=float(result["mu"]),
        mu_b=1.0,
        L=len(sequences[0]),
    )

    np.testing.assert_allclose(
        np.asarray(native_stats["N"]),
        np.asarray(oracle["N"]),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(native_stats["M"]),
        np.asarray(oracle["M"]),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(native_stats["q_post"]),
        np.asarray(oracle["q"]),
        rtol=1e-10,
        atol=1e-12,
    )
    assert native_ll == pytest.approx(float(oracle["log_likelihood"]), rel=1e-12, abs=1e-12)
    assert native_ll == pytest.approx(float(result["log_likelihood"]), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "family_name",
    [
        "psmcfa_clean",
        "psmcfa_missing",
        "psmcfa_multi_record",
        "multihetsep_single_pair",
        "multihetsep_multi_pair",
        "multihetsep_multi_file",
        "multihetsep_skip_ambiguous",
    ],
)
def test_esmc2_native_matches_upstream_across_public_input_families(
    tmp_path: Path,
    family_name: str,
    upstream_env: None,
) -> None:
    data = _build_public_family_data(family_name, tmp_path)
    expected_sequences = _sequences_from_smcdata(copy.deepcopy(data))
    kwargs = {
        "n_states": 6,
        "n_iterations": 1,
        "estimate_rho": False,
        "mu": 1e-8,
        "generation_time": 1.0,
    }
    native = esmc2(
        copy.deepcopy(data),
        implementation="native",
        **kwargs,
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **kwargs,
    ).results["esmc2"]

    _assert_fit_parity(
        native,
        upstream,
        tc_rel_tol=5e-3,
        xi_rel_tol=2e-2,
        beta_abs_tol=1e-12,
        sigma_abs_tol=1e-12,
        final_ll_abs_tol=5e-3,
    )

    provenance = upstream["upstream"]
    assert provenance["input_family"] == (
        "psmcfa" if family_name.startswith("psmcfa") else "multihetsep"
    )
    assert provenance["n_sequences"] == len(expected_sequences)
    assert provenance["sequence_lengths"] == [len(seq) for seq in expected_sequences]
    assert provenance["used_sequence_indices"] == list(range(len(expected_sequences)))


def test_esmc2_native_matches_upstream_on_grouped_beta_fit_for_multi_record_psmcfa(
    tmp_path: Path,
    upstream_env: None,
) -> None:
    data = _build_public_family_data("psmcfa_multi_record", tmp_path)
    kwargs = {
        "n_states": 6,
        "n_iterations": 1,
        "estimate_beta": True,
        "estimate_sigma": False,
        "estimate_rho": False,
        "beta": 0.3,
        "sigma": 0.0,
        "pop_vect": [3, 3],
        "mu": 1e-8,
        "generation_time": 1.0,
    }
    native = esmc2(
        copy.deepcopy(data),
        implementation="native",
        **kwargs,
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **kwargs,
    ).results["esmc2"]

    _assert_fit_parity(
        native,
        upstream,
        tc_rel_tol=5e-3,
        xi_rel_tol=2e-2,
        beta_abs_tol=5e-4,
        sigma_abs_tol=1e-12,
        final_ll_abs_tol=5e-3,
    )


@pytest.mark.parametrize(
    ("estimate_beta", "estimate_sigma", "beta", "sigma"),
    [
        (True, False, 0.3, 0.0),
        (False, True, 0.5, 0.2),
    ],
)
def test_esmc2_upstream_backend_normalizes_final_hmm_state_for_beta_sigma_runs(
    clean_pairwise_data: SmcData,
    upstream_env: None,
    estimate_beta: bool,
    estimate_sigma: bool,
    beta: float,
    sigma: float,
) -> None:
    result = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_beta=estimate_beta,
        estimate_sigma=estimate_sigma,
        estimate_rho=False,
        beta=beta,
        sigma=sigma,
        mu=1e-8,
        generation_time=1.0,
        implementation="upstream",
        upstream_options={"capture_hmm_builder": True},
    ).results["esmc2"]

    builder = result["upstream"]["hmm_builder"]
    raw_tc = np.asarray(result["upstream"]["Tc_returned"], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(result["Tc"]),
        np.asarray(builder["Tc"]),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(result["t"]),
        np.asarray(builder["t"]),
        rtol=1e-10,
        atol=1e-12,
    )
    assert raw_tc.shape == (6,)
    assert not np.allclose(np.asarray(result["Tc"]), raw_tc, rtol=1e-10, atol=1e-12)


def test_esmc2_native_matches_upstream_on_single_iteration_fit(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    native = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        backend="native",
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=1,
        estimate_rho=False,
        mu=1e-8,
        generation_time=1.0,
        backend="upstream",
    ).results["esmc2"]

    tc_rel = np.abs(np.asarray(native["Tc"]) - np.asarray(upstream["Tc"])) / np.maximum(
        np.abs(np.asarray(upstream["Tc"])),
        1e-12,
    )
    xi_rel = np.abs(np.asarray(native["Xi"]) - np.asarray(upstream["Xi"])) / np.maximum(
        np.abs(np.asarray(upstream["Xi"])),
        1e-12,
    )
    xi_corr = float(np.corrcoef(np.asarray(native["Xi"]), np.asarray(upstream["Xi"]))[0, 1])
    loglik_delta = float(abs(float(native["log_likelihood"]) - float(upstream["log_likelihood"])))

    assert native["rho"] == pytest.approx(native["rho_per_sequence"] / (2 * 800), rel=1e-12)
    assert upstream["rho"] == pytest.approx(upstream["rho_per_sequence"] / (2 * 800), rel=1e-12)

    assert float(np.max(tc_rel)) < 1e-12
    assert float(np.max(xi_rel)) < 1e-4
    assert xi_corr > 0.999999
    assert loglik_delta < 1e-4
    assert native["rho"] == pytest.approx(upstream["rho"], rel=1e-12)
    assert native["rho_per_sequence"] == pytest.approx(upstream["rho_per_sequence"], rel=1e-12)
    assert native["beta"] == pytest.approx(upstream["beta"], rel=1e-12)
    assert native["sigma"] == pytest.approx(upstream["sigma"], rel=1e-12)
    assert native["mu"] == pytest.approx(upstream["mu"], rel=1e-12)
    assert native["theta"] == pytest.approx(upstream["theta"], rel=1e-12)


def test_esmc2_native_matches_upstream_when_rho_redo_extends_iterations(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    native = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=2,
        estimate_rho=True,
        mu=1e-8,
        generation_time=1.0,
        backend="native",
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(clean_pairwise_data),
        n_states=6,
        n_iterations=2,
        estimate_rho=True,
        mu=1e-8,
        generation_time=1.0,
        backend="upstream",
    ).results["esmc2"]

    tc_rel = np.abs(np.asarray(native["Tc"]) - np.asarray(upstream["Tc"])) / np.maximum(
        np.abs(np.asarray(upstream["Tc"])),
        1e-12,
    )
    xi_rel = np.abs(np.asarray(native["Xi"]) - np.asarray(upstream["Xi"])) / np.maximum(
        np.abs(np.asarray(upstream["Xi"])),
        1e-12,
    )
    xi_corr = float(np.corrcoef(np.asarray(native["Xi"]), np.asarray(upstream["Xi"]))[0, 1])
    loglik_delta = float(abs(float(native["log_likelihood"]) - float(upstream["log_likelihood"])))

    assert len(native["rounds"]) > 2
    assert float(np.max(tc_rel)) < 1e-12
    assert float(np.max(xi_rel)) < 1e-4
    assert xi_corr > 0.999999
    assert loglik_delta < 1e-4
    assert native["rho"] == pytest.approx(upstream["rho"], rel=2e-5)
    assert native["rho_per_sequence"] == pytest.approx(upstream["rho_per_sequence"], rel=2e-5)
    assert native["mu"] == pytest.approx(upstream["mu"], rel=1e-12)
    assert native["theta"] == pytest.approx(upstream["theta"], rel=1e-12)


@pytest.mark.parametrize(
    ("estimate_beta", "estimate_sigma", "beta", "sigma"),
    [
        (True, False, 0.3, 0.0),
        (False, True, 0.5, 0.2),
        (True, True, 0.3, 0.2),
    ],
)
def test_esmc2_native_matches_upstream_on_beta_sigma_fit_branches(
    clean_pairwise_data: SmcData,
    upstream_env: None,
    estimate_beta: bool,
    estimate_sigma: bool,
    beta: float,
    sigma: float,
) -> None:
    kwargs = {
        "n_states": 6,
        "n_iterations": 1,
        "estimate_beta": estimate_beta,
        "estimate_sigma": estimate_sigma,
        "estimate_rho": False,
        "beta": beta,
        "sigma": sigma,
        "mu": 1e-8,
        "generation_time": 1.0,
    }
    native = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="native",
        **kwargs,
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **kwargs,
    ).results["esmc2"]

    _assert_fit_parity(
        native,
        upstream,
        tc_rel_tol=3e-3,
        xi_rel_tol=4e-3,
        beta_abs_tol=1e-3,
        sigma_abs_tol=5e-4,
        final_ll_abs_tol=1.5e-3,
    )


def test_esmc2_native_matches_upstream_on_grouped_beta_fit(
    clean_pairwise_data: SmcData,
    upstream_env: None,
) -> None:
    kwargs = {
        "n_states": 6,
        "n_iterations": 1,
        "estimate_beta": True,
        "estimate_sigma": False,
        "estimate_rho": False,
        "beta": 0.3,
        "sigma": 0.0,
        "pop_vect": [3, 3],
        "mu": 1e-8,
        "generation_time": 1.0,
    }
    native = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="native",
        **kwargs,
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **kwargs,
    ).results["esmc2"]

    _assert_fit_parity(
        native,
        upstream,
        tc_rel_tol=1e-4,
        xi_rel_tol=1e-4,
        beta_abs_tol=1e-4,
        sigma_abs_tol=1e-12,
        final_ll_abs_tol=1e-5,
    )


@pytest.mark.parametrize(
    ("estimate_beta", "estimate_sigma", "beta", "sigma"),
    [
        (True, False, 0.3, 0.0),
        (False, True, 1.0, 0.2),
        (True, True, 0.3, 0.2),
    ],
)
def test_esmc2_native_matches_upstream_on_all_singleton_grouped_fit(
    clean_pairwise_data: SmcData,
    upstream_env: None,
    estimate_beta: bool,
    estimate_sigma: bool,
    beta: float,
    sigma: float,
) -> None:
    kwargs = {
        "n_states": 6,
        "n_iterations": 1,
        "estimate_beta": estimate_beta,
        "estimate_sigma": estimate_sigma,
        "estimate_rho": False,
        "beta": beta,
        "sigma": sigma,
        "pop_vect": [1, 1, 1, 1, 1, 1],
        "mu": 1e-8,
        "generation_time": 1.0,
    }
    native = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="native",
        **kwargs,
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(clean_pairwise_data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **kwargs,
    ).results["esmc2"]

    _assert_fit_parity(
        native,
        upstream,
        tc_rel_tol=4e-3,
        xi_rel_tol=2e-2,
        beta_abs_tol=2e-3,
        sigma_abs_tol=5e-4,
        final_ll_abs_tol=3e-3,
    )
