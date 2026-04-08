"""MSMC-IM validation against the vendored upstream CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.io import read_msmc_combined_output, read_msmc_im_output
from smckit.tl import msmc_im
from smckit.tl._implementation import NativeTrustWarning
from smckit.tl._msmc_im import _read_msmc_im_fittingdetails, _threshold_migration_rates

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
UPSTREAM = ROOT / "vendor" / "MSMC-IM" / "MSMC_IM.py"

ORACLE_CASES = [
    pytest.param(
        "default",
        "1*2+25*1+1*2+1*3",
        (1e-8, 1e-6),
        1.25e-8,
        15000.0,
        15000.0,
        1e-4,
        id="default",
    ),
    pytest.param(
        "lighter_penalty",
        "1*2+25*1+1*2+1*3",
        (1e-9, 1e-7),
        1.25e-8,
        18000.0,
        12000.0,
        5e-5,
        id="lighter_penalty",
    ),
    pytest.param(
        "alt_pattern",
        "2*1+1*2+25*1+1*3",
        (1e-8, 1e-6),
        1.25e-8,
        16000.0,
        14000.0,
        8e-5,
        id="alt_pattern",
    ),
    pytest.param(
        "asymmetric_stronger_penalty_low_m",
        "1*2+25*1+1*2+1*3",
        (2e-8, 2e-6),
        1.25e-8,
        22000.0,
        9000.0,
        2e-6,
        id="asymmetric_stronger_penalty_low_m",
    ),
]

ARRAY_FIELDS = (
    "left_boundary",
    "right_boundary",
    "N1",
    "N2",
    "N1_raw",
    "N2_raw",
    "m",
    "m_thresholded",
    "M",
)


def _run_vendored_cli(
    tmp_path: Path,
    *,
    name: str,
    pattern: str,
    beta: tuple[float, float],
    mu: float,
    N1_init: float,
    N2_init: float,
    m_init: float,
) -> dict[str, object]:
    out_prefix = tmp_path / name
    mplconfigdir = tmp_path / "mplconfig"
    mplconfigdir.mkdir()

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mplconfigdir)

    subprocess.run(
        [
            sys.executable,
            str(UPSTREAM),
            "-mu",
            str(mu),
            "-N1",
            str(N1_init),
            "-N2",
            str(N2_init),
            "-m",
            str(m_init),
            "-p",
            pattern,
            "-beta",
            f"{beta[0]},{beta[1]}",
            "-o",
            str(out_prefix),
            "--printfittingdetails",
            "--xlog",
            str(INPUT),
        ],
        cwd=UPSTREAM.parent,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    estimates_matches = sorted(tmp_path.glob(f"{name}.b1_*.b2_*.MSMC_IM.estimates.txt"))
    fittingdetails_matches = sorted(tmp_path.glob(f"{name}.b1_*.b2_*.MSMC_IM.fittingdetails.txt"))
    assert len(estimates_matches) == 1
    assert len(fittingdetails_matches) == 1

    estimates = read_msmc_im_output(estimates_matches[0])
    fittingdetails = _read_msmc_im_fittingdetails(fittingdetails_matches[0])

    np.testing.assert_allclose(
        fittingdetails["left_boundary"],
        estimates["left_boundary"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(fittingdetails["N1"], estimates["N1"], rtol=5e-10, atol=1e-12)
    np.testing.assert_allclose(fittingdetails["N2"], estimates["N2"], rtol=5e-10, atol=1e-12)

    return {
        "left_boundary": estimates["left_boundary"],
        "right_boundary": _expected_right_boundary(mu=mu),
        "N1": estimates["N1"],
        "N2": estimates["N2"],
        "N1_raw": fittingdetails["N1_raw"],
        "N2_raw": fittingdetails["N2_raw"],
        "m": estimates["m"],
        "m_thresholded": _threshold_migration_rates(estimates["m"], estimates["M"]),
        "M": estimates["M"],
        "init_chi_square": float(fittingdetails["init_chi_square"]),
        "final_chi_square": float(fittingdetails["final_chi_square"]),
        "split_time_quantiles": _expected_split_time_quantiles(
            estimates["left_boundary"],
            estimates["M"],
        ),
        "pattern": pattern,
        "beta": beta,
    }


def _expected_right_boundary(mu: float = 1.25e-8) -> np.ndarray:
    msmc = read_msmc_combined_output(INPUT, mu=mu)
    left = np.asarray(msmc["left_boundary"], dtype=np.float64)
    right = np.asarray(msmc["right_boundary"], dtype=np.float64).copy()
    right[-1] = left[-1] * 4.0
    return right


def _expected_split_time_quantiles(
    left_boundary: np.ndarray,
    cumulative_migration: np.ndarray,
) -> dict[float, float]:
    quantiles: dict[float, float] = {}
    max_migration = float(np.max(cumulative_migration))

    for q in [0.25, 0.5, 0.75]:
        if max_migration < q:
            continue
        if cumulative_migration[0] >= q:
            quantiles[q] = q / cumulative_migration[0] * left_boundary[0]
            continue

        idx = int(np.searchsorted(cumulative_migration, q, side="left"))
        frac = (
            (q - cumulative_migration[idx - 1])
            / (cumulative_migration[idx] - cumulative_migration[idx - 1])
        )
        quantiles[q] = left_boundary[idx - 1] + frac * (
            left_boundary[idx] - left_boundary[idx - 1]
        )

    return quantiles


def _assert_split_quantiles_match(
    result: dict[str, object],
    expected: dict[float, float],
) -> None:
    assert result["split_time_quantiles"].keys() == expected.keys()
    for q, t in expected.items():
        assert result["split_time_quantiles"][q] == pytest.approx(t)


def _assert_matches_oracle(result: dict[str, object], ref: dict[str, object]) -> None:
    for field in ARRAY_FIELDS:
        atol = 0.0 if field in {"left_boundary", "right_boundary"} else 1e-12
        rtol = 0.0 if field in {"left_boundary", "right_boundary"} else 5e-10
        np.testing.assert_allclose(result[field], ref[field], rtol=rtol, atol=atol)

    assert result["init_chi_square"] == pytest.approx(ref["init_chi_square"])
    assert result["final_chi_square"] == pytest.approx(ref["final_chi_square"])
    assert result["pattern"] == ref["pattern"]
    assert result["beta"] == ref["beta"]
    _assert_split_quantiles_match(result, ref["split_time_quantiles"])


def _assert_public_payloads_match(
    native: dict[str, object],
    upstream: dict[str, object],
) -> None:
    assert set(native.keys()) == {
        *ARRAY_FIELDS,
        "init_chi_square",
        "final_chi_square",
        "split_time_quantiles",
        "pattern",
        "beta",
        "implementation_requested",
        "implementation",
    }
    assert set(upstream.keys()) == set(native.keys()) | {"backend", "upstream"}

    for field in ARRAY_FIELDS:
        atol = 0.0 if field in {"left_boundary", "right_boundary"} else 1e-12
        rtol = 0.0 if field in {"left_boundary", "right_boundary"} else 5e-10
        np.testing.assert_allclose(native[field], upstream[field], rtol=rtol, atol=atol)

    assert native["init_chi_square"] == pytest.approx(upstream["init_chi_square"])
    assert native["final_chi_square"] == pytest.approx(upstream["final_chi_square"])
    assert native["split_time_quantiles"] == pytest.approx(upstream["split_time_quantiles"])
    assert native["pattern"] == upstream["pattern"]
    assert native["beta"] == upstream["beta"]


@pytest.mark.parametrize(
    ("name", "pattern", "beta", "mu", "N1_init", "N2_init", "m_init"),
    ORACLE_CASES,
)
def test_msmc_im_native_and_upstream_match_vendored_cli_on_oracle_matrix(
    tmp_path: Path,
    name: str,
    pattern: str,
    beta: tuple[float, float],
    mu: float,
    N1_init: float,
    N2_init: float,
    m_init: float,
) -> None:
    ref = _run_vendored_cli(
        tmp_path,
        name=name,
        pattern=pattern,
        beta=beta,
        mu=mu,
        N1_init=N1_init,
        N2_init=N2_init,
        m_init=m_init,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = msmc_im(
            INPUT,
            implementation="native",
            pattern=pattern,
            beta=beta,
            mu=mu,
            N1_init=N1_init,
            N2_init=N2_init,
            m_init=m_init,
        ).results["msmc_im"]

    assert not any(isinstance(item.message, NativeTrustWarning) for item in caught)
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []

    upstream = msmc_im(
        INPUT,
        implementation="upstream",
        pattern=pattern,
        beta=beta,
        mu=mu,
        N1_init=N1_init,
        N2_init=N2_init,
        m_init=m_init,
    ).results["msmc_im"]

    _assert_matches_oracle(native, ref)
    _assert_matches_oracle(upstream, ref)
    _assert_public_payloads_match(native, upstream)

    assert native["implementation_requested"] == "native"
    assert native["implementation"] == "native"
    assert "backend" not in native
    assert "upstream" not in native

    assert upstream["implementation_requested"] == "upstream"
    assert upstream["implementation"] == "upstream"
    assert upstream["backend"] == "upstream"
    assert upstream["upstream"]["effective_args"]["pattern"] == pattern
    assert upstream["upstream"]["effective_args"]["beta"] == [float(beta[0]), float(beta[1])]
    assert upstream["upstream"]["effective_args"]["mu"] == pytest.approx(mu)
    assert upstream["upstream"]["effective_args"]["printfittingdetails"] is True
    assert upstream["upstream"]["effective_args"]["xlog"] is True
    assert Path(upstream["upstream"]["estimates_path"]).name.endswith(".MSMC_IM.estimates.txt")
    assert Path(upstream["upstream"]["fittingdetails_path"]).name.endswith(
        ".MSMC_IM.fittingdetails.txt"
    )


def test_msmc_im_public_upstream_runner_matches_vendored_cli_for_in_memory_input(
    tmp_path: Path,
) -> None:
    ref = _run_vendored_cli(
        tmp_path,
        name="in_memory",
        pattern="1*2+25*1+1*2+1*3",
        beta=(1e-8, 1e-6),
        mu=1.25e-8,
        N1_init=15000.0,
        N2_init=15000.0,
        m_init=1e-4,
    )
    msmc_combined = dict(read_msmc_combined_output(INPUT, mu=1.25e-8))
    msmc_combined.pop("source_path", None)
    data = SmcData()
    data.uns["msmc_combined"] = msmc_combined

    upstream = msmc_im(data, implementation="upstream").results["msmc_im"]

    _assert_matches_oracle(upstream, ref)
    assert upstream["implementation"] == "upstream"
    assert Path(upstream["upstream"]["input_path"]).name == "combined.final.txt"
