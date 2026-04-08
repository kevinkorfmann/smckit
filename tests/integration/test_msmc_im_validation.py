"""MSMC-IM validation against the vendored upstream CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from smckit.io import read_msmc_combined_output, read_msmc_im_output
from smckit.tl import msmc_im

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

NATIVE_ORACLE_CASES = [
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
]


def _run_vendored_cli(
    tmp_path: Path,
    *,
    name: str = "yoruba_french",
    pattern: str = "1*2+25*1+1*2+1*3",
    beta: tuple[float, float] = (1e-8, 1e-6),
    mu: float = 1.25e-8,
    N1_init: float = 15000.0,
    N2_init: float = 15000.0,
    m_init: float = 1e-4,
) -> dict[str, np.ndarray]:
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
            "--xlog",
            str(INPUT),
        ],
        cwd=UPSTREAM.parent,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    matches = sorted(tmp_path.glob(f"{name}.b1_*.b2_*.MSMC_IM.estimates.txt"))
    assert len(matches) == 1
    return read_msmc_im_output(matches[0])


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


def _assert_split_quantiles_match(result: dict[str, object]) -> None:
    expected = _expected_split_time_quantiles(
        np.asarray(result["left_boundary"], dtype=np.float64),
        np.asarray(result["M"], dtype=np.float64),
    )
    assert result["split_time_quantiles"].keys() == expected.keys()
    for q, t in expected.items():
        assert result["split_time_quantiles"][q] == pytest.approx(t)


def _assert_matches_oracle(
    result: dict[str, object],
    ref: dict[str, np.ndarray],
    *,
    mu: float = 1.25e-8,
) -> None:
    np.testing.assert_allclose(
        result["left_boundary"],
        ref["left_boundary"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(result["N1"], ref["N1"], rtol=5e-10, atol=1e-12)
    np.testing.assert_allclose(result["N2"], ref["N2"], rtol=5e-10, atol=1e-12)
    np.testing.assert_allclose(result["m"], ref["m"], rtol=5e-10, atol=1e-12)
    np.testing.assert_allclose(result["M"], ref["M"], rtol=5e-10, atol=1e-12)
    np.testing.assert_allclose(
        result["right_boundary"],
        _expected_right_boundary(mu=mu),
        rtol=0.0,
        atol=0.0,
    )
    _assert_split_quantiles_match(result)


def test_msmc_im_matches_vendored_cli(tmp_path: Path) -> None:
    ref = _run_vendored_cli(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ours = msmc_im(INPUT, implementation="native").results["msmc_im"]

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []

    _assert_matches_oracle(ours, ref)


@pytest.mark.parametrize(
    ("name", "pattern", "beta", "mu", "N1_init", "N2_init", "m_init"),
    NATIVE_ORACLE_CASES,
)
def test_msmc_im_matches_vendored_cli_on_nondefault_cases(
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
        ours = msmc_im(
            INPUT,
            implementation="native",
            pattern=pattern,
            beta=beta,
            mu=mu,
            N1_init=N1_init,
            N2_init=N2_init,
            m_init=m_init,
        ).results["msmc_im"]

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []

    _assert_matches_oracle(ours, ref, mu=mu)


def test_msmc_im_public_upstream_runner_matches_vendored_cli(tmp_path: Path) -> None:
    ref = _run_vendored_cli(tmp_path)
    upstream = msmc_im(INPUT, implementation="upstream").results["msmc_im"]

    _assert_matches_oracle(upstream, ref)
    assert upstream["implementation"] == "upstream"
    assert upstream["upstream"]["effective_args"]["pattern"] == "1*2+25*1+1*2+1*3"
    assert upstream["upstream"]["effective_args"]["mu"] == pytest.approx(1.25e-8)
