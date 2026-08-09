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
