"""End-to-end test: run PSMC on synthetic data."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import smckit
from smckit.io import read_psmcfa
from smckit.tl._psmc import psmc


def _generate_synthetic_psmcfa(path: Path, n_sites: int = 50000, het_rate: float = 0.001, seed: int = 42):
    """Generate a synthetic PSMCFA file with constant Ne."""
    rng = np.random.default_rng(seed)
    seq = rng.choice(
        [ord("T"), ord("K")],
        size=n_sites,
        p=[1 - het_rate, het_rate],
    ).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b">synthetic_chr1\n")
        for i in range(0, len(seq), 60):
            f.write(bytes(seq[i : i + 60]) + b"\n")


@pytest.mark.slow
def test_psmc_runs(tmp_path):
    """Test that PSMC runs end-to-end and produces sane output."""
    fa = tmp_path / "test.psmcfa"
    _generate_synthetic_psmcfa(fa, n_sites=10000, het_rate=0.001)

    data = read_psmcfa(fa)
    data = psmc(
        data,
        pattern="1+1+1",
        n_iterations=3,
        max_t=15.0,
        tr_ratio=4.0,
        mu=1.25e-8,
        generation_time=25.0,
        seed=123,
    )

    res = data.results["psmc"]
    assert "ne" in res
    assert "time_years" in res
    assert "theta" in res
    assert "rho" in res
    assert len(res["rounds"]) == 4  # round 0 + 3 iterations
    assert res["theta"] > 0
    assert res["rho"] > 0
    assert np.all(res["ne"] > 0)
    assert np.all(np.isfinite(res["ne"]))
    assert res["log_likelihood"] < 0  # log prob is negative


@pytest.mark.slow
def test_psmc_likelihood_improves(tmp_path):
    """EM iterations should improve or maintain log-likelihood."""
    fa = tmp_path / "test.psmcfa"
    _generate_synthetic_psmcfa(fa, n_sites=5000, het_rate=0.001)

    data = read_psmcfa(fa)
    data = psmc(
        data,
        pattern="1+1+1",
        n_iterations=5,
        seed=42,
    )

    rounds = data.results["psmc"]["rounds"]
    lls = [r["log_likelihood"] for r in rounds if "log_likelihood" in r]
    # LL should generally increase (allow small fluctuations from Nelder-Mead)
    assert len(lls) >= 2
    # Final should be better than first
    assert lls[-1] >= lls[0] - abs(lls[0]) * 0.01


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 5, 6])
def test_psmc_real_fixture_avoids_zero_division(seed):
    """The packaged quickstart fixture should not fail for unlucky seeds."""
    data = read_psmcfa(smckit.io.example_path("psmc/NA12878_chr22.psmcfa"))
    data = psmc(
        data,
        pattern="4+5*3+4",
        n_iterations=2,
        max_t=15.0,
        tr_ratio=4.0,
        mu=1.25e-8,
        generation_time=25.0,
        implementation="native",
        seed=seed,
    )

    res = data.results["psmc"]
    assert res["theta"] > 0
    assert res["rho"] > 0
    assert np.all(np.isfinite(res["ne"]))
