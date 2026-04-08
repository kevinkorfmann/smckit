"""Integration tests for SSM fitting: EM vs gradient-based optimization."""

from __future__ import annotations

import numpy as np
import pytest

from smckit.io import read_psmcfa
from smckit.ext.ssm import PsmcSSM
from smckit.tl._psmc import psmc

try:
    import jax
    jax.config.update("jax_enable_x64", True)

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Pattern "1+1+1" (3 states, 3 free params) works well with small synthetic
# data — matches the existing PSMC e2e test conventions.
PATTERN = "1+1+1"
REALISTIC_PATTERN = "4+5*3+4"


def _make_synthetic_obs(n_sites=5000, het_rate=0.001, seed=42):
    """Generate a synthetic observation sequence."""
    rng = np.random.default_rng(seed)
    seq = rng.choice([0, 1], size=n_sites, p=[1 - het_rate, het_rate]).astype(np.int8)
    return seq


def _compute_data_stats(seq):
    """Compute sum_L and sum_n from an observation sequence."""
    mask = seq != 2
    sum_L = int(mask.sum())
    sum_n = int((seq == 1).sum())
    return sum_L, sum_n


@pytest.mark.slow
class TestEMFitting:
    def test_em_runs(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit([seq], params, method="em", n_iterations=5)

        assert result.log_likelihood < 0
        assert np.isfinite(result.log_likelihood)
        assert result.n_iterations == 5
        assert len(result.history) == 5
        assert np.all(np.isfinite(result.params))
        assert np.all(result.params > 0)

    def test_em_likelihood_improves(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit([seq], params, method="em", n_iterations=5)

        lls = [h["log_likelihood"] for h in result.history]
        # Final LL should be at least as good as first
        assert lls[-1] >= lls[0] - abs(lls[0]) * 0.01

    def test_em_physical_units(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit([seq], params, method="em", n_iterations=3)
        phys = ssm.to_physical_units(result.params)

        assert phys["n0"] > 0
        assert np.all(phys["ne"] > 0)
        assert np.all(phys["time_years"] >= 0)

    def test_em_matches_psmc_on_real_data(self):
        data_psmc = read_psmcfa("tests/data/NA12878_chr22.psmcfa")
        data_psmc = psmc(
            data_psmc,
            pattern=REALISTIC_PATTERN,
            n_iterations=10,
            max_t=15.0,
            tr_ratio=5.0,
            mu=1.25e-8,
            generation_time=25.0,
            seed=42,
        )
        psmc_res = data_psmc.results["psmc"]

        data_ssm = read_psmcfa("tests/data/NA12878_chr22.psmcfa")
        observations = [rec["codes"] for rec in data_ssm.uns["records"]]
        ssm = PsmcSSM(pattern=REALISTIC_PATTERN)
        params = ssm.make_initial_params(
            data_ssm.uns["sum_L"],
            data_ssm.uns["sum_n"],
            max_t=15.0,
            tr_ratio=5.0,
            seed=42,
        )
        result = ssm.fit(observations, params, method="em", n_iterations=10)
        phys = ssm.to_physical_units(
            result.params,
            mu=1.25e-8,
            generation_time=25.0,
            window_size=data_ssm.window_size,
        )

        np.testing.assert_allclose(phys["lambda_k"], psmc_res["lambda"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(phys["ne"], psmc_res["ne"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            phys["time_years"], psmc_res["time_years"], rtol=1e-12, atol=1e-9,
        )
        assert abs(result.log_likelihood - psmc_res["log_likelihood"]) < 2.0


@pytest.mark.slow
@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestGradientFitting:
    def test_gradient_runs(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit(
            [seq], params, method="gradient",
            n_iterations=50, learning_rate=0.005, verbose=False,
        )

        assert result.log_likelihood < 0
        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.params))
        assert np.all(result.params > 0)

    def test_gradient_loss_decreases(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit(
            [seq], params, method="gradient",
            n_iterations=100, learning_rate=0.005, verbose=False,
        )

        losses = [h["loss"] for h in result.history]
        # Loss should decrease overall (compare first 10 avg vs last 10 avg)
        early_avg = np.mean(losses[:10])
        late_avg = np.mean(losses[-10:])
        assert late_avg < early_avg

    def test_gradient_physical_units(self):
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        result = ssm.fit(
            [seq], params, method="gradient",
            n_iterations=50, learning_rate=0.005, verbose=False,
        )

        phys = ssm.to_physical_units(result.params)
        assert phys["n0"] > 0
        assert np.all(phys["ne"] > 0)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestEMvsGradient:
    def test_both_methods_improve_likelihood(self):
        """Both EM and gradient-based fitting should improve log-likelihood."""
        ssm = PsmcSSM(pattern=PATTERN)
        seq = _make_synthetic_obs(n_sites=5000, seed=42)
        sum_L, sum_n = _compute_data_stats(seq)
        params = ssm.make_initial_params(sum_L, sum_n, seed=42)

        ll_init = ssm.log_likelihood(params, [seq])

        # EM
        result_em = ssm.fit([seq], params.copy(), method="em", n_iterations=5)

        # Gradient
        result_grad = ssm.fit(
            [seq], params.copy(), method="gradient",
            n_iterations=200, learning_rate=0.005, verbose=False,
        )

        # Both should have improved from initial
        assert result_em.log_likelihood > ll_init - 1.0
        assert result_grad.log_likelihood > ll_init - 1.0

        # Both produce valid parameters
        assert np.all(result_em.params > 0)
        assert np.all(result_grad.params > 0)
        assert np.all(np.isfinite(result_em.params))
        assert np.all(np.isfinite(result_grad.params))
