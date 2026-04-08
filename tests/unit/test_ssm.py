"""Tests for the SSM extension framework."""

from __future__ import annotations

import numpy as np
import pytest

from smckit.backends._numba import compute_hmm_params_jit, compute_time_intervals_jit
from smckit.ext.ssm._base import FitResult, SmcStateSpace
from smckit.ext.ssm._psmc_ssm import PsmcSSM

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


class TestSmcStateSpace:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SmcStateSpace()

    def test_subclass_with_methods_works(self):
        class DummySSM(SmcStateSpace):
            def __init__(self):
                self.n_states = 2
                self.n_obs = 2

            def transition_matrix(self, params):
                return np.eye(2)

            def emission_matrix(self, params):
                return np.eye(2)

            def initial_distribution(self, params):
                return np.array([0.5, 0.5])

        ssm = DummySSM()
        assert ssm.n_states == 2

    def test_fit_result_fields(self):
        r = FitResult(
            params=np.array([1.0, 2.0]),
            log_likelihood=-100.0,
            n_iterations=10,
            converged=True,
        )
        assert r.log_likelihood == -100.0
        assert r.converged
        assert r.history == []


# ---------------------------------------------------------------------------
# PsmcSSM tests
# ---------------------------------------------------------------------------


class TestPsmcSSM:
    @pytest.fixture
    def ssm(self):
        return PsmcSSM(pattern="4+5*3+4")

    @pytest.fixture
    def simple_ssm(self):
        return PsmcSSM(pattern="4+2")

    @pytest.fixture
    def params(self, ssm):
        """Realistic parameter vector."""
        n_params = ssm.n_free + 3
        p = np.ones(n_params, dtype=np.float64)
        p[0] = 0.01  # theta
        p[1] = 0.003  # rho
        p[2] = 15.0  # max_t
        return p

    @pytest.fixture
    def simple_params(self, simple_ssm):
        n_params = simple_ssm.n_free + 3
        p = np.ones(n_params, dtype=np.float64)
        p[0] = 0.01
        p[1] = 0.003
        p[2] = 15.0
        return p

    def test_init_default_pattern(self, ssm):
        assert ssm.n == 22
        assert ssm.n_states == 23
        assert ssm.n_free == 7
        assert ssm.n_obs == 3

    def test_init_simple_pattern(self, simple_ssm):
        # pattern "4+2" = 6 states, n = 5, n_free = 2
        assert simple_ssm.n == 5
        assert simple_ssm.n_states == 6
        assert simple_ssm.n_free == 2

    def test_transition_matrix_stochastic(self, ssm, params):
        a = ssm.transition_matrix(params)
        assert a.shape == (23, 23)
        np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(a >= 0)

    def test_emission_matrix(self, ssm, params):
        e = ssm.emission_matrix(params)
        assert e.shape == (3, 23)
        # Rows 0+1 should sum to 1 (homo + het)
        np.testing.assert_allclose(e[0] + e[1], 1.0, atol=1e-12)
        # Row 2 (missing) should be all ones
        np.testing.assert_allclose(e[2], 1.0, atol=1e-12)
        assert np.all(e >= 0)

    def test_initial_distribution(self, ssm, params):
        sigma = ssm.initial_distribution(params)
        assert sigma.shape == (23,)
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-10)
        assert np.all(sigma >= 0)

    def test_matches_numba_backend(self, ssm, params):
        """PsmcSSM matrices must match compute_hmm_params_jit exactly."""
        t = compute_time_intervals_jit(ssm.n, float(params[2]), ssm.alpha)
        a_ref, e_ref, sigma_ref, _, _ = compute_hmm_params_jit(
            params, ssm.par_map, ssm.n, t, False,
        )

        a = ssm.transition_matrix(params)
        e = ssm.emission_matrix(params)
        sigma = ssm.initial_distribution(params)

        np.testing.assert_allclose(a, a_ref, atol=1e-12)
        np.testing.assert_allclose(e, e_ref, atol=1e-12)
        np.testing.assert_allclose(sigma, sigma_ref, atol=1e-12)

    def test_make_initial_params(self, ssm):
        params = ssm.make_initial_params(sum_L=100000, sum_n=100, seed=42)
        assert params.shape == (ssm.n_free + 3,)
        assert params[0] > 0  # theta
        assert params[1] > 0  # rho
        assert params[2] == 15.0  # max_t
        assert np.all(params[3:] > 0)  # lambdas

    def test_to_physical_units(self, ssm, params):
        result = ssm.to_physical_units(params)
        assert "ne" in result
        assert "time_years" in result
        assert "lambda_k" in result
        assert "n0" in result
        assert result["n0"] > 0
        assert len(result["ne"]) == ssm.n_states
        assert np.all(result["ne"] > 0)

    def test_log_likelihood(self, simple_ssm, simple_params):
        """Log-likelihood via base class should return finite negative value."""
        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=500, p=[0.99, 0.01]).astype(np.int8)
        ll = simple_ssm.log_likelihood(simple_params, [seq])
        assert np.isfinite(ll)
        assert ll < 0


# ---------------------------------------------------------------------------
# JAX backend tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJaxBackend:
    @pytest.fixture
    def ssm(self):
        return PsmcSSM(pattern="4+5*3+4")

    @pytest.fixture
    def simple_ssm(self):
        return PsmcSSM(pattern="4+2")

    @pytest.fixture
    def params(self, ssm):
        n_params = ssm.n_free + 3
        p = np.ones(n_params, dtype=np.float64)
        p[0] = 0.01
        p[1] = 0.003
        p[2] = 15.0
        return p

    @pytest.fixture
    def simple_params(self, simple_ssm):
        n_params = simple_ssm.n_free + 3
        p = np.ones(n_params, dtype=np.float64)
        p[0] = 0.01
        p[1] = 0.003
        p[2] = 15.0
        return p

    def test_time_intervals_match(self, ssm, params):
        from smckit.ext.ssm._jax_backend import compute_time_intervals_jax

        t_numba = compute_time_intervals_jit(ssm.n, float(params[2]), ssm.alpha)
        t_jax = compute_time_intervals_jax(ssm.n, params[2], ssm.alpha)

        np.testing.assert_allclose(np.asarray(t_jax), t_numba, atol=1e-12)

    def test_hmm_params_match_numba(self, ssm, params):
        """JAX compute_hmm_params must match Numba output."""
        from smckit.ext.ssm._jax_backend import (
            compute_hmm_params_jax,
            compute_time_intervals_jax,
        )

        t_numba = compute_time_intervals_jit(ssm.n, float(params[2]), ssm.alpha)
        a_ref, e_ref, sigma_ref, _, _ = compute_hmm_params_jit(
            params, ssm.par_map, ssm.n, t_numba, False,
        )

        t_jax = compute_time_intervals_jax(ssm.n, params[2], ssm.alpha)
        par_map_jax = jnp.array(ssm.par_map, dtype=jnp.int32)
        params_jax = jnp.array(params, dtype=jnp.float64)
        a_jax, e_jax, sigma_jax = compute_hmm_params_jax(
            params_jax, par_map_jax, ssm.n, t_jax, False,
        )

        np.testing.assert_allclose(np.asarray(a_jax), a_ref, atol=1e-10)
        np.testing.assert_allclose(np.asarray(e_jax), e_ref, atol=1e-10)
        np.testing.assert_allclose(np.asarray(sigma_jax), sigma_ref, atol=1e-10)

    def test_hmm_params_match_varying_lambda(self, ssm):
        """Test with non-uniform lambda values."""
        from smckit.ext.ssm._jax_backend import (
            compute_hmm_params_jax,
            compute_time_intervals_jax,
        )

        rng = np.random.default_rng(123)
        n_params = ssm.n_free + 3
        params = np.zeros(n_params, dtype=np.float64)
        params[0] = 0.015
        params[1] = 0.004
        params[2] = 10.0
        params[3:] = 0.5 + rng.random(ssm.n_free)

        t_numba = compute_time_intervals_jit(ssm.n, float(params[2]), ssm.alpha)
        a_ref, e_ref, sigma_ref, _, _ = compute_hmm_params_jit(
            params, ssm.par_map, ssm.n, t_numba, False,
        )

        t_jax = compute_time_intervals_jax(ssm.n, params[2], ssm.alpha)
        a_jax, e_jax, sigma_jax = compute_hmm_params_jax(
            jnp.array(params), jnp.array(ssm.par_map, dtype=jnp.int32),
            ssm.n, t_jax, False,
        )

        np.testing.assert_allclose(np.asarray(a_jax), a_ref, atol=1e-10)
        np.testing.assert_allclose(np.asarray(e_jax), e_ref, atol=1e-10)
        np.testing.assert_allclose(np.asarray(sigma_jax), sigma_ref, atol=1e-10)

    def test_forward_log_matches_numba(self, simple_ssm, simple_params):
        """JAX log-space forward must produce same log-likelihood as Numba."""
        from smckit.backends._numba import forward_jit, log_likelihood_jit
        from smckit.ext.ssm._jax_backend import (
            compute_hmm_params_jax,
            compute_time_intervals_jax,
            forward_log,
        )

        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=1000, p=[0.99, 0.01]).astype(np.int8)

        # Numba reference
        t_numba = compute_time_intervals_jit(
            simple_ssm.n, float(simple_params[2]), simple_ssm.alpha,
        )
        a_ref, e_ref, sigma_ref, _, _ = compute_hmm_params_jit(
            simple_params, simple_ssm.par_map, simple_ssm.n, t_numba, False,
        )
        f, s = forward_jit(a_ref, e_ref, sigma_ref, seq)
        ll_numba = log_likelihood_jit(s)

        # JAX
        t_jax = compute_time_intervals_jax(
            simple_ssm.n, simple_params[2], simple_ssm.alpha,
        )
        a_jax, e_jax, sigma_jax = compute_hmm_params_jax(
            jnp.array(simple_params), jnp.array(simple_ssm.par_map, dtype=jnp.int32),
            simple_ssm.n, t_jax, False,
        )
        log_a = jnp.log(jnp.maximum(a_jax, 1e-300))
        log_e = jnp.log(jnp.maximum(e_jax, 1e-300))
        log_a0 = jnp.log(jnp.maximum(sigma_jax, 1e-300))
        ll_jax = forward_log(log_a, log_e, log_a0, jnp.array(seq, dtype=jnp.int32))

        np.testing.assert_allclose(float(ll_jax), ll_numba, atol=1e-4)

    def test_gradient_is_finite(self, simple_ssm, simple_params):
        """Gradient of negative log-likelihood must be finite."""
        from smckit.ext.ssm._jax_backend import _neg_log_likelihood

        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=500, p=[0.99, 0.01]).astype(np.int8)
        obs = [jnp.array(seq, dtype=jnp.int32)]

        x = jnp.log(jnp.array(simple_params, dtype=jnp.float64))
        par_map = jnp.array(simple_ssm.par_map, dtype=jnp.int32)

        grad_fn = jax.grad(
            lambda x: _neg_log_likelihood(
                x, par_map, simple_ssm.n, simple_ssm.alpha, obs, False
            )
        )
        grads = grad_fn(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_gradient_direction(self, simple_ssm, simple_params):
        """Verify gradients point in sensible directions."""
        from smckit.ext.ssm._jax_backend import _neg_log_likelihood

        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=500, p=[0.99, 0.01]).astype(np.int8)
        obs = [jnp.array(seq, dtype=jnp.int32)]

        x = jnp.log(jnp.array(simple_params, dtype=jnp.float64))
        par_map = jnp.array(simple_ssm.par_map, dtype=jnp.int32)

        def loss_fn(x):
            return _neg_log_likelihood(
                x, par_map, simple_ssm.n, simple_ssm.alpha, obs, False
            )

        loss0 = loss_fn(x)
        grads = jax.grad(loss_fn)(x)

        # Small step in negative gradient direction should reduce loss
        x_new = x - 0.001 * grads
        loss1 = loss_fn(x_new)
        assert float(loss1) <= float(loss0) + 1e-6  # allow tiny numerical noise
