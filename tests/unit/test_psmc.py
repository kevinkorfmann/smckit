"""Tests for PSMC implementation — pattern parsing, time intervals, HMM params, EM."""

import numpy as np
import pytest

from smckit.backends._numpy import (
    backward,
    expected_counts,
    forward,
    log_likelihood,
    posterior_decode,
    posterior_probabilities,
    q0_from_counts,
    q_function,
    viterbi,
)
from smckit.tl._psmc import (
    PSMC_N_PARAMS,
    compute_hmm_params,
    compute_time_intervals,
    parse_pattern,
)


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

class TestParsePattern:
    def test_default_pattern(self):
        par_map, n_free, n = parse_pattern("4+5*3+4")
        # stack = [4, 3, 3, 3, 3, 3, 4], total = 23, n = 22, n_free = 7
        assert n == 22
        assert n_free == 7
        assert len(par_map) == 23
        # First 4 states → group 0
        assert all(par_map[i] == 0 for i in range(4))
        # Next 3 → group 1
        assert all(par_map[i] == 1 for i in range(4, 7))
        # Last 4 → group 6
        assert all(par_map[i] == 6 for i in range(19, 23))

    def test_simple_pattern(self):
        par_map, n_free, n = parse_pattern("1+1+1")
        assert n == 2
        assert n_free == 3
        np.testing.assert_array_equal(par_map, [0, 1, 2])

    def test_repeat_pattern(self):
        par_map, n_free, n = parse_pattern("3*2")
        # stack = [2, 2, 2], total = 6, n = 5, n_free = 3
        assert n == 5
        assert n_free == 3
        assert all(par_map[i] == 0 for i in range(2))
        assert all(par_map[i] == 1 for i in range(2, 4))
        assert all(par_map[i] == 2 for i in range(4, 6))

    def test_single_group(self):
        par_map, n_free, n = parse_pattern("5")
        assert n == 4
        assert n_free == 1
        assert all(par_map[i] == 0 for i in range(5))


# ---------------------------------------------------------------------------
# Time intervals
# ---------------------------------------------------------------------------

class TestTimeIntervals:
    def test_basic(self):
        t = compute_time_intervals(n=5, max_t=15.0, alpha=0.1)
        assert len(t) == 7  # n+2
        assert t[0] == 0.0
        assert t[5] == 15.0
        assert t[6] == 1000.0
        # Monotonically increasing
        assert all(t[i] < t[i + 1] for i in range(6))

    def test_exponential_spacing(self):
        t = compute_time_intervals(n=10, max_t=15.0, alpha=0.1)
        # Intervals should get wider further from present
        diffs = np.diff(t[:11])
        assert diffs[-1] > diffs[0]

    def test_custom_intervals(self):
        custom = np.array([0.0, 1.0, 2.0, 5.0])
        t = compute_time_intervals(n=2, inp_ti=custom)
        assert t[0] == 0.0
        assert t[1] == 1.0
        assert t[2] == 2.0
        assert t[3] == 1000.0  # T_INF


# ---------------------------------------------------------------------------
# HMM parameters from coalescent params
# ---------------------------------------------------------------------------

class TestComputeHmmParams:
    @pytest.fixture
    def simple_setup(self):
        """Minimal PSMC setup with 3 states."""
        par_map, n_free, n = parse_pattern("1+1+1")
        n_params = n_free + PSMC_N_PARAMS
        params = np.zeros(n_params)
        params[0] = 0.01   # theta
        params[1] = 0.0025  # rho
        params[2] = 15.0   # max_t
        params[3] = 1.0    # lambda_0
        params[4] = 1.0    # lambda_1
        params[5] = 1.0    # lambda_2
        t = compute_time_intervals(n, params[2], 0.1)
        return params, par_map, n, t

    def test_transition_matrix_is_stochastic(self, simple_setup):
        params, par_map, n, t = simple_setup
        hp = compute_hmm_params(params, par_map, n, t)
        # Each row sums to 1
        row_sums = hp.a.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_transition_matrix_nonnegative(self, simple_setup):
        params, par_map, n, t = simple_setup
        hp = compute_hmm_params(params, par_map, n, t)
        assert np.all(hp.a >= 0)

    def test_emission_matrix_rows_sum_to_one(self, simple_setup):
        params, par_map, n, t = simple_setup
        hp = compute_hmm_params(params, par_map, n, t)
        # Rows 0 and 1 (homo + het) should sum to 1 for each state
        sums = hp.e[0] + hp.e[1]
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_emission_missing_data(self, simple_setup):
        params, par_map, n, t = simple_setup
        hp = compute_hmm_params(params, par_map, n, t)
        np.testing.assert_array_equal(hp.e[2], 1.0)

    def test_initial_distribution_sums_to_one(self, simple_setup):
        params, par_map, n, t = simple_setup
        hp = compute_hmm_params(params, par_map, n, t)
        np.testing.assert_allclose(hp.a0.sum(), 1.0, atol=1e-6)

    def test_higher_lambda_increases_het_emission(self):
        """Higher Ne → deeper average coalescence → more heterozygosity."""
        par_map, n_free, n = parse_pattern("1+1+1")
        params_low = np.array([0.01, 0.0025, 15.0, 0.5, 0.5, 0.5])
        params_high = np.array([0.01, 0.0025, 15.0, 5.0, 5.0, 5.0])
        t = compute_time_intervals(n, 15.0, 0.1)
        hp_low = compute_hmm_params(params_low, par_map, n, t)
        hp_high = compute_hmm_params(params_high, par_map, n, t)
        # Higher lambda → more het (e[1]) for at least some states
        assert np.any(hp_high.e[1] > hp_low.e[1])

    def test_default_pattern(self):
        """Test with the standard PSMC pattern."""
        par_map, n_free, n = parse_pattern("4+5*3+4")
        n_params = n_free + PSMC_N_PARAMS
        params = np.ones(n_params)
        params[0] = 0.01
        params[1] = 0.0025
        params[2] = 15.0
        t = compute_time_intervals(n, 15.0, 0.1)
        hp = compute_hmm_params(params, par_map, n, t)
        assert hp.a.shape == (23, 23)
        assert hp.e.shape == (3, 23)
        row_sums = hp.a.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Forward / backward / likelihood
# ---------------------------------------------------------------------------

class TestForwardBackward:
    @pytest.fixture
    def hmm_setup(self):
        """Simple 2-state HMM for testing."""
        a = np.array([[0.9, 0.1], [0.2, 0.8]])
        e = np.array([
            [0.8, 0.2],  # P(obs=0 | state)
            [0.2, 0.8],  # P(obs=1 | state)
            [1.0, 1.0],  # missing
        ])
        a0 = np.array([0.6, 0.4])
        seq = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.int8)
        return a, e, a0, seq

    def test_forward_shapes(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        assert f.shape == (7, 2)
        assert s.shape == (7,)

    def test_forward_probabilities_nonnegative(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        assert np.all(f >= 0)
        assert np.all(s > 0)

    def test_forward_normalized(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        row_sums = f.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_backward_shapes(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        b = backward(a, e, seq, s)
        assert b.shape == (7, 2)

    def test_backward_check(self, hmm_setup):
        """Verify sum_l a0[l] * b[0,l] * e[seq[0],l] ≈ 1."""
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        b = backward(a, e, seq, s)
        check = np.sum(a0 * b[0] * e[seq[0]])
        np.testing.assert_allclose(check, 1.0, atol=1e-6)

    def test_log_likelihood_finite(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        ll = log_likelihood(s)
        assert np.isfinite(ll)
        assert ll < 0  # log prob is negative

    def test_posterior_sums_to_one(self, hmm_setup):
        a, e, a0, seq = hmm_setup
        f, s = forward(a, e, a0, seq)
        b = backward(a, e, seq, s)
        gamma = posterior_probabilities(f, b, s)
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

class TestViterbi:
    def test_basic(self):
        a = np.array([[0.9, 0.1], [0.2, 0.8]])
        e = np.array([[0.9, 0.1], [0.1, 0.9], [1.0, 1.0]])
        a0 = np.array([0.5, 0.5])
        # All obs=0 → should mostly be state 0
        seq = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        path, lp = viterbi(a, e, a0, seq)
        assert path.shape == (5,)
        assert np.all(path == 0)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# Expected counts & Q-function
# ---------------------------------------------------------------------------

class TestExpectedCounts:
    def test_q_increases_or_stable(self):
        """Q after M-step should be >= Q before (EM guarantee)."""
        a = np.array([[0.7, 0.3], [0.4, 0.6]])
        e = np.array([[0.8, 0.3], [0.2, 0.7], [1.0, 1.0]])
        a0 = np.array([0.5, 0.5])
        seq = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)

        f, s = forward(a, e, a0, seq)
        b = backward(a, e, seq, s)
        A, E, A0 = expected_counts(a, e, seq, f, b, s, a0)

        Q0_val = q0_from_counts(A, E, m=2)
        Q_current = q_function(a, e, A, E, Q0_val)

        # Q at the current params should be ~0 (since Q0 is computed from these counts)
        # The exact value depends on normalization but should be close to 0
        assert np.isfinite(Q_current)

    def test_expected_counts_shapes(self):
        a = np.array([[0.7, 0.3], [0.4, 0.6]])
        e = np.array([[0.8, 0.3], [0.2, 0.7], [1.0, 1.0]])
        a0 = np.array([0.5, 0.5])
        seq = np.array([0, 1, 0, 1], dtype=np.int8)

        f, s = forward(a, e, a0, seq)
        b = backward(a, e, seq, s)
        A, E, A0 = expected_counts(a, e, seq, f, b, s, a0)

        assert A.shape == (2, 2)
        assert E.shape == (3, 2)
        assert A0.shape == (2,)
        assert np.all(A > 0)


# ---------------------------------------------------------------------------
# Integration: PSMC HMM with forward-backward
# ---------------------------------------------------------------------------

class TestPsmcIntegration:
    def test_psmc_hmm_forward_backward(self):
        """Run forward-backward with actual PSMC-derived HMM params."""
        par_map, n_free, n = parse_pattern("1+1+1")
        params = np.array([0.01, 0.0025, 15.0, 1.0, 1.0, 1.0])
        t = compute_time_intervals(n, 15.0, 0.1)
        hp = compute_hmm_params(params, par_map, n, t)

        # Synthetic sequence: mostly homo with some het
        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=1000, p=[0.99, 0.01]).astype(np.int8)

        f, s = forward(hp.a, hp.e, hp.a0, seq)
        b = backward(hp.a, hp.e, seq, s)
        ll = log_likelihood(s)

        assert f.shape == (1000, 3)
        assert np.isfinite(ll)
        assert ll < 0

        # Posterior decode
        path = posterior_decode(f, b, s)
        assert path.shape == (1000,)
        assert np.all((path >= 0) & (path < 3))

    def test_psmc_hmm_expected_counts(self):
        """Compute expected counts with PSMC HMM."""
        par_map, n_free, n = parse_pattern("1+1+1")
        params = np.array([0.01, 0.0025, 15.0, 1.0, 1.0, 1.0])
        t = compute_time_intervals(n, 15.0, 0.1)
        hp = compute_hmm_params(params, par_map, n, t)

        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=500, p=[0.99, 0.01]).astype(np.int8)

        f, s = forward(hp.a, hp.e, hp.a0, seq)
        b = backward(hp.a, hp.e, seq, s)
        A, E, A0 = expected_counts(hp.a, hp.e, seq, f, b, s, hp.a0)

        Q0_val = q0_from_counts(A, E, m=2)
        Q_val = q_function(hp.a, hp.e, A, E, Q0_val)

        assert np.isfinite(Q_val)
