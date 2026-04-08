"""Tests for eSMC2 implementation — HMM construction, forward-backward, EM."""

import numpy as np
import pytest

from smckit.backends._numba_esmc2 import (
    esmc2_backward,
    esmc2_build_emission_matrix,
    esmc2_build_hmm,
    esmc2_build_time_boundaries,
    esmc2_build_transition_matrix,
    esmc2_equilibrium_probs,
    esmc2_expected_counts,
    esmc2_expected_times,
    esmc2_forward,
    esmc2_forward_loglik,
)
from smckit.tl._esmc2 import _zip_sequence_to_upstream_numeric

# ---------------------------------------------------------------------------
# Time boundaries
# ---------------------------------------------------------------------------


class TestTimeBoundaries:
    def test_basic_properties(self):
        Tc = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        assert len(Tc) == 10
        assert Tc[0] == 0.0
        # Monotonically increasing
        for i in range(1, len(Tc)):
            assert Tc[i] > Tc[i - 1]

    def test_dormancy_scales_time(self):
        """Dormancy (β < 1) should stretch time boundaries by 1/β²."""
        Tc_no_dorm = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Tc_dorm = esmc2_build_time_boundaries(10, beta=0.5, sigma=0.0)
        # With β=0.5, Tc should be 4x larger (1/β² = 4)
        np.testing.assert_allclose(Tc_dorm, Tc_no_dorm * 4.0, rtol=1e-10)

    def test_selfing_scales_time(self):
        """Selfing (σ > 0) should shrink time boundaries by (2-σ)/2."""
        Tc_no_self = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Tc_self = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.5)
        # With σ=0.5, factor is (2-0.5)/2 = 0.75
        np.testing.assert_allclose(Tc_self, Tc_no_self * 0.75, rtol=1e-10)

    def test_reduces_to_psmc_without_dormancy_selfing(self):
        """With β=1, σ=0, should give standard PSMC-like time boundaries."""
        Tc = esmc2_build_time_boundaries(20, beta=1.0, sigma=0.0)
        # Tc[k] = -log(1 - k/n) for standard coalescent
        for k in range(1, 20):
            expected = -np.log(1.0 - k / 20.0)
            np.testing.assert_allclose(Tc[k], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Expected coalescent times
# ---------------------------------------------------------------------------


class TestExpectedTimes:
    def test_basic_properties(self):
        Tc = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Xi = np.ones(10, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.0)
        assert len(t) == 10
        # t[k] should be between Tc[k] and Tc[k+1] for interior states
        for k in range(9):
            assert t[k] >= Tc[k], f"t[{k}]={t[k]} < Tc[{k}]={Tc[k]}"
            assert t[k] <= Tc[k + 1], f"t[{k}]={t[k]} > Tc[{k + 1}]={Tc[k + 1]}"
        # Last state: t[n-1] >= Tc[n-1]
        assert t[9] >= Tc[9]

    def test_monotonically_increasing(self):
        Tc = esmc2_build_time_boundaries(15, beta=0.8, sigma=0.3)
        Xi = np.ones(15, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=0.8, sigma=0.3)
        for i in range(1, len(t)):
            assert t[i] > t[i - 1]


# ---------------------------------------------------------------------------
# Equilibrium probabilities
# ---------------------------------------------------------------------------


class TestEquilibriumProbs:
    def test_sums_to_one(self):
        Tc = esmc2_build_time_boundaries(20, beta=1.0, sigma=0.0)
        Xi = np.ones(20, dtype=np.float64)
        q = esmc2_equilibrium_probs(Tc, Xi, beta=1.0, sigma=0.0)
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-10)

    def test_all_positive(self):
        Tc = esmc2_build_time_boundaries(15, beta=0.7, sigma=0.2)
        Xi = np.ones(15, dtype=np.float64) * 1.5
        q = esmc2_equilibrium_probs(Tc, Xi, beta=0.7, sigma=0.2)
        assert np.all(q > 0)

    def test_with_variable_pop_sizes(self):
        Tc = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Xi = np.array([0.5, 1.0, 1.5, 2.0, 1.0, 0.8, 0.6, 1.2, 1.0, 0.5])
        q = esmc2_equilibrium_probs(Tc, Xi, beta=1.0, sigma=0.0)
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-10)
        assert np.all(q > 0)


# ---------------------------------------------------------------------------
# Emission matrix
# ---------------------------------------------------------------------------


class TestEmissionMatrix:
    def test_basic_structure(self):
        Tc = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Xi = np.ones(10, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.0)
        e = esmc2_build_emission_matrix(mu=0.001, mu_b=1.0, Tc=Tc, t=t, beta=1.0, n=10)

        assert e.shape == (3, 10)
        # P(hom) + P(het) = 1 for each state
        for k in range(10):
            np.testing.assert_allclose(e[0, k] + e[1, k], 1.0, atol=1e-10)
        # P(missing) = 1
        np.testing.assert_allclose(e[2, :], 1.0)

    def test_het_increases_with_time(self):
        """More recent coalescences → fewer mutations → more homozygous."""
        Tc = esmc2_build_time_boundaries(10, beta=1.0, sigma=0.0)
        Xi = np.ones(10, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.0)
        e = esmc2_build_emission_matrix(mu=0.001, mu_b=1.0, Tc=Tc, t=t, beta=1.0, n=10)

        # P(het) should increase with state index (deeper coalescence)
        for k in range(1, 10):
            assert e[1, k] >= e[1, k - 1]

    def test_seed_bank_mutation_rate(self):
        """mu_b < 1 reduces effective mutation rate in seed bank periods."""
        Tc = esmc2_build_time_boundaries(10, beta=0.5, sigma=0.0)
        Xi = np.ones(10, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=0.5, sigma=0.0)

        e_std = esmc2_build_emission_matrix(mu=0.001, mu_b=1.0, Tc=Tc, t=t, beta=0.5, n=10)
        e_low = esmc2_build_emission_matrix(mu=0.001, mu_b=0.5, Tc=Tc, t=t, beta=0.5, n=10)

        # Lower mu_b → less mutation → more homozygous (higher e[0])
        for k in range(1, 10):
            assert e_low[0, k] >= e_std[0, k] - 1e-10


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_columns_sum_to_one(self):
        n = 10
        Tc = esmc2_build_time_boundaries(n, beta=1.0, sigma=0.0)
        Xi = np.ones(n, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.0)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta=1.0, sigma=0.0, rho=10.0, L=10000)

        col_sums = Q.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)

    def test_non_negative(self):
        n = 15
        Tc = esmc2_build_time_boundaries(n, beta=0.8, sigma=0.2)
        Xi = np.ones(n, dtype=np.float64) * 1.2
        t = esmc2_expected_times(Tc, Xi, beta=0.8, sigma=0.2)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta=0.8, sigma=0.2, rho=5.0, L=50000)

        assert np.all(Q >= -1e-10), f"Negative entries: min={Q.min()}"

    def test_diagonal_dominant(self):
        """Transition matrix should be diagonally dominant for reasonable parameters."""
        n = 10
        Tc = esmc2_build_time_boundaries(n, beta=1.0, sigma=0.0)
        Xi = np.ones(n, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.0)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta=1.0, sigma=0.0, rho=5.0, L=100000)

        for i in range(n):
            assert Q[i, i] > 0.5, f"Q[{i},{i}]={Q[i, i]} not dominant"

    def test_with_dormancy(self):
        """Transition matrix should still be valid with dormancy."""
        n = 10
        Tc = esmc2_build_time_boundaries(n, beta=0.5, sigma=0.0)
        Xi = np.ones(n, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=0.5, sigma=0.0)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta=0.5, sigma=0.0, rho=5.0, L=50000)

        col_sums = Q.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)

    def test_with_selfing(self):
        """Transition matrix should still be valid with selfing."""
        n = 10
        Tc = esmc2_build_time_boundaries(n, beta=1.0, sigma=0.8)
        Xi = np.ones(n, dtype=np.float64)
        t = esmc2_expected_times(Tc, Xi, beta=1.0, sigma=0.8)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta=1.0, sigma=0.8, rho=5.0, L=50000)

        col_sums = Q.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Full HMM build
# ---------------------------------------------------------------------------


class TestBuildHMM:
    def test_all_outputs_valid(self):
        n = 10
        Xi = np.ones(n, dtype=np.float64)
        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, 1.0, 0.0, 5.0, 0.001, 1.0, 50000)

        assert Q.shape == (n, n)
        assert q.shape == (n,)
        assert t.shape == (n,)
        assert Tc.shape == (n,)
        assert e.shape == (3, n)

        np.testing.assert_allclose(Q.sum(axis=0), 1.0, atol=1e-8)
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Forward-backward algorithms
# ---------------------------------------------------------------------------


class TestForwardBackward:
    @pytest.fixture
    def simple_hmm(self):
        n = 5
        Xi = np.ones(n, dtype=np.float64)
        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, 1.0, 0.0, 2.0, 0.001, 1.0, 10000)
        return Q, q, e

    def test_forward_loglik_finite(self, simple_hmm):
        Q, q, e = simple_hmm
        seq = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
        ll = esmc2_forward_loglik(Q, e, q, seq)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative

    def test_forward_returns_valid(self, simple_hmm):
        Q, q, e = simple_hmm
        seq = np.array([0, 0, 1, 0, 0], dtype=np.int8)
        f, s = esmc2_forward(Q, e, q, seq)

        assert f.shape == (5, 5)
        assert s.shape == (5,)
        # Forward probabilities sum to ~1 at each position (after scaling)
        for u in range(5):
            np.testing.assert_allclose(f[u].sum(), 1.0, atol=1e-8)

    def test_backward_returns_valid(self, simple_hmm):
        Q, q, e = simple_hmm
        seq = np.array([0, 0, 1, 0, 0], dtype=np.int8)
        f, s = esmc2_forward(Q, e, q, seq)
        b = esmc2_backward(Q, e, seq, s)

        assert b.shape == (5, 5)
        # Last position backward should be all 1s
        np.testing.assert_allclose(b[4], 1.0)

    def test_forward_backward_consistency(self, simple_hmm):
        """Forward * backward should give consistent posteriors."""
        Q, q, e = simple_hmm
        seq = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        f, s = esmc2_forward(Q, e, q, seq)
        b = esmc2_backward(Q, e, seq, s)

        # gamma[u] = f[u] * b[u] / sum(f[u] * b[u])
        for u in range(len(seq)):
            gamma = f[u] * b[u]
            gamma_sum = gamma.sum()
            assert gamma_sum > 0
            gamma /= gamma_sum
            np.testing.assert_allclose(gamma.sum(), 1.0, atol=1e-10)

    def test_loglik_matches_forward(self, simple_hmm):
        """forward_loglik should match sum of log scaling factors."""
        Q, q, e = simple_hmm
        seq = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
        ll_direct = esmc2_forward_loglik(Q, e, q, seq)
        f, s = esmc2_forward(Q, e, q, seq)
        ll_from_s = np.sum(np.log(s))
        np.testing.assert_allclose(ll_direct, ll_from_s, rtol=1e-8)


# ---------------------------------------------------------------------------
# Expected counts
# ---------------------------------------------------------------------------


class TestExpectedCounts:
    def test_transition_counts_shape(self):
        n = 5
        Xi = np.ones(n, dtype=np.float64)
        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, 1.0, 0.0, 2.0, 0.001, 1.0, 10000)
        seq = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
        f, s = esmc2_forward(Q, e, q, seq)
        b = esmc2_backward(Q, e, seq, s)
        N, M = esmc2_expected_counts(Q, e, seq, f, b, s, q)

        assert N.shape == (n, n)
        assert M.shape == (3, n)
        assert np.all(N >= -1e-10)
        assert np.all(M >= -1e-10)

    def test_emission_counts_sum(self):
        """Total emission counts should approximately equal sequence length."""
        n = 5
        Xi = np.ones(n, dtype=np.float64)
        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, 1.0, 0.0, 2.0, 0.001, 1.0, 10000)
        seq = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        f, s = esmc2_forward(Q, e, q, seq)
        b = esmc2_backward(Q, e, seq, s)
        N, M = esmc2_expected_counts(Q, e, seq, f, b, s, q)

        np.testing.assert_allclose(M.sum(), len(seq), atol=0.1)


class TestZipEncoding:
    def test_zip_sequence_matches_upstream_numeric_encoding(self):
        seq = np.array([0] * 16 + [1] + [0] * 23, dtype=np.int8)
        zipped = _zip_sequence_to_upstream_numeric(seq)
        np.testing.assert_array_equal(
            zipped,
            np.array([0, 0, 0, 13, 12, 11, 1, 14, 12, 11, 0], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Integration: eSMC2 on synthetic data
# ---------------------------------------------------------------------------


class TestEsmc2Integration:
    def test_runs_on_psmcfa_data(self):
        """eSMC2 should run without errors on PSMC-style input."""
        from smckit._core import SmcData

        rng = np.random.default_rng(42)
        L = 5000
        seq = rng.choice([0, 1], size=L, p=[0.99, 0.01]).astype(np.int8)

        data = SmcData()
        data.uns["records"] = [{"codes": seq}]
        data.uns["sum_L"] = int((seq < 2).sum())
        data.uns["sum_n"] = int((seq == 1).sum())

        from smckit.tl._esmc2 import esmc2

        result = esmc2(
            data,
            n_states=10,
            n_iterations=2,
            estimate_rho=False,
            mu=1e-8,
            generation_time=1.0,
            implementation="native",
        )

        assert "esmc2" in result.results
        r = result.results["esmc2"]
        assert "ne" in r
        assert "time_years" in r
        assert "Xi" in r
        assert len(r["ne"]) == 10
        assert r["log_likelihood"] < 0
        assert np.all(np.isfinite(r["ne"]))

    def test_runs_with_dormancy_estimation(self):
        """eSMC2 should run when estimating dormancy."""
        from smckit._core import SmcData

        rng = np.random.default_rng(123)
        L = 3000
        seq = rng.choice([0, 1], size=L, p=[0.99, 0.01]).astype(np.int8)

        data = SmcData()
        data.uns["records"] = [{"codes": seq}]
        data.uns["sum_L"] = int((seq < 2).sum())
        data.uns["sum_n"] = int((seq == 1).sum())

        from smckit.tl._esmc2 import esmc2

        result = esmc2(
            data,
            n_states=8,
            n_iterations=2,
            estimate_beta=True,
            estimate_rho=False,
            beta=0.8,
            mu=1e-8,
            implementation="native",
        )

        r = result.results["esmc2"]
        assert 0.0 < r["beta"] <= 1.0

    def test_runs_with_selfing_estimation(self):
        """eSMC2 should run when estimating selfing."""
        from smckit._core import SmcData

        rng = np.random.default_rng(456)
        L = 3000
        seq = rng.choice([0, 1], size=L, p=[0.99, 0.01]).astype(np.int8)

        data = SmcData()
        data.uns["records"] = [{"codes": seq}]
        data.uns["sum_L"] = int((seq < 2).sum())
        data.uns["sum_n"] = int((seq == 1).sum())

        from smckit.tl._esmc2 import esmc2

        result = esmc2(
            data,
            n_states=8,
            n_iterations=2,
            estimate_sigma=True,
            estimate_rho=False,
            sigma=0.3,
            mu=1e-8,
            implementation="native",
        )

        r = result.results["esmc2"]
        assert 0.0 <= r["sigma"] < 1.0

    def test_theta_uses_callable_site_scaling(self):
        """Theta should follow the upstream callable-site normalization."""
        from smckit._core import SmcData
        from smckit.tl._esmc2 import esmc2

        seq = np.array([0, 0, 1, 2, 2, 0, 2, 0, 2, 2], dtype=np.int8)
        data = SmcData(
            uns={
                "records": [{"codes": seq}],
                "sum_L": int((seq < 2).sum()),
                "sum_n": int((seq == 1).sum()),
            }
        )

        result = esmc2(
            data,
            n_states=4,
            n_iterations=1,
            estimate_rho=False,
            mu=1e-8,
            implementation="native",
        ).results["esmc2"]

        # Upstream theta_W = het_sites / (callable_sites / total_length) = 1 / (5/10) = 2.
        assert result["theta"] == pytest.approx(2.0, rel=1e-12, abs=1e-12)

    def test_native_reports_rho_in_public_units(self):
        """Public rho should match upstream's per-bp-per-2Ne convention."""
        from smckit._core import SmcData
        from smckit.tl._esmc2 import esmc2

        rng = np.random.default_rng(999)
        seq = rng.choice([0, 1], size=200, p=[0.99, 0.01]).astype(np.int8)
        data = SmcData(
            uns={
                "records": [{"codes": seq}],
                "sum_L": int((seq < 2).sum()),
                "sum_n": int((seq == 1).sum()),
            }
        )

        result = esmc2(
            data,
            n_states=4,
            n_iterations=1,
            estimate_rho=False,
            rho_over_theta=1.0,
            mu=1e-8,
            implementation="native",
        ).results["esmc2"]

        assert result["rho"] == pytest.approx(
            result["rho_per_sequence"] / (2 * len(seq)),
            rel=1e-12,
        )
        assert result["rho_per_sequence"] == pytest.approx(result["theta"], rel=1e-12)
