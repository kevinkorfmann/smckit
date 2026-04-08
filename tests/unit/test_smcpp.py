"""Tests for SMC++ implementation — lineage counting, SFS weights, CSFS, HMM, E2E."""

import json
import os
import subprocess
import numpy as np
import pytest

from smckit.tl._smcpp import (
    _accumulate_after_t_exact,
    _augmented_after_t_rate_matrix,
    _augmented_after_t_states,
    _backward_spans,
    _collect_expectation_stats,
    _compute_onepop_transition_and_pi,
    _compute_psmc_style_transition,
    _compute_onepop_raw_csfs_tensor,
    _emission_vector_onepop,
    _expand_onepop_model_pieces,
    _incorporate_theta_into_csfs,
    _forward_spans,
    _initial_after_t_distribution,
    _interpolate_occupation,
    _lineage_rate_matrix,
    _log_likelihood_spans,
    _m_step_objective,
    _precompute_cumulative_occupation,
    _preprocess_onepop_records,
    _propagate_lineages,
    _resolve_upstream_smcpp_python,
    _sfs_weights,
    _lift_csfs_to_two_distinguished,
    _watterson_theta,
    compute_csfs,
    compute_hmm_params,
    compute_time_intervals,
    encode_obs,
    observation_space_size,
)

# ---------------------------------------------------------------------------
# Time intervals
# ---------------------------------------------------------------------------

class TestTimeIntervals:
    def test_basic(self):
        t = compute_time_intervals(8, max_t=15.0, alpha=0.1)
        assert len(t) == 9  # K+1
        assert t[0] == 0.0
        assert t[8] == 15.0
        assert all(t[i] < t[i + 1] for i in range(8))

    def test_exponential_spacing(self):
        t = compute_time_intervals(10, max_t=15.0, alpha=0.1)
        # First interval should be smallest
        diffs = np.diff(t)
        assert diffs[0] < diffs[-1]

    def test_single_interval(self):
        t = compute_time_intervals(1, max_t=10.0)
        assert len(t) == 2
        assert t[0] == 0.0
        assert t[1] == 10.0


class TestObservationEncoding:
    def test_observation_space_size_surrogate(self):
        assert observation_space_size(5, n_distinguished=1) == 13

    def test_observation_space_size_upstream_onepop(self):
        assert observation_space_size(5, n_distinguished=2) == 19

    def test_encode_obs_surrogate_layout(self):
        assert encode_obs(0, 0, 5, n_distinguished=1) == 0
        assert encode_obs(0, 3, 5, n_distinguished=1) == 3
        assert encode_obs(1, 2, 5, n_distinguished=1) == 8
        assert encode_obs(-1, -1, 5, n_distinguished=1) == 12

    def test_encode_obs_upstream_layout(self):
        assert encode_obs(0, 0, 5, n_distinguished=2) == 0
        assert encode_obs(1, 0, 5, n_distinguished=2) == 6
        assert encode_obs(2, 5, 5, n_distinguished=2) == 17
        assert encode_obs(-1, -1, 5, n_distinguished=2) == 18

    def test_lift_csfs_to_two_distinguished_preserves_column_sums(self):
        e_one = np.zeros((13, 2))
        e_one[0, :] = [0.6, 0.5]
        e_one[1, :] = [0.1, 0.1]
        e_one[2, :] = [0.05, 0.1]
        e_one[6, :] = [0.15, 0.1]
        e_one[7, :] = [0.1, 0.1]
        e_one[8, :] = [0.0, 0.1]
        e_one[-1, :] = 1.0

        e_two = _lift_csfs_to_two_distinguished(e_one, n_undist=5)

        np.testing.assert_allclose(e_two[:-1, :].sum(axis=0), [1.0, 1.0])
        np.testing.assert_allclose(e_two[-1, :], [1.0, 1.0])
        np.testing.assert_allclose(e_two[encode_obs(1, 0, 5, 2), :], e_one[6, :])
        np.testing.assert_allclose(
            e_two[encode_obs(0, 1, 5, 2), :] + e_two[encode_obs(2, 4, 5, 2), :],
            e_one[1, :],
        )

    def test_incorporate_theta_uses_exact_exponential_transform(self):
        csfs = np.zeros((7, 1))
        csfs[1, 0] = 2.0
        csfs[3, 0] = 1.0

        e = _incorporate_theta_into_csfs(csfs, theta_rate=0.5)

        expected_variant_mass = -np.expm1(-0.5 * 3.0)
        np.testing.assert_allclose(e[1, 0], expected_variant_mass * (2.0 / 3.0))
        np.testing.assert_allclose(e[3, 0], expected_variant_mass * (1.0 / 3.0))
        np.testing.assert_allclose(e[0, 0], np.exp(-0.5 * 3.0))
        np.testing.assert_allclose(e[:-1, 0].sum(), 1.0)
        assert e[-1, 0] == 1.0


# ---------------------------------------------------------------------------
# Lineage counting death process
# ---------------------------------------------------------------------------

class TestLineageRateMatrix:
    def test_shape(self):
        Q = _lineage_rate_matrix(5)
        assert Q.shape == (5, 5)

    def test_rows_sum_to_zero(self):
        Q = _lineage_rate_matrix(10)
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_absorbing_state(self):
        Q = _lineage_rate_matrix(5)
        # j=1 (index 0) is absorbing: no transitions out
        assert Q[0, 0] == 0.0

    def test_coalescence_rates(self):
        Q = _lineage_rate_matrix(4)
        # j=2 (index 1): rate = 2*1/2 = 1
        assert Q[1, 1] == -1.0
        assert Q[1, 0] == 1.0
        # j=3 (index 2): rate = 3*2/2 = 3
        assert Q[2, 2] == -3.0
        assert Q[2, 1] == 3.0
        # j=4 (index 3): rate = 4*3/2 = 6
        assert Q[3, 3] == -6.0
        assert Q[3, 2] == 6.0


class TestPropagateLineages:
    def test_starts_at_n(self):
        Q = _lineage_rate_matrix(5)
        t = np.array([0.0, 1.0])
        eta = np.array([1.0])
        p = _propagate_lineages(Q, 5, t, eta)
        assert p.shape == (2, 5)
        # At time 0: all probability on j=5 (index 4)
        np.testing.assert_allclose(p[0], [0, 0, 0, 0, 1])

    def test_probability_sums_to_one(self):
        Q = _lineage_rate_matrix(10)
        t = compute_time_intervals(5, max_t=10.0)
        eta = np.ones(5)
        p = _propagate_lineages(Q, 10, t, eta)
        for k in range(6):
            np.testing.assert_allclose(p[k].sum(), 1.0, atol=1e-10)

    def test_lineages_decrease(self):
        Q = _lineage_rate_matrix(10)
        t = compute_time_intervals(5, max_t=10.0)
        eta = np.ones(5)
        p = _propagate_lineages(Q, 10, t, eta)
        j_vec = np.arange(1, 11, dtype=np.float64)
        expected_A = [float(j_vec @ p[k]) for k in range(6)]
        # Expected number of lineages should decrease
        for i in range(5):
            assert expected_A[i] >= expected_A[i + 1]

    def test_large_eta_slow_coalescence(self):
        Q = _lineage_rate_matrix(5)
        t = np.array([0.0, 0.01])
        # Very large eta = slow coalescence
        eta = np.array([1000.0])
        p = _propagate_lineages(Q, 5, t, eta)
        # Most probability should stay at j=5
        assert p[1, 4] > 0.99

    def test_interpolate_occupation_matches_boundary_values(self):
        Q = _lineage_rate_matrix(4)
        t = compute_time_intervals(4, max_t=6.0)
        eta = np.ones(4)
        p_bound = _propagate_lineages(Q, 4, t, eta)
        R = _precompute_cumulative_occupation(Q, 4, t, eta, p_bound)

        np.testing.assert_allclose(
            _interpolate_occupation(R, t, t[0], Q, eta, p_bound, 4),
            R[0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _interpolate_occupation(R, t, t[2], Q, eta, p_bound, 4),
            R[2],
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# SFS weights
# ---------------------------------------------------------------------------

class TestSfsWeights:
    def test_shape(self):
        w = _sfs_weights(5)
        assert w.shape == (5, 5)

    def test_rows_sum_to_one(self):
        w = _sfs_weights(10)
        for j in range(10):
            row_sum = w[j].sum()
            if row_sum > 0:
                np.testing.assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_j_equals_n_all_singletons(self):
        n = 8
        w = _sfs_weights(n)
        # j=n (index n-1): all singletons, so b=1 with prob 1
        np.testing.assert_allclose(w[n - 1, 0], 1.0)
        assert w[n - 1, 1:].sum() < 1e-10

    def test_j_equals_1_all_n(self):
        n = 6
        w = _sfs_weights(n)
        # j=1 (index 0): single lineage subtends all n
        np.testing.assert_allclose(w[0, n - 1], 1.0)

    def test_j_equals_2_uniform(self):
        n = 5
        w = _sfs_weights(n)
        # j=2: P(b) = 1/(n-1) for b = 1..n-1
        expected = 1.0 / (n - 1)
        for b in range(1, n):
            np.testing.assert_allclose(w[1, b - 1], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

class TestEncodeObs:
    def test_missing(self):
        assert encode_obs(-1, -1, 10) == 22

    def test_monomorphic(self):
        assert encode_obs(0, 0, 10) == 0

    def test_a0_b_positive(self):
        # (a=0, b=3) with n_undist=10 → obs = 3
        assert encode_obs(0, 3, 10) == 3

    def test_a1_b0(self):
        # (a=1, b=0) with n_undist=10 → obs = 11
        assert encode_obs(1, 0, 10) == 11

    def test_a1_b_positive(self):
        # (a=1, b=5) with n_undist=10 → obs = 16
        assert encode_obs(1, 5, 10) == 16


# ---------------------------------------------------------------------------
# CSFS (emission matrix)
# ---------------------------------------------------------------------------

class TestCSFS:
    @pytest.fixture
    def csfs_params(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        _, _, avg_t, E_A = _compute_psmc_style_transition(n_undist, K, t, eta, 0.0005)
        return n_undist, K, t, eta, avg_t, E_A

    def test_shape(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        n_obs = 2 * (n_undist + 1) + 1
        assert e.shape == (n_obs, K)

    def test_columns_sum_to_one(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        for k in range(K):
            col_sum = e[:-1, k].sum()
            np.testing.assert_allclose(col_sum, 1.0, atol=1e-6)

    def test_monomorphic_dominates(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        for k in range(K):
            assert e[0, k] > 0.5

    def test_missing_is_one(self):
        K = 4
        n_undist = 3
        t = compute_time_intervals(K)
        eta = np.ones(K)
        _, _, avg_t, E_A = _compute_psmc_style_transition(n_undist, K, t, eta, 0.0005)
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        n_obs = 2 * (n_undist + 1) + 1
        for k in range(K):
            assert e[n_obs - 1, k] == 1.0

    def test_trunk_weight_increases_with_time(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        trunk_early = e[n_undist + 1, 0]
        trunk_late = e[n_undist + 1, K - 1]
        assert trunk_late > trunk_early

    def test_all_derived_row_is_zero(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(n_undist, K, t, eta, theta=0.001, avg_t=avg_t, E_A_mid=E_A)
        all_derived_idx = 2 * n_undist + 1
        np.testing.assert_allclose(e[all_derived_idx], 0.0, atol=1e-12)

    def test_two_distinguished_shape_and_normalization(self, csfs_params):
        n_undist, K, t, eta, avg_t, E_A = csfs_params
        e = compute_csfs(
            n_undist,
            K,
            t,
            eta,
            theta=0.001,
            avg_t=avg_t,
            E_A_mid=E_A,
            n_distinguished=2,
        )
        assert e.shape == (observation_space_size(n_undist, 2), K)
        np.testing.assert_allclose(e[:-1].sum(axis=0), 1.0, atol=1e-6)

    @pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    )
    def test_two_distinguished_raw_csfs_matches_upstream(self):
        t = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
        eta = np.array([1.0, 2.0, 1.5], dtype=np.float64)
        n_undist = 5

        native = _compute_onepop_raw_csfs_tensor(n_undist, t, eta)

        script = """
import json
import numpy as np
from smcpp.model import PiecewiseModel
from smcpp import _smcpp
t=np.array([0.0,0.5,1.5,3.0], dtype=float)
eta=np.array([1.0,2.0,1.5], dtype=float)
model=PiecewiseModel(eta, np.diff(t), 0.5, None)
out=[]
for k in range(len(eta)):
    out.append(np.asarray(_smcpp.raw_sfs(model, 5, float(t[k]), float(t[k+1])), dtype=float).tolist())
print(json.dumps(out))
"""
        proc = subprocess.run(
            [_resolve_upstream_smcpp_python(), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        upstream = np.asarray(json.loads(proc.stdout), dtype=np.float64)

        np.testing.assert_allclose(native, upstream, rtol=1e-10, atol=1e-10)


class TestExactAfterT:
    def test_augmented_states_for_two_undistinguished(self):
        states, _ = _augmented_after_t_states(2)
        assert states == [(1, 3), (2, 2)]

    def test_augmented_rate_matrix_rows_sum_to_zero(self):
        states, state_index = _augmented_after_t_states(4)
        Q = _augmented_after_t_rate_matrix(4, states, state_index)
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)

    def test_initial_after_t_distribution_splits_partner_size_exactly(self):
        n_undist = 3
        states, state_index = _augmented_after_t_states(n_undist)
        w = _sfs_weights(n_undist)
        p_at_T = np.array([0.0, 1.0, 0.0])  # exactly j=2 lineages at T

        p_init = _initial_after_t_distribution(
            p_at_T, n_undist, states, state_index, w
        )

        assert np.isclose(p_init[state_index[(2, 2)]], 0.5)
        assert np.isclose(p_init[state_index[(2, 3)]], 0.5)
        np.testing.assert_allclose(p_init.sum(), 1.0)

    def test_accumulate_after_t_exact_matches_no_coalescence_limit(self):
        n_undist = 2
        n_total = n_undist + 1
        n_obs = 2 * (n_undist + 1) + 1
        states, state_index = _augmented_after_t_states(n_undist)
        xi_after = np.zeros(n_obs, dtype=np.float64)
        occ = np.zeros(len(states), dtype=np.float64)
        occ[state_index[(2, 2)]] = 1.0

        _accumulate_after_t_exact(xi_after, occ, states, n_undist, n_total)

        assert np.isclose(xi_after[n_undist + 1 + 1], 0.5)  # (a=1, b=1)
        assert np.isclose(xi_after[1], 0.5)  # (a=0, b=1)
        assert np.isclose(xi_after.sum(), 1.0)


class TestOnePopPreprocessing:
    def test_preprocessing_preserves_sample_size_for_legacy_triplets(self):
        records = [{
            "name": "synthetic",
            "n_undist": 5,
            "n_distinguished": 2,
            "observations": [
                (1500, 0, 0),
                (1, 1, 2),
                (900, 0, 0),
                (1, 0, 3),
            ],
        }]

        processed = _preprocess_onepop_records(records, n_undist=5)

        assert processed
        assert all(len(obs) == 4 for obs in processed[0]["observations"])
        assert any(obs[3] == 5 for obs in processed[0]["observations"])

    def test_reduced_onepop_emissions_use_observation_scale(self):
        t = np.array([0.0, 0.1, 0.3], dtype=np.float64)
        eta = np.array([0.2, 0.25], dtype=np.float64)
        hidden_states = np.array([0.0, 0.1, 0.2, 0.3, np.inf], dtype=np.float64)

        hp_unscaled = compute_hmm_params(
            eta,
            5,
            t,
            theta=1e-4,
            rho_base=8e-05,
            n_distinguished=2,
            hidden_states=hidden_states,
            observation_scale=1.0,
        )
        hp_scaled = compute_hmm_params(
            eta,
            5,
            t,
            theta=1e-4,
            rho_base=8e-05,
            n_distinguished=2,
            hidden_states=hidden_states,
            observation_scale=100.0,
        )

        reduced_unscaled = _emission_vector_onepop(hp_unscaled, 0, 0, 0)
        reduced_scaled = _emission_vector_onepop(hp_scaled, 0, 0, 0)

        assert np.all(reduced_scaled < reduced_unscaled)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

class TestTransition:
    def test_shape(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        a, _, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        assert a.shape == (K, K)

    def test_rows_sum_to_one(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        a, _, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        for k in range(K):
            np.testing.assert_allclose(a[k].sum(), 1.0, atol=1e-10)

    def test_diagonal_dominates(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        a, _, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        for k in range(K):
            assert a[k, k] > 0.5

    def test_no_recombination(self):
        K = 4
        n_undist = 3
        t = compute_time_intervals(K)
        eta = np.ones(K)
        a, _, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.0)
        np.testing.assert_allclose(a, np.eye(K), atol=1e-10)


class TestPsmcStyleTransition:
    def test_sigma_sums_to_one(self):
        K = 8
        n_undist = 10
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        _, sigma, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-6)

    def test_sigma_all_positive(self):
        K = 8
        n_undist = 10
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        _, sigma, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        assert all(s >= 0 for s in sigma)

    def test_large_sample_early_coalescence(self):
        K = 8
        n_undist = 100
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        _, sigma, _, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        assert sigma[0] > sigma[K - 1]

    def test_avg_t_within_intervals(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        _, _, avg_t, _ = _compute_psmc_style_transition(n_undist, K, t, eta, 0.001)
        for k in range(K):
            assert t[k] <= avg_t[k] <= t[k + 1]

    def test_two_distinguished_hmm_uses_hidden_state_count(self):
        n_undist = 5
        t = np.array([0.0, 0.15, 0.55, 1.25], dtype=np.float64)
        eta = np.array([1.0, 1.2, 0.9], dtype=np.float64)
        hidden_states = np.array([0.0, 0.08, 0.18, 0.30, 0.46, 0.68, 1.05, np.inf], dtype=np.float64)

        hp = compute_hmm_params(
            eta,
            n_undist,
            t,
            theta=0.001,
            rho_base=0.0005,
            n_distinguished=2,
            hidden_states=hidden_states,
        )

        assert hp.a.shape == (7, 7)
        assert hp.e.shape == (observation_space_size(n_undist, 2), 7)
        np.testing.assert_allclose(hp.a0.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(hp.a.sum(axis=1), 1.0, atol=1e-8)

    @pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    )
    def test_two_distinguished_transition_matches_upstream(self):
        t = np.array([0.0, 0.08381274920658673, 0.3042668234667235, 0.6811355526388657], dtype=np.float64)
        eta = np.array([1766.54216941, 1741.82971868, 1734.69034739], dtype=np.float64) / 8000.0
        hidden_states = np.array(
            [
                0.0,
                0.08381274920658673,
                0.18294219146177287,
                0.3042668234667235,
                0.46068147837872525,
                0.6811355526388657,
                1.0580042818110085,
                np.inf,
            ],
            dtype=np.float64,
        )

        t_internal, eta_internal = _expand_onepop_model_pieces(t, eta)
        native_t, native_pi = _compute_onepop_transition_and_pi(
            t_internal,
            eta_internal,
            hidden_states,
            8e-05,
        )

        script = """
import json
import numpy as np
from smcpp import _smcpp, model, spline
obs = [np.array([[1, 0, 0, 5]], dtype=np.int32)]
t = np.array([0.0, 0.08381274920658673, 0.3042668234667235, 0.6811355526388657], dtype=float)
eta = np.array([1766.54216941, 1741.82971868, 1734.69034739], dtype=float) / 8000.0
hs = np.array([0.0, 0.08381274920658673, 0.18294219146177287, 0.3042668234667235, 0.46068147837872525, 0.6811355526388657, 1.0580042818110085, np.inf], dtype=float)
im = _smcpp.PyOnePopInferenceManager(5, obs, hs, ('pop1',), 0.0)
m = model.SMCModel(t[1:], 4000.0, spline.Piecewise, 'pop1')
m[:] = np.log(eta)
im.model = m
im.theta = 1e-4
im.rho = 8e-05
im.alpha = 1
im.E_step(True)
print(json.dumps({
    "pi": np.asarray(im.pi).astype(float).reshape(-1).tolist(),
    "transition": np.asarray(im.transition).astype(float).tolist(),
}))
"""
        proc = subprocess.run(
            [_resolve_upstream_smcpp_python(), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        upstream = json.loads(proc.stdout)

        np.testing.assert_allclose(native_pi, np.asarray(upstream["pi"]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            native_t,
            np.asarray(upstream["transition"]),
            rtol=1e-7,
            atol=2e-7,
        )

    @pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    )
    def test_two_distinguished_expectation_stats_match_upstream(self):
        t = np.array([0.0, 0.08381274920658673, 0.3042668234667235, 0.6811355526388657], dtype=np.float64)
        eta = np.array([1766.54216941, 1741.82971868, 1734.69034739], dtype=np.float64) / 8000.0
        hidden_states = np.array(
            [
                0.0,
                0.08381274920658673,
                0.18294219146177287,
                0.3042668234667235,
                0.46068147837872525,
                0.6811355526388657,
                1.0580042818110085,
                np.inf,
            ],
            dtype=np.float64,
        )
        observations = [
            (4999, 0, 0),
            (1, 0, 0),
            (4999, 0, 0),
            (1, 1, 2),
            (4999, 0, 0),
            (1, 2, 4),
        ]
        record = {"name": "synthetic2", "observations": observations, "n_undist": 5, "n_distinguished": 2}

        hp = compute_hmm_params(
            eta,
            5,
            t,
            theta=1e-4,
            rho_base=8e-05,
            n_distinguished=2,
            hidden_states=hidden_states,
        )
        native = _collect_expectation_stats(hp, [record])
        native_q = _m_step_objective(
            np.log(eta),
            native,
            5,
            t,
            1e-4,
            8e-05,
            10.0,
            n_distinguished=2,
            hidden_states=hidden_states,
        )

        script = """
import json
import numpy as np
from smcpp import _smcpp, model, spline
obs = [np.array([[4999,0,0,5],[1,0,0,5],[4999,0,0,5],[1,1,2,5],[4999,0,0,5],[1,2,4,5]], dtype=np.int32)]
hs = np.array([0.0,0.08381274920658673,0.18294219146177287,0.3042668234667235,0.46068147837872525,0.6811355526388657,1.0580042818110085,np.inf], dtype=float)
t = np.array([0.0,0.08381274920658673,0.3042668234667235,0.6811355526388657], dtype=float)
eta = np.array([1766.54216941,1741.82971868,1734.69034739], dtype=float) / 8000.0
im = _smcpp.PyOnePopInferenceManager(5, obs, hs, ('pop1',), 0.0)
m = model.SMCModel(t[1:], 4000.0, spline.Piecewise, 'pop1')
m[:] = np.log(eta)
im.model = m
im.theta = 1e-4
im.rho = 8e-05
im.alpha = 1
im.E_step(True)
print(json.dumps({
    "gamma0": np.asarray(im.gammas[0][:,0], dtype=float).tolist(),
    "gamma_sums": {str(key): np.asarray(val, dtype=float).tolist() for key, val in im.gamma_sums[0].items()},
    "xisum": np.asarray(im.xisums[0], dtype=float).tolist(),
    "q": float(-im.Q() + 10.0 * m.regularizer()),
}))
"""
        proc = subprocess.run(
            [_resolve_upstream_smcpp_python(), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        upstream = json.loads(proc.stdout)

        np.testing.assert_allclose(native.gamma0, np.asarray(upstream["gamma0"]), rtol=0.04, atol=1e-6)

        for key, vec in upstream["gamma_sums"].items():
            a_obs, b_obs, _ = eval(key)
            upstream_vec = np.asarray(vec, dtype=np.float64)
            native_vec = native.gamma_sums.get((a_obs, b_obs, 5))
            if native_vec is None:
                obs_idx = encode_obs(a_obs, b_obs, 5, 2)
                native_vec = native.gamma_sums[obs_idx]
            np.testing.assert_allclose(native_vec.sum(), upstream_vec.sum(), rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(native_vec, upstream_vec, rtol=0.16, atol=1e-6)

        upstream_xisum = np.asarray(upstream["xisum"], dtype=np.float64)
        np.testing.assert_allclose(native.xisum.sum(), upstream_xisum.sum(), rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(native.xisum, upstream_xisum, rtol=0.19, atol=1e-6)
        np.testing.assert_allclose(native_q, upstream["q"], rtol=0.01, atol=0.25)


# ---------------------------------------------------------------------------
# Forward/backward with spans
# ---------------------------------------------------------------------------

class TestForwardSpans:
    @pytest.fixture
    def simple_hmm(self):
        K = 4
        n_undist = 3
        t = compute_time_intervals(K, max_t=10.0)
        eta = np.ones(K)
        hp = compute_hmm_params(eta, n_undist, t, theta=0.001, rho_base=0.0005)
        return hp

    def test_returns_correct_length(self, simple_hmm):
        obs = [(100, 0, 0), (1, 0, 1), (50, 0, 0), (1, 1, 0), (200, 0, 0), (1, 0, 2)]
        f_list, s_list = _forward_spans(simple_hmm, obs)
        assert len(f_list) == len(obs)
        assert len(s_list) == len(obs)

    def test_forward_probabilities_sum_to_one(self, simple_hmm):
        obs = [(100, 0, 0), (1, 0, 1), (50, 0, 0), (1, 1, 0)]
        f_list, _ = _forward_spans(simple_hmm, obs)
        for f in f_list:
            np.testing.assert_allclose(f.sum(), 1.0, atol=1e-6)

    def test_log_likelihood_finite(self, simple_hmm):
        obs = [(100, 0, 0), (1, 0, 1), (200, 0, 0), (1, 1, 0), (150, 0, 0), (1, 0, 2)]
        _, s_list = _forward_spans(simple_hmm, obs)
        ll = _log_likelihood_spans(s_list)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative

    def test_forward_matches_expanded_sequence(self, simple_hmm):
        obs = [(7, 0, 0), (1, 0, 2), (5, 0, 0), (1, 1, 0)]
        f_list, s_list = _forward_spans(simple_hmm, obs)
        ll_span = _log_likelihood_spans(s_list)

        f = simple_hmm.a0.copy()
        ll_full = 0.0
        for span, a_obs, b_obs in obs:
            obs_idx = encode_obs(a_obs, b_obs, simple_hmm.n_undist)
            e_obs = simple_hmm.e[obs_idx]
            for _ in range(span):
                f = e_obs * (simple_hmm.a.T @ f)
                s = f.sum()
                ll_full += np.log(max(s, 1e-300))
                f /= max(s, 1e-300)

        np.testing.assert_allclose(ll_span, ll_full, atol=1e-10)
        np.testing.assert_allclose(f_list[-1], f, atol=1e-10)


class TestBackwardSpans:
    @pytest.fixture
    def simple_hmm(self):
        K = 4
        n_undist = 3
        t = compute_time_intervals(K, max_t=10.0)
        eta = np.ones(K)
        hp = compute_hmm_params(eta, n_undist, t, theta=0.001, rho_base=0.0005)
        return hp

    def test_returns_correct_length(self, simple_hmm):
        obs = [(100, 0, 0), (1, 0, 1), (50, 0, 0), (1, 1, 0)]
        _, s_list = _forward_spans(simple_hmm, obs)
        b_list = _backward_spans(simple_hmm, obs, s_list)
        assert len(b_list) == len(obs)


# ---------------------------------------------------------------------------
# Full HMM params
# ---------------------------------------------------------------------------

class TestComputeHmmParams:
    def test_creates_valid_params(self):
        K = 8
        n_undist = 5
        t = compute_time_intervals(K, max_t=15.0)
        eta = np.ones(K)
        hp = compute_hmm_params(eta, n_undist, t, theta=0.001, rho_base=0.0005)

        assert hp.a.shape == (K, K)
        assert hp.a0.shape == (K,)
        assert hp.n_undist == n_undist

        # Transition rows sum to 1
        for k in range(K):
            np.testing.assert_allclose(hp.a[k].sum(), 1.0, atol=1e-10)

        # Initial distribution sums to 1
        np.testing.assert_allclose(hp.a0.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: smcpp()
# ---------------------------------------------------------------------------

class TestSmcppEndToEnd:
    def test_synthetic_constant_population(self):
        """Run SMC++ on synthetic data from a constant-size population."""
        from smckit._core import SmcData
        from smckit.tl._smcpp import smcpp

        n_undist = 5
        rng = np.random.default_rng(42)

        # Generate synthetic observations: mostly monomorphic with rare variants
        observations = []
        for _ in range(200):
            span = rng.geometric(0.001)  # ~1000 bp monomorphic runs
            observations.append((int(span), 0, 0))
            a = rng.integers(0, 2)
            if a == 0:
                b = rng.integers(1, n_undist + 1)
            else:
                b = rng.integers(0, n_undist + 1)
            observations.append((1, int(a), int(b)))

        data = SmcData(
            uns={
                "records": [{"name": "synthetic", "observations": observations, "n_distinguished": 1}],
                "n_undist": n_undist,
                "n_distinguished": 1,
            },
        )

        result = smcpp(
            data,
            n_intervals=8,
            max_t=10.0,
            max_iterations=5,
            regularization=10.0,
            seed=42,
            backend="native",
        )

        assert "smcpp" in result.results
        r = result.results["smcpp"]
        assert r["ne"].ndim == 1
        assert r["ne"].shape == (r["n_intervals"],)
        assert all(r["ne"] > 0)
        assert np.isfinite(r["log_likelihood"])
        assert r["n_undist"] == n_undist
        assert r["n_distinguished"] == 1

    def test_missing_observations_do_not_count_toward_theta(self):
        from smckit._core import SmcData
        from smckit.tl._smcpp import smcpp

        data = SmcData(
            uns={
                "records": [{
                    "name": "synthetic",
                    "observations": [
                        (10, 0, 0),
                        (5, -1, -1),
                        (1, 0, 1),
                    ],
                }],
                "n_undist": 5,
                "n_distinguished": 1,
            },
        )

        result = smcpp(
            data,
            n_intervals=4,
            max_t=5.0,
            max_iterations=1,
            regularization=1.0,
            seed=42,
            backend="native",
        )

        expected_theta = 1.0 / (10 * (np.log(6) + 0.5 / 6 + 0.57721) + (np.log(6) + 0.5 / 6 + 0.57721))
        assert np.isclose(result.results["smcpp"]["theta"], expected_theta)

    def test_watterson_theta_matches_upstream_rowwise_formula(self):
        records = [{
            "n_undist": 5,
            "observations": [
                (10, 0, 0),
                (3, -1, -1),
                (2, 1, 0),
                (4, 0, 2),
            ],
        }]

        theta = _watterson_theta(records)
        h = np.log(6) + 0.5 / 6 + 0.57721
        expected = (2 + 4) / ((10 + 2 + 4) * h)
        assert np.isclose(theta, expected)

    @pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    )
    def test_upstream_backend_runs_on_synthetic_contig(self):
        from smckit._core import SmcData
        from smckit.tl._smcpp import smcpp

        observations = []
        for i in range(120):
            observations.append((9999, 0, 0))
            observations.append((1, i % 2, (i % 5) + 1))

        data = SmcData(
            uns={
                "records": [{"name": "synthetic", "observations": observations}],
                "n_undist": 5,
            },
        )

        result = smcpp(
            data,
            n_intervals=4,
            max_iterations=1,
            regularization=10.0,
            seed=42,
            backend="upstream",
        )

        r = result.results["smcpp"]
        assert r["backend"] == "upstream"
        assert r["ne"].shape == (4,)
        assert np.all(r["ne"] > 0)
        assert np.isfinite(r["log_likelihood"])
        assert "model" in r["upstream"]

    @pytest.mark.skipif(
        _resolve_upstream_smcpp_python() is None,
        reason="Upstream SMC++ side environment not available",
    )
    def test_auto_backend_prefers_upstream_when_available(self):
        from smckit._core import SmcData
        from smckit.tl._smcpp import smcpp

        observations = []
        for i in range(120):
            observations.append((9999, 0, 0))
            observations.append((1, i % 2, (i % 5) + 1))

        data = SmcData(
            uns={
                "records": [{"name": "synthetic", "observations": observations}],
                "n_undist": 5,
            },
        )

        result = smcpp(
            data,
            n_intervals=4,
            max_iterations=1,
            regularization=10.0,
            seed=42,
        )

        assert result.results["smcpp"]["backend"] == "upstream"
