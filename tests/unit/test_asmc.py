"""Unit tests for ASMC reimplementation."""

from __future__ import annotations

import gzip
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from smckit._core import SmcData
from smckit.io._asmc import DecodingQuantities, read_decoding_quantities
from smckit.tl._asmc import (
    AsmcResult,
    PairObservations,
    asmc,
    backward,
    compute_posteriors,
    encode_pair,
    forward,
    posterior_map_tmrca,
    posterior_mean_tmrca,
    prepare_emissions,
    round_morgans,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dq() -> DecodingQuantities:
    """Build a small DecodingQuantities with 4 states for testing."""
    states = 4
    dq = DecodingQuantities()
    dq.states = states
    dq.csfs_samples = 0
    dq.initial_state_prob = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    dq.expected_times = np.array([10.0, 50.0, 200.0, 1000.0], dtype=np.float32)
    dq.discretization = np.array([0.0, 20.0, 100.0, 500.0, 5000.0], dtype=np.float32)
    dq.time_vector = np.array([10.0, 50.0, 200.0, 1000.0], dtype=np.float32)
    dq.column_ratios = np.array([0.1, 0.2, 0.15, 0.0], dtype=np.float32)

    # Simple transition vectors for one distance key
    dist = round_morgans(1e-4)
    B = np.array([0.01, 0.02, 0.01, 0.0], dtype=np.float32)
    U = np.array([0.01, 0.02, 0.015, 0.0], dtype=np.float32)
    D = np.array([0.95, 0.90, 0.85, 0.80], dtype=np.float32)
    RR = np.array([0.1, 0.15, 0.12, 0.0], dtype=np.float32)

    dq.B_vectors = {dist: B}
    dq.U_vectors = {dist: U}
    dq.D_vectors = {dist: D}
    dq.row_ratio_vectors = {dist: RR}

    dq.classic_emission = np.array([
        [0.95, 0.80, 0.60, 0.40],  # obs=0 (homozygous)
        [0.05, 0.20, 0.40, 0.60],  # obs=1 (heterozygous)
    ], dtype=np.float32)
    dq.compressed_emission = dq.classic_emission.copy()

    return dq


@pytest.fixture
def small_haplotypes() -> np.ndarray:
    """4 haplotypes, 10 sites."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(4, 10), dtype=np.uint8)


@pytest.fixture
def small_genetic_positions() -> np.ndarray:
    """10 sites with uniform spacing."""
    return np.linspace(0.0, 0.001, 10, dtype=np.float32)


# ---------------------------------------------------------------------------
# round_morgans
# ---------------------------------------------------------------------------

class TestRoundMorgans:
    def test_small_value_returns_min(self):
        assert round_morgans(1e-15) == 1e-10

    def test_zero_returns_min(self):
        assert round_morgans(0.0) == 1e-10

    def test_negative_returns_min(self):
        assert round_morgans(-0.01) == 1e-10

    def test_precision_0(self):
        result = round_morgans(0.77426, precision=0)
        assert abs(result - 0.8) < 1e-6

    def test_precision_1(self):
        result = round_morgans(0.77426, precision=1)
        assert abs(result - 0.77) < 1e-6

    def test_value_0_1_unchanged(self):
        result = round_morgans(0.1, precision=2)
        assert abs(result - 0.1) < 1e-6

    def test_deterministic(self):
        v = 0.00345
        assert round_morgans(v) == round_morgans(v)


# ---------------------------------------------------------------------------
# Pair encoding
# ---------------------------------------------------------------------------

class TestEncodePair:
    def test_shape(self, small_haplotypes):
        obs = encode_pair(small_haplotypes, 0, 1)
        assert obs.obs_is_zero.shape == (10,)
        assert obs.obs_is_two.shape == (10,)

    def test_indices_stored(self, small_haplotypes):
        obs = encode_pair(small_haplotypes, 2, 3)
        assert obs.hap_i == 2
        assert obs.hap_j == 3

    def test_obs_values_binary(self, small_haplotypes):
        obs = encode_pair(small_haplotypes, 0, 1)
        assert set(np.unique(obs.obs_is_zero)).issubset({0.0, 1.0})
        assert set(np.unique(obs.obs_is_two)).issubset({0.0, 1.0})

    def test_identical_haplotypes(self):
        haps = np.zeros((2, 5), dtype=np.uint8)
        obs = encode_pair(haps, 0, 1)
        np.testing.assert_array_equal(obs.obs_is_zero, np.ones(5))
        np.testing.assert_array_equal(obs.obs_is_two, np.zeros(5))

    def test_both_derived(self):
        haps = np.ones((2, 5), dtype=np.uint8)
        obs = encode_pair(haps, 0, 1)
        # Both carry derived: XOR=0 -> obs_is_zero=1, AND=1 -> obs_is_two=1
        # (obs_is_zero means "not heterozygous", includes both-derived)
        np.testing.assert_array_equal(obs.obs_is_zero, np.ones(5))
        np.testing.assert_array_equal(obs.obs_is_two, np.ones(5))

    def test_heterozygous(self):
        haps = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        obs = encode_pair(haps, 0, 1)
        # All sites differ: XOR=1 -> obs_is_zero=0, obs_is_two=0
        np.testing.assert_array_equal(obs.obs_is_zero, np.zeros(3))
        np.testing.assert_array_equal(obs.obs_is_two, np.zeros(3))

    def test_obs_is_two_implies_obs_is_zero(self, small_haplotypes):
        """obs_is_two=1 implies obs_is_zero=1 (both are set when both derived)."""
        obs = encode_pair(small_haplotypes, 0, 1)
        # Whenever obs_is_two=1, obs_is_zero must also be 1
        assert np.all(obs.obs_is_zero[obs.obs_is_two == 1] == 1)


# ---------------------------------------------------------------------------
# Emission preparation
# ---------------------------------------------------------------------------

class TestPrepareEmissions:
    def test_shapes(self, small_dq, small_genetic_positions):
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, 10,
            use_csfs=False,
        )
        assert e1.shape == (10, 4)
        assert e0m1.shape == (10, 4)
        assert e2m0.shape == (10, 4)

    def test_non_csfs_emission2_is_zero(self, small_dq, small_genetic_positions):
        """When not using CSFS, emission2 - emission0 = 0."""
        _, _, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, 10,
            use_csfs=False,
        )
        np.testing.assert_array_equal(e2m0, 0.0)

    def test_emission_consistency(self, small_dq, small_genetic_positions):
        """e1 + e0m1 should equal the classic emission[0] (homozygous)."""
        e1, e0m1, _ = prepare_emissions(
            small_dq, small_genetic_positions, 10,
            use_csfs=False, decoding_sequence=True,
        )
        np.testing.assert_allclose(
            e1[0] + e0m1[0],
            small_dq.classic_emission[0],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert alpha.shape == (n_sites, 4)

    def test_non_negative(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert np.all(alpha >= 0)

    def test_finite(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert np.all(np.isfinite(alpha))

    def test_initial_position_uses_prior(self, small_dq, small_genetic_positions):
        """Alpha at position 0 should be proportional to prior * emission."""
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        # alpha[0] should be normalized (sums to 1 after scaling)
        np.testing.assert_allclose(alpha[0].sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward algorithm
# ---------------------------------------------------------------------------

class TestBackward:
    def test_output_shape(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert beta.shape == (n_sites, 4)

    def test_non_negative(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert np.all(beta >= 0)

    def test_finite(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        assert np.all(np.isfinite(beta))

    def test_last_position_uniform(self, small_dq, small_genetic_positions):
        """Beta at last position should be uniform (scaled)."""
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        # All states should have the same beta at last position
        np.testing.assert_allclose(
            beta[-1], beta[-1, 0] * np.ones(4), atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Posterior computation
# ---------------------------------------------------------------------------

class TestPosteriors:
    def test_normalized(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        posteriors = compute_posteriors(alpha, beta)

        # Each row should sum to 1
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_non_negative(self, small_dq, small_genetic_positions):
        n_sites = 10
        e1, e0m1, e2m0 = prepare_emissions(
            small_dq, small_genetic_positions, n_sites, use_csfs=False,
        )
        obs_zero = np.ones(n_sites, dtype=np.float32)
        obs_two = np.zeros(n_sites, dtype=np.float32)

        alpha = forward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        beta = backward(
            small_dq, e1, e0m1, e2m0, obs_zero, obs_two,
            small_genetic_positions,
        )
        posteriors = compute_posteriors(alpha, beta)
        assert np.all(posteriors >= 0)


class TestPosteriorMean:
    def test_bounded(self, small_dq):
        posteriors = np.array([[0.5, 0.3, 0.15, 0.05]], dtype=np.float32)
        means = posterior_mean_tmrca(posteriors, small_dq.expected_times)
        assert means[0] >= small_dq.expected_times.min()
        assert means[0] <= small_dq.expected_times.max()

    def test_shape(self, small_dq):
        posteriors = np.ones((5, 4), dtype=np.float32) / 4
        means = posterior_mean_tmrca(posteriors, small_dq.expected_times)
        assert means.shape == (5,)


class TestPosteriorMAP:
    def test_returns_state_index(self, small_dq):
        posteriors = np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)
        maps = posterior_map_tmrca(posteriors, small_dq.expected_times, small_dq.initial_state_prob)
        # Upstream perPairMAP stores the plain posterior argmax state index.
        assert maps[0] == 2
        assert np.issubdtype(maps.dtype, np.integer)

    def test_shape(self, small_dq):
        posteriors = np.ones((5, 4), dtype=np.float32) / 4
        maps = posterior_map_tmrca(posteriors, small_dq.expected_times, small_dq.initial_state_prob)
        assert maps.shape == (5,)


# ---------------------------------------------------------------------------
# Decoding quantities I/O
# ---------------------------------------------------------------------------

class TestDecodingQuantitiesIO:
    def test_roundtrip_minimal(self, tmp_path):
        """Write a minimal decoding quantities file and read it back."""
        dq_path = tmp_path / "test.decodingQuantities.gz"

        content = """\
TransitionType
SMC
States
3
CSFSSamples
5
TimeVector
10.0 50.0 200.0
ExpectedTimes
15.0 75.0 350.0
Discretization
0.0 30.0 100.0 500.0
ClassicEmission
0.95 0.80 0.60
0.05 0.20 0.40
CompressedAscertainedEmission
0.94 0.79 0.59
0.06 0.21 0.41
InitialStateProb
0.5 0.3 0.2
ColumnRatios
0.1 0.2 0.15
Bvectors
0.0001 0.01 0.02 0.01
Uvectors
0.0001 0.01 0.02 0.015
Dvectors
0.0001 0.95 0.90 0.85
RowRatios
0.0001 0.1 0.15 0.12
"""
        with gzip.open(dq_path, "wt") as f:
            f.write(content)

        dq = read_decoding_quantities(dq_path)

        assert dq.states == 3
        assert dq.csfs_samples == 5
        np.testing.assert_allclose(dq.initial_state_prob, [0.5, 0.3, 0.2], atol=1e-5)
        np.testing.assert_allclose(dq.expected_times, [15.0, 75.0, 350.0], atol=1e-3)
        assert dq.discretization.shape == (4,)
        assert dq.classic_emission.shape == (2, 3)
        assert dq.compressed_emission.shape == (2, 3)
        assert len(dq.D_vectors) == 1
        assert len(dq.B_vectors) == 1
        assert len(dq.U_vectors) == 1
        assert len(dq.row_ratio_vectors) == 1


# ---------------------------------------------------------------------------
# End-to-end ASMC
# ---------------------------------------------------------------------------

class TestAsmcEndToEnd:
    def test_basic_run(self, small_dq, small_haplotypes, small_genetic_positions):
        """Run ASMC on synthetic data and check result structure."""
        n_sites = small_haplotypes.shape[1]

        data = SmcData()
        data.sequences = small_haplotypes
        data.uns["haplotypes"] = small_haplotypes
        data.uns["genetic_positions"] = small_genetic_positions
        data.uns["physical_positions"] = np.arange(n_sites, dtype=np.int32) * 1000
        data.uns["decoding_quantities"] = small_dq

        result = asmc(data, pairs=[(0, 1)])

        assert "asmc" in data.results
        res = data.results["asmc"]
        assert res["n_pairs_decoded"] == 1
        assert res["sum_of_posteriors"].shape == (n_sites, small_dq.states)
        assert len(res["per_pair_posterior_means"]) == 1
        assert res["per_pair_posterior_means"][0].shape == (n_sites,)

    def test_multiple_pairs(self, small_dq, small_haplotypes, small_genetic_positions):
        n_sites = small_haplotypes.shape[1]

        data = SmcData()
        data.sequences = small_haplotypes
        data.uns["haplotypes"] = small_haplotypes
        data.uns["genetic_positions"] = small_genetic_positions
        data.uns["physical_positions"] = np.arange(n_sites, dtype=np.int32) * 1000
        data.uns["decoding_quantities"] = small_dq

        result = asmc(data, pairs=[(0, 1), (0, 2), (1, 3)])

        assert data.results["asmc"]["n_pairs_decoded"] == 3
        assert len(data.results["asmc"]["per_pair_posterior_means"]) == 3

    def test_posteriors_sum_positive(self, small_dq, small_haplotypes, small_genetic_positions):
        n_sites = small_haplotypes.shape[1]

        data = SmcData()
        data.sequences = small_haplotypes
        data.uns["haplotypes"] = small_haplotypes
        data.uns["genetic_positions"] = small_genetic_positions
        data.uns["physical_positions"] = np.arange(n_sites, dtype=np.int32) * 1000
        data.uns["decoding_quantities"] = small_dq

        asmc(data, pairs=[(0, 1)])
        sop = data.results["asmc"]["sum_of_posteriors"]
        assert np.all(sop >= 0)
        assert np.all(np.isfinite(sop))
