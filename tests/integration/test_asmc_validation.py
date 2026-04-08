"""Integration test: validate smckit ASMC against original C++ reference outputs.

Uses the ASMC_data test fixtures (n300 array example) with the
30-100-2000_CEU decoding quantities. Compares posterior mean TMRCA
and MAP outputs against the regression reference files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from smckit.io._asmc import read_asmc, read_decoding_quantities, read_hap, read_map
from smckit.tl._asmc import (
    asmc,
    backward,
    compute_posteriors,
    encode_pair,
    forward,
    posterior_map_tmrca,
    posterior_mean_tmrca,
    prepare_emissions,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "ASMC" / "ASMC_data"
EXAMPLE_ROOT = DATA_DIR / "examples" / "asmc" / "exampleFile.n300.array"
DQ_FILE = DATA_DIR / "decoding_quantities" / "30-100-2000_CEU.decodingQuantities.gz"
REF_MEANS = DATA_DIR / "testing" / "asmc" / "regression" / "regression.perPairPosteriorMeans.gz"
REF_MAPS = DATA_DIR / "testing" / "asmc" / "regression" / "regression.perPairMAP.gz"

SKIP_REASON = "ASMC_data submodule not initialized"


def _data_available() -> bool:
    return DQ_FILE.exists() and Path(str(EXAMPLE_ROOT) + ".hap.gz").exists()


@pytest.fixture(scope="module")
def asmc_data():
    """Load full ASMC test data."""
    if not _data_available():
        pytest.skip(SKIP_REASON)
    return read_asmc(str(EXAMPLE_ROOT), str(DQ_FILE))


@pytest.fixture(scope="module")
def ref_posterior_means():
    if not REF_MEANS.exists():
        pytest.skip(SKIP_REASON)
    return np.loadtxt(str(REF_MEANS))


@pytest.fixture(scope="module")
def ref_maps():
    if not REF_MAPS.exists():
        pytest.skip(SKIP_REASON)
    return np.loadtxt(str(REF_MAPS))


# ---------------------------------------------------------------------------
# Data loading validation
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_haplotype_shape(self, asmc_data):
        haps = asmc_data.uns["haplotypes"]
        assert haps.shape == (300, 6760)

    def test_genetic_positions_shape(self, asmc_data):
        gen = asmc_data.uns["genetic_positions"]
        assert gen.shape == (6760,)

    def test_genetic_positions_monotonic(self, asmc_data):
        gen = asmc_data.uns["genetic_positions"]
        assert np.all(gen[1:] >= gen[:-1])

    def test_decoding_quantities_states(self, asmc_data):
        dq = asmc_data.uns["decoding_quantities"]
        assert dq.states == 69

    def test_decoding_quantities_initial_prob(self, asmc_data):
        dq = asmc_data.uns["decoding_quantities"]
        np.testing.assert_allclose(dq.initial_state_prob.sum(), 1.0, atol=1e-4)

    def test_undistinguished_counts_shape(self, asmc_data):
        uc = asmc_data.uns["undistinguished_counts"]
        assert uc.shape == (6760, 3)

    def test_undistinguished_counts_range(self, asmc_data):
        uc = asmc_data.uns["undistinguished_counts"]
        dq = asmc_data.uns["decoding_quantities"]
        # Undistinguished counts should be in [-1, csfs_samples-2]
        assert uc.min() >= -1
        assert uc.max() <= dq.csfs_samples - 2


# ---------------------------------------------------------------------------
# Single-pair HMM validation
# ---------------------------------------------------------------------------

class TestSinglePairHMM:
    """Validate HMM forward/backward on a single haplotype pair."""

    @pytest.fixture
    def single_pair_result(self, asmc_data):
        """Decode a single haplotype pair and return posteriors."""
        dq = asmc_data.uns["decoding_quantities"]
        haps = asmc_data.uns["haplotypes"]
        gen_pos = asmc_data.uns["genetic_positions"]
        uc = asmc_data.uns["undistinguished_counts"]
        n_sites = haps.shape[1]

        e1, e0m1, e2m0 = prepare_emissions(
            dq, gen_pos, n_sites,
            use_csfs=True,
            fold_data=True,
            undistinguished_counts=uc,
        )

        obs = encode_pair(haps, 1, 2)

        alpha = forward(dq, e1, e0m1, e2m0, obs.obs_is_zero, obs.obs_is_two, gen_pos)
        beta = backward(dq, e1, e0m1, e2m0, obs.obs_is_zero, obs.obs_is_two, gen_pos)
        posteriors = compute_posteriors(alpha, beta)

        return posteriors, dq

    def test_posteriors_normalized(self, single_pair_result):
        posteriors, _ = single_pair_result
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_posteriors_nearly_non_negative(self, single_pair_result):
        """Posteriors may have tiny negatives from CSFS emission cancellation."""
        posteriors, _ = single_pair_result
        assert posteriors.min() > -0.02

    def test_posteriors_finite(self, single_pair_result):
        posteriors, _ = single_pair_result
        assert np.all(np.isfinite(posteriors))

    def test_posterior_means_reasonable(self, single_pair_result):
        posteriors, dq = single_pair_result
        means = posterior_mean_tmrca(posteriors, dq.expected_times)
        # Means should be in the range of expected times
        assert means.min() >= dq.expected_times[0]
        assert means.max() <= dq.expected_times[-1]
        # Means should be in same order of magnitude as reference (~1e3 to ~1e5)
        assert means.mean() > 100
        assert means.mean() < 200000


# ---------------------------------------------------------------------------
# Full regression comparison
# ---------------------------------------------------------------------------

class TestRegressionComparison:
    """Compare smckit ASMC against original C++ regression reference."""

    @pytest.fixture(scope="class")
    def smckit_results(self, asmc_data):
        """Run smckit ASMC on the same 3 pairs as the regression test."""
        pairs = [(1, 2), (2, 3), (3, 4)]
        asmc(
            asmc_data,
            pairs=pairs,
            mode="array",
            fold_data=True,
            store_per_pair_posterior_mean=True,
            store_per_pair_map=True,
        )
        return asmc_data.results["asmc"]

    def test_correct_number_of_pairs(self, smckit_results):
        assert smckit_results["n_pairs_decoded"] == 3

    def test_posterior_means_shape(self, smckit_results):
        means = smckit_results["per_pair_posterior_means"]
        assert len(means) == 3
        assert means[0].shape == (6760,)

    def test_posterior_means_finite(self, smckit_results):
        for m in smckit_results["per_pair_posterior_means"]:
            assert np.all(np.isfinite(m))

    def test_posterior_means_positive(self, smckit_results):
        for m in smckit_results["per_pair_posterior_means"]:
            assert np.all(m > 0)

    def test_posterior_means_order_of_magnitude(self, smckit_results, ref_posterior_means):
        """Posterior means should stay in the same numerical regime as upstream."""
        our_means = np.array(smckit_results["per_pair_posterior_means"])
        # Both should be in the ~1000 to ~130000 range
        assert our_means.min() > 100, f"Min too low: {our_means.min()}"
        assert our_means.max() < 500000, f"Max too high: {our_means.max()}"
        # Reference range
        ref_min, ref_max = ref_posterior_means.min(), ref_posterior_means.max()
        our_min, our_max = our_means.min(), our_means.max()
        # Within 1 order of magnitude
        assert our_min < ref_min * 10
        assert our_max > ref_max / 10

    def test_posterior_means_match_reference_within_tolerance(
        self, smckit_results, ref_posterior_means
    ):
        """Posterior means should remain numerically close to the reference."""
        our_means = np.array(smckit_results["per_pair_posterior_means"])
        assert np.allclose(our_means, ref_posterior_means, rtol=1e-2)

    def test_sum_of_posteriors_shape(self, smckit_results):
        sop = smckit_results["sum_of_posteriors"]
        assert sop.shape == (6760, 69)

    def test_sum_of_posteriors_nearly_non_negative(self, smckit_results):
        """Sum of posteriors may have tiny negatives from CSFS emission cancellation."""
        sop = smckit_results["sum_of_posteriors"]
        assert sop.min() > -0.1

    def test_map_estimates_shape(self, smckit_results):
        maps = smckit_results["per_pair_maps"]
        assert len(maps) == 3
        assert maps[0].shape == (6760,)

    def test_map_estimates_are_valid_state_indices(self, smckit_results):
        """MAP outputs should match upstream ASMC state-index semantics."""
        n_states = len(smckit_results["expected_times"])
        for m in smckit_results["per_pair_maps"]:
            assert np.issubdtype(m.dtype, np.integer)
            assert m.min() >= 0
            assert m.max() < n_states

    def test_map_agreement_with_reference(self, smckit_results, ref_maps):
        """MAP state indices should stay close to the upstream regression output."""
        our_maps = np.array(smckit_results["per_pair_maps"])
        ref_maps = ref_maps.astype(np.int32)
        assert np.mean(our_maps == ref_maps) > 0.75
        for pair_idx in range(3):
            corr = np.corrcoef(our_maps[pair_idx], ref_maps[pair_idx])[0, 1]
            assert corr > 0.97, f"Pair {pair_idx}: MAP correlation {corr:.3f} too low"
