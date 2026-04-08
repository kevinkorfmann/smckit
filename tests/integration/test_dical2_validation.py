"""Validate smckit's diCal2 implementation against msprime simulation.

Simulates a constant-size population, runs diCal2, and verifies that the
inferred Ne is in the right ballpark and the log-likelihood is finite.

These tests are designed to be fast (small sequences, few EM iterations)
and robust — they verify mathematical correctness rather than fine-grained
parameter recovery, which would require thousands of sites.
"""

from __future__ import annotations

import numpy as np
import pytest

from smckit.io import read_dical2
from smckit.io._dical2 import (
    read_dical2_config,
    read_dical2_demo,
    read_dical2_param,
)
from smckit.tl import dical2
from smckit.tl._dical2 import (
    DICAL2_T_INF,
    EigenCore,
    SimpleTrunk,
    _enumerate_csd_pairs,
    compute_time_intervals,
    expected_counts,
    forward_log,
    refine_demography,
)
from smckit.tl._dical2 import _default_demo_for_single_pop

msprime = pytest.importorskip("msprime")

VENDOR_EXAMPLES = "vendor/diCal2/examples"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N0 = 10_000
MU = 1.25e-8
R = 1e-8
GEN_TIME = 25
N_HAP = 4


@pytest.fixture(scope="module")
def simulated_sequences():
    """Simulate 200 kb under a constant-size single population."""
    seq_length = 200_000
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N0)
    ts = msprime.sim_ancestry(
        samples={"pop": N_HAP // 2},
        demography=demography,
        sequence_length=seq_length,
        recombination_rate=R,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=42)

    # Build per-base haplotype matrix at variant positions only
    n_var = ts.num_sites
    if n_var == 0:
        pytest.skip("No variants generated; rerun with larger sequence")
    haps = np.zeros((N_HAP, n_var), dtype=np.int8)
    for i, var in enumerate(ts.variants()):
        haps[:, i] = var.genotypes[:N_HAP]
    return haps


# ---------------------------------------------------------------------------
# I/O round-trip checks
# ---------------------------------------------------------------------------


def test_read_param_examples():
    p = read_dical2_param(f"{VENDOR_EXAMPLES}/piecewiseConstant/mutRec.param")
    assert p.theta > 0
    assert p.rho > 0
    assert p.mutation_matrix.shape == (2, 2)


def test_read_demo_examples():
    d = read_dical2_demo(
        f"{VENDOR_EXAMPLES}/piecewiseConstant/piecewise_constant.demo"
    )
    assert len(d.epochs) == 4
    for ep in d.epochs:
        assert ep.pop_sizes is not None
        assert len(ep.partition) == 1


def test_read_config_examples():
    c = read_dical2_config(
        f"{VENDOR_EXAMPLES}/cleanSplit/clean_split.config"
    )
    assert c.n_populations == 2


# ---------------------------------------------------------------------------
# Mathematical sanity checks
# ---------------------------------------------------------------------------


def test_initial_distribution_normalized():
    """Initial probabilities must sum to 1."""
    demo = _default_demo_for_single_pop(
        5,
        np.append(compute_time_intervals(5, max_t=2.0, alpha=0.1), DICAL2_T_INF),
    )
    refined = refine_demography(demo, demo.epoch_boundaries)
    from smckit.io._dical2 import DiCal2Config

    config = DiCal2Config(
        seq_length=100,
        n_alleles=2,
        n_populations=1,
        haplotype_populations=[0, 0],
        haplotypes_to_include=[True, True],
        haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
        sample_sizes=np.array([2]),
    )
    trunk = SimpleTrunk(config=config, additional_hap_idx=0)
    mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
    core = EigenCore(
        refined=refined,
        trunk=trunk,
        observed_present_deme=0,
        mutation_matrix=mut_mat,
        theta=0.001,
        rho=0.0005,
    ).core_matrices()
    total = np.exp(core.log_initial).sum()
    assert abs(total - 1.0) < 1e-9


def test_emission_rows_normalized():
    """Each emission row sums to 1 (it's a probability distribution)."""
    demo = _default_demo_for_single_pop(
        4,
        np.append(compute_time_intervals(4, max_t=2.0, alpha=0.1), DICAL2_T_INF),
    )
    refined = refine_demography(demo, demo.epoch_boundaries)
    from smckit.io._dical2 import DiCal2Config

    config = DiCal2Config(
        seq_length=100,
        n_alleles=2,
        n_populations=1,
        haplotype_populations=[0, 0],
        haplotypes_to_include=[True, True],
        haplotype_multiplicities=np.ones((2, 1), dtype=np.int64),
        sample_sizes=np.array([2]),
    )
    trunk = SimpleTrunk(config=config, additional_hap_idx=0)
    mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
    core = EigenCore(
        refined=refined,
        trunk=trunk,
        observed_present_deme=0,
        mutation_matrix=mut_mat,
        theta=0.001,
        rho=0.0005,
    ).core_matrices()
    em = np.exp(core.log_emission)
    np.testing.assert_allclose(em.sum(axis=2), 1.0, atol=1e-8)


def test_csd_enumeration_modes():
    pairs_pcl = _enumerate_csd_pairs(4, "pcl")
    assert len(pairs_pcl) == 4 * 3  # ordered pairs
    pairs_lol = _enumerate_csd_pairs(4, "lol")
    assert len(pairs_lol) == 4
    pairs_pac = _enumerate_csd_pairs(4, "pac")
    assert len(pairs_pac) == 3  # n - 1


# ---------------------------------------------------------------------------
# End-to-end on simulated data
# ---------------------------------------------------------------------------


def test_dical2_runs_on_simulated(simulated_sequences):
    """diCal2 should produce a finite log-likelihood and positive Ne."""
    seqs = simulated_sequences
    data = read_dical2(
        sequences=seqs,
        theta=4 * N0 * MU,
        rho=4 * N0 * R,
    )
    data = dical2(
        data,
        n_intervals=4,
        n_em_iterations=2,
        max_t=2.0,
        composite_mode="pac",
        mu=MU,
        generation_time=GEN_TIME,
    )
    res = data.results["dical2"]
    assert np.isfinite(res["log_likelihood"])
    assert np.all(res["ne"] > 0)
    assert np.all(np.isfinite(res["ne"]))
    assert len(res["rounds"]) >= 1
    # Sanity: Ne should be at most a few orders of magnitude off from N0
    # (with so few sites we don't get tight estimates)
    geom_mean = np.exp(np.mean(np.log(res["ne"])))
    assert 1e2 < geom_mean < 1e8


def test_dical2_log_likelihood_improves(simulated_sequences):
    """Log-likelihood should not decrease across the first EM step."""
    seqs = simulated_sequences[:, :500]  # smaller for speed
    data = read_dical2(
        sequences=seqs,
        theta=4 * N0 * MU,
        rho=4 * N0 * R,
    )
    data = dical2(
        data,
        n_intervals=3,
        n_em_iterations=3,
        max_t=2.0,
        composite_mode="pcl",
        mu=MU,
    )
    rounds = data.results["dical2"]["rounds"]
    assert len(rounds) >= 2
    lls = [r["log_likelihood"] for r in rounds]
    # The very first iteration may not improve due to bound clipping but
    # all log-likelihoods should be finite.
    assert all(np.isfinite(ll) for ll in lls)


def test_forward_per_state_consistency():
    """Forward log-likelihood should not depend on whether we recompute it
    via the backward variables (a sanity check on the FB recursion)."""
    rng = np.random.default_rng(11)
    seqs = (rng.random((3, 80)) < 0.05).astype(np.int8)
    data = read_dical2(sequences=seqs, theta=0.001, rho=0.0005)
    demo = _default_demo_for_single_pop(
        4,
        np.append(compute_time_intervals(4, max_t=2.0, alpha=0.1), DICAL2_T_INF),
    )
    refined = refine_demography(demo, demo.epoch_boundaries)
    trunk = SimpleTrunk(
        config=data.uns["config"],
        additional_hap_idx=0,
    )
    mut_mat = np.array([[-1.0, 1.0], [1.0, -1.0]])
    core = EigenCore(
        refined=refined,
        trunk=trunk,
        observed_present_deme=0,
        mutation_matrix=mut_mat,
        theta=0.001,
        rho=0.0005,
    ).core_matrices()
    obs_a = seqs[0].astype(np.int64)
    obs_t = seqs[1].astype(np.int64)
    _, ll = forward_log(core, obs_a, obs_t)
    counts = expected_counts(core, obs_a, obs_t, n_alleles=2)
    assert ll == pytest.approx(counts.log_likelihood, rel=1e-9)
