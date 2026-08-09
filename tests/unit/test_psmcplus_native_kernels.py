"""Independent native PSMC+ kernels against frozen upstream intermediates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smckit.backends._numba_psmcplus import (
    psmcplus_backward,
    psmcplus_conditional_transition,
    psmcplus_emission_midpoints,
    psmcplus_emission_vector,
    psmcplus_expected_times,
    psmcplus_expected_transition_counts,
    psmcplus_forward,
    psmcplus_marginal_recombination,
    psmcplus_stationary_distribution,
    psmcplus_time_boundaries,
    psmcplus_transition_stack,
)

ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = ROOT / "tests/data/psmcplus/kernel_oracle_v1.npz"
UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"


def _oracle() -> dict[str, np.ndarray]:
    with np.load(ORACLE_PATH) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_time_expected_and_transition_kernels_match_frozen_upstream() -> None:
    oracle = _oracle()
    assert str(oracle["oracle_commit"]) == UPSTREAM_COMMIT
    boundaries = psmcplus_time_boundaries(
        int(oracle["number_states"]),
        float(oracle["spread_1"]),
        float(oracle["spread_2"]),
        -1.0,
    )
    expected_times = psmcplus_expected_times(
        boundaries,
        oracle["inverse_sizes"],
        False,
    )
    conditional = psmcplus_conditional_transition(
        boundaries,
        oracle["inverse_sizes"],
        expected_times,
    )
    transitions = psmcplus_transition_stack(
        conditional,
        expected_times,
        float(oracle["rho_per_bin"]),
        oracle["recombination_factors"],
        False,
    )

    np.testing.assert_allclose(boundaries, oracle["boundaries"], rtol=0, atol=0)
    np.testing.assert_allclose(
        expected_times,
        oracle["expected_times"],
        rtol=2e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        conditional,
        oracle["conditional_transition"],
        rtol=2e-14,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        transitions,
        oracle["transition_stack"],
        rtol=2e-14,
        atol=2e-15,
    )
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(transitions.sum(axis=2), 1.0, rtol=0, atol=2e-15)
    assert np.all(conditional >= 0)
    assert np.all(transitions >= 0)


def test_time_grid_alternatives_and_nonexponential_transitions_are_valid() -> None:
    oracle = _oracle()
    boundaries = psmcplus_time_boundaries(4, 0.1, 50.0, 3.0)
    assert boundaries[-1] == boundaries[-2] * 3.0
    midpoint_times = psmcplus_expected_times(boundaries, np.ones(4), True)
    np.testing.assert_allclose(midpoint_times, 0.5 * (boundaries[:-1] + boundaries[1:]))

    conditional = psmcplus_conditional_transition(
        oracle["boundaries"],
        oracle["inverse_sizes"],
        oracle["expected_times"],
    )
    transitions = psmcplus_transition_stack(
        conditional,
        oracle["expected_times"],
        float(oracle["rho_per_bin"]),
        oracle["recombination_factors"],
        True,
    )
    np.testing.assert_allclose(transitions.sum(axis=2), 1.0, rtol=0, atol=2e-15)
    assert np.all(transitions >= 0)


def test_emission_kernels_match_every_frozen_upstream_position() -> None:
    oracle = _oracle()
    midpoints = psmcplus_emission_midpoints(oracle["boundaries"], False)
    np.testing.assert_allclose(midpoints, oracle["emission_midpoints"], rtol=0, atol=0)
    for position in range(oracle["heterozygotes"].size):
        actual = psmcplus_emission_vector(
            int(oracle["heterozygotes"][position]),
            int(oracle["masked_bases"][position]),
            int(oracle["bin_size"]),
            float(oracle["theta"]),
            midpoints,
            float(oracle["mutation_factors"][position]),
        )
        np.testing.assert_allclose(
            actual,
            oracle["emission_vectors"][position],
            rtol=2e-14,
            atol=1e-15,
        )


def test_forward_backward_posterior_and_likelihood_match_upstream() -> None:
    oracle = _oracle()
    initial = psmcplus_stationary_distribution(oracle["baseline_transition"])
    np.testing.assert_allclose(
        initial,
        oracle["initial_distribution"],
        rtol=5e-13,
        atol=2e-14,
    )
    forward, scales = psmcplus_forward(
        oracle["heterozygotes"],
        oracle["masked_bases"],
        oracle["mutation_factors"],
        oracle["recombination_indices"],
        oracle["transition_stack"],
        oracle["initial_distribution"],
        int(oracle["bin_size"]),
        float(oracle["theta"]),
        oracle["emission_midpoints"],
    )
    backward = psmcplus_backward(
        oracle["heterozygotes"],
        oracle["masked_bases"],
        oracle["mutation_factors"],
        oracle["recombination_indices"],
        oracle["transition_stack"],
        scales,
        int(oracle["bin_size"]),
        float(oracle["theta"]),
        oracle["emission_midpoints"],
    )
    posterior = forward * backward
    posterior /= posterior.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(forward, oracle["forward"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(scales, oracle["scales"], rtol=2e-15, atol=1e-18)
    np.testing.assert_allclose(backward, oracle["backward"], rtol=2e-14, atol=3e-14)
    np.testing.assert_allclose(posterior, oracle["posterior"], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        np.log(scales).sum(), oracle["log_likelihood"], rtol=2e-15, atol=2e-15
    )


def test_em_evidence_and_marginal_recombination_match_upstream_semantics() -> None:
    oracle = _oracle()
    counts = psmcplus_expected_transition_counts(
        oracle["heterozygotes"],
        oracle["masked_bases"],
        oracle["mutation_factors"],
        oracle["recombination_indices"],
        oracle["transition_stack"],
        oracle["forward"],
        oracle["backward"],
        int(oracle["bin_size"]),
        float(oracle["theta"]),
        oracle["emission_midpoints"],
        True,
    )
    np.testing.assert_allclose(
        counts.sum(axis=0),
        oracle["expected_transition_counts"],
        rtol=2e-14,
        atol=1e-15,
    )
    marginal = psmcplus_marginal_recombination(
        oracle["heterozygotes"],
        oracle["masked_bases"],
        oracle["mutation_factors"],
        oracle["recombination_indices"],
        oracle["transition_stack"],
        oracle["forward"],
        oracle["backward"],
        int(oracle["bin_size"]),
        float(oracle["theta"]),
        oracle["emission_midpoints"],
    )
    np.testing.assert_allclose(
        marginal,
        oracle["marginal_recombination"],
        rtol=2e-14,
        atol=1e-15,
    )
    np.testing.assert_allclose(marginal.sum(axis=1), 1.0, rtol=0, atol=2e-15)
