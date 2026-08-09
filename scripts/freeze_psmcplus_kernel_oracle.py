#!/usr/bin/env python3
"""Freeze PSMC+ intermediate arrays from the immutable upstream source.

This script intentionally imports only the pinned implementation under
``vendor/PSMCplus``. It is an oracle generator, not part of native execution.
Run it from the repository root with the ``psmcplus`` optional dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/psmcplus/kernel_oracle_v1.npz"),
    )
    parser.add_argument(
        "--preprocessing-output",
        type=Path,
        default=Path("tests/data/psmcplus/preprocessing_oracle_v1.npz"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "vendor/PSMCplus"
    if not source.is_dir():
        raise SystemExit("Initialize the pinned vendor/PSMCplus submodule first.")

    # NumPy 2 removed this alias. The preservation runner applies the same
    # process-local compatibility adjustment without changing vendor files.
    np.math = math  # type: ignore[attr-defined]
    sys.path.insert(0, str(source))

    from BaumWelch import get_stationary_distribution_theory  # noqa: PLC0415
    from transition_matrix import (  # noqa: PLC0415
        Transition_Matrix,
        e_beta,
        time_intervals,
    )
    from utils import (  # noqa: PLC0415
        backward_matmul_scaled_fcn,
        bin_sequence,
        calculate_transition_evidence,
        forward_matmul_scaled_fcn,
        write_emission_probs_b_slice,
        write_Q_array_withR,
    )

    number_states = 4
    spread_1 = 0.1
    spread_2 = 50.0
    inverse_sizes = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    rho_per_bin = 0.075
    theta = 0.0012
    bin_size = 100
    recombination_factors = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    heterozygotes = np.array([0, 1, 0, 2, 1, 0], dtype=np.int64)
    masked_bases = np.array([0, 0, 4, 0, 10, 0], dtype=np.int64)
    mutation_factors = np.array([1.0, 0.8, 1.2, 1.0, 0.6, 1.5])
    recombination_indices = np.array([1, 0, 1, 2, 1, 0], dtype=np.int64)

    boundaries = time_intervals(number_states, spread_1, spread_2)
    expected_times = np.array(
        [
            e_beta(
                number_states,
                boundaries,
                inverse_sizes,
                None,
                None,
                None,
                None,
                state,
                False,
            )
            for state in range(number_states)
        ]
    )
    transition_model = Transition_Matrix(
        D=number_states,
        spread_1=spread_1,
        spread_2=spread_2,
    )
    conditional_transition = transition_model.write_tm(
        lambda_A=inverse_sizes,
        lambda_B=None,
        T_S_index=None,
        T_E_index=None,
        gamma=None,
        rho=1,
    )
    baseline_model = Transition_Matrix(
        D=number_states,
        spread_1=spread_1,
        spread_2=spread_2,
    )
    baseline_transition = baseline_model.write_tm(
        lambda_A=inverse_sizes,
        lambda_B=None,
        T_S_index=None,
        T_E_index=None,
        gamma=None,
        rho=rho_per_bin,
    )
    transition_stack = write_Q_array_withR(
        baseline_transition,
        recombination_factors,
        rho_per_bin,
        number_states,
        spread_1,
        spread_2,
        inverse_sizes,
        False,
    )
    emission_midpoints = np.array(
        [0.5 * (boundaries[state] + boundaries[state + 1]) for state in range(number_states)]
    )
    emission_midpoints[-1] = boundaries[-2] + 1.5
    emission_vectors = np.stack(
        [
            write_emission_probs_b_slice(
                number_states,
                bin_size,
                theta,
                emission_midpoints,
                mutation_factors[position],
                masked_bases[position] * bin_size + heterozygotes[position],
            )
            for position in range(heterozygotes.size)
        ]
    )
    initial_distribution = np.real(get_stationary_distribution_theory(baseline_transition))
    sequence = masked_bases * bin_size + heterozygotes
    initial_emissions = np.zeros(
        (number_states, bin_size * bin_size + int(heterozygotes.max()) + 1)
    )
    initial_emissions[:, sequence[0]] = write_emission_probs_b_slice(
        number_states,
        bin_size,
        theta,
        emission_midpoints,
        1.0,
        sequence[0],
    )
    forward, scales = forward_matmul_scaled_fcn(
        sequence,
        number_states,
        initial_distribution,
        initial_emissions,
        transition_stack,
        bin_size,
        theta,
        emission_midpoints,
        np.arange(mutation_factors.size),
        mutation_factors,
        recombination_indices,
    )
    backward = backward_matmul_scaled_fcn(
        sequence,
        number_states,
        transition_stack,
        bin_size,
        theta,
        emission_midpoints,
        initial_emissions,
        scales,
        np.arange(mutation_factors.size),
        mutation_factors,
        recombination_indices,
    )

    def sequence_function(_file: int):
        return (
            sequence,
            np.arange(mutation_factors.size),
            mutation_factors,
            recombination_indices,
            recombination_factors,
        )

    expected_transition_counts, log_likelihood = calculate_transition_evidence(
        sequence_function,
        0,
        number_states,
        initial_distribution,
        initial_emissions,
        transition_stack,
        theta,
        rho_per_bin,
        bin_size,
        int(heterozygotes.max()),
        emission_midpoints,
        spread_1,
        spread_2,
        False,
    )
    posterior = (forward * backward).T
    posterior /= posterior.sum(axis=1, keepdims=True)
    marginal_recombination = np.empty((heterozygotes.size - 1, 2))
    for position in range(heterozygotes.size - 1):
        next_position = position + 1
        weighted_backward = backward[:, next_position] * emission_vectors[next_position]
        joint = (
            forward[:, position, np.newaxis]
            * weighted_backward[np.newaxis, :]
            * transition_stack[recombination_indices[next_position]]
        )
        unchanged = np.trace(joint)
        changed = joint.sum() - unchanged
        marginal_recombination[position] = (changed, unchanged) / joint.sum()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        oracle_commit=np.array(UPSTREAM_COMMIT),
        number_states=np.array(number_states),
        spread_1=np.array(spread_1),
        spread_2=np.array(spread_2),
        inverse_sizes=inverse_sizes,
        rho_per_bin=np.array(rho_per_bin),
        theta=np.array(theta),
        bin_size=np.array(bin_size),
        recombination_factors=recombination_factors,
        heterozygotes=heterozygotes,
        masked_bases=masked_bases,
        mutation_factors=mutation_factors,
        recombination_indices=recombination_indices,
        boundaries=boundaries,
        expected_times=expected_times,
        conditional_transition=conditional_transition,
        baseline_transition=baseline_transition,
        transition_stack=transition_stack,
        emission_midpoints=emission_midpoints,
        emission_vectors=emission_vectors,
        initial_distribution=initial_distribution,
        forward=forward.T,
        backward=backward.T,
        scales=scales,
        log_likelihood=np.array(log_likelihood),
        posterior=posterior,
        expected_transition_counts=expected_transition_counts,
        marginal_recombination=marginal_recombination,
    )
    print(output)

    fixture_dir = root / "tests/data/psmcplus"
    input_path = fixture_dir / "preprocessing_masked.mhs"
    mutation_path = fixture_dir / "preprocessing_mutation.bed"
    recombination_path = fixture_dir / "preprocessing_recombination.bed"
    preprocessing = bin_sequence(
        str(input_path),
        10,
        {str(input_path): str(mutation_path)},
        {str(input_path): str(recombination_path)},
    )
    number_bins = int(preprocessing[3] / 10)
    preprocessing_heterozygotes = np.zeros(number_bins, dtype=np.int64)
    preprocessing_masks = np.zeros(number_bins, dtype=np.int64)
    preprocessing_heterozygotes[preprocessing[0][0]] = preprocessing[0][1]
    preprocessing_masks[preprocessing[1][0]] = preprocessing[1][1]

    def fixture_hash(path: Path) -> np.ndarray:
        return np.array(hashlib.sha256(path.read_bytes()).hexdigest())

    preprocessing_output = args.preprocessing_output.resolve()
    preprocessing_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preprocessing_output,
        oracle_commit=np.array(UPSTREAM_COMMIT),
        bin_size=np.array(10),
        input_sha256=fixture_hash(input_path),
        mutation_map_sha256=fixture_hash(mutation_path),
        recombination_map_sha256=fixture_hash(recombination_path),
        sequence_length=np.array(preprocessing[3]),
        number_heterozygotes=np.array(preprocessing[4]),
        number_masked_bases=np.array(preprocessing[5]),
        maximum_heterozygotes=np.array(preprocessing[2]),
        heterozygotes=preprocessing_heterozygotes,
        masked_bases=preprocessing_masks,
        mutation_indices=preprocessing[6],
        mutation_factors=preprocessing[7],
        mutation_factor_sequence=preprocessing[7][preprocessing[6]],
        recombination_indices=preprocessing[8],
        recombination_factors=preprocessing[9],
        recombination_factor_sequence=preprocessing[9][preprocessing[8]],
    )
    print(preprocessing_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
