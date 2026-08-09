"""Independent Numba kernels for the panmictic PSMC+ hidden Markov model.

The formulas follow the discretized SMC model described by Li and Durbin and
the locus-specific emission rescaling described by Cousins et al.  This module
does not import or execute the preserved PSMC+ implementation; frozen upstream
intermediate arrays are used as external numerical oracles in the test suite.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True)
def psmcplus_time_boundaries(
    number_states: int,
    spread_1: float,
    spread_2: float,
    final_time_factor: float,
) -> np.ndarray:
    """Return the PSMC+ coalescent-time grid.

    ``final_time_factor`` uses a non-positive value as the internal sentinel
    for the original exponential-grid behavior.
    """
    boundaries = np.zeros(number_states + 1, dtype=np.float64)
    log_span = math.log1p(spread_2 / spread_1)
    stop = number_states - 1 if final_time_factor > 0.0 else number_states
    for index in range(stop):
        boundaries[index + 1] = spread_1 * math.exp((index / number_states) * log_span - 1.0)
    if final_time_factor > 0.0:
        boundaries[number_states] = boundaries[number_states - 1] * final_time_factor
    return boundaries


@numba.njit(cache=True)
def psmcplus_expected_times(
    boundaries: np.ndarray,
    inverse_sizes: np.ndarray,
    use_midpoints: bool,
) -> np.ndarray:
    """Return the conditional mean coalescence time in every interval."""
    number_states = inverse_sizes.size
    expected = np.empty(number_states, dtype=np.float64)
    for state in range(number_states):
        left = boundaries[state]
        width = boundaries[state + 1] - left
        if use_midpoints:
            expected[state] = left + 0.5 * width
            continue
        rate = inverse_sizes[state]
        scaled_width = rate * width
        if scaled_width > 50.0:
            expected[state] = left + 1.0 / rate
        else:
            expected[state] = left + 1.0 / rate - width / math.expm1(scaled_width)
    return expected


@numba.njit(cache=True)
def _integrated_survival(rate: float, width: float, factor: float) -> float:
    """Integral of ``exp(-factor * rate * t)`` from zero to ``width``."""
    return -math.expm1(-factor * rate * width) / (factor * rate)


@numba.njit(cache=True)
def psmcplus_conditional_transition(
    boundaries: np.ndarray,
    inverse_sizes: np.ndarray,
    expected_times: np.ndarray,
) -> np.ndarray:
    """Build transition probabilities conditional on a recombination event.

    The off-diagonal entries integrate over the possible recombination time on
    the current genealogy and subsequent coalescence time. The diagonal is the
    residual probability, including recombinations that return to the same
    discretized state.
    """
    number_states = inverse_sizes.size
    widths = np.empty(number_states, dtype=np.float64)
    cumulative_hazard = np.zeros(number_states + 1, dtype=np.float64)
    integral_one = np.empty(number_states, dtype=np.float64)
    integral_two = np.empty(number_states, dtype=np.float64)
    second_moment_tail = np.zeros(number_states + 1, dtype=np.float64)

    for state in range(number_states):
        width = boundaries[state + 1] - boundaries[state]
        rate = inverse_sizes[state]
        widths[state] = width
        cumulative_hazard[state + 1] = cumulative_hazard[state] + rate * width
        integral_one[state] = _integrated_survival(rate, width, 1.0)
        integral_two[state] = _integrated_survival(rate, width, 2.0)
        second_moment_tail[state + 1] = (
            second_moment_tail[state] * math.exp(-2.0 * rate * width) + integral_two[state]
        )

    conditional = np.zeros((number_states, number_states), dtype=np.float64)
    for current in range(number_states):
        current_time = expected_times[current]
        current_rate = inverse_sizes[current]

        for destination in range(current):
            destination_rate = inverse_sizes[destination]
            width = widths[destination]
            integral_h = (width - integral_two[destination]) / (2.0 * destination_rate)
            conditional[current, destination] = (
                destination_rate
                / current_time
                * (second_moment_tail[destination] * integral_two[destination] + integral_h)
            )

        elapsed = current_time - boundaries[current]
        within_survival = math.exp(-current_rate * elapsed)
        partial_integral_two = _integrated_survival(current_rate, elapsed, 2.0)
        recombination_prefix = (
            second_moment_tail[current] * within_survival * within_survival + partial_integral_two
        )
        for destination in range(current + 1, number_states):
            hazard = current_rate * (boundaries[current + 1] - current_time)
            hazard += cumulative_hazard[destination] - cumulative_hazard[current + 1]
            survival_to_destination = math.exp(-hazard)
            conditional[current, destination] = (
                inverse_sizes[destination]
                / current_time
                * survival_to_destination
                * integral_one[destination]
                * recombination_prefix
            )

        off_diagonal_sum = 0.0
        for destination in range(number_states):
            if destination != current:
                off_diagonal_sum += conditional[current, destination]
        conditional[current, current] = 1.0 - off_diagonal_sum
    return conditional


@numba.njit(cache=True)
def psmcplus_transition_stack(
    conditional_transition: np.ndarray,
    expected_times: np.ndarray,
    rho_per_bin: float,
    recombination_factors: np.ndarray,
    nonexponential: bool,
) -> np.ndarray:
    """Create one transition matrix for every local recombination factor.

    For the nonexponential option, the original implementation applies the
    linear approximation at the baseline rate and exponential relative-rate
    rescaling for a supplied map. The same observable semantics are retained
    here for native-versus-upstream validation.
    """
    number_rates = recombination_factors.size
    number_states = expected_times.size
    matrices = np.empty((number_rates, number_states, number_states), dtype=np.float64)
    for rate_index in range(number_rates):
        local_factor = recombination_factors[rate_index]
        for current in range(number_states):
            base_argument = rho_per_bin * expected_times[current]
            if nonexponential:
                baseline_exponential = -math.expm1(-base_argument)
                local_exponential = -math.expm1(-base_argument * local_factor)
                if baseline_exponential > 0.0:
                    recombination_probability = (
                        base_argument / baseline_exponential * local_exponential
                    )
                else:
                    recombination_probability = base_argument * local_factor
            else:
                recombination_probability = -math.expm1(-base_argument * local_factor)

            row_sum = 0.0
            for destination in range(number_states):
                if destination == current:
                    continue
                value = conditional_transition[current, destination] * recombination_probability
                matrices[rate_index, current, destination] = value
                row_sum += value
            matrices[rate_index, current, current] = 1.0 - row_sum
    return matrices


@numba.njit(cache=True)
def psmcplus_emission_midpoints(
    boundaries: np.ndarray,
    midpoint_last_interval: bool,
) -> np.ndarray:
    """Return the representative coalescence time used by emissions."""
    number_states = boundaries.size - 1
    midpoints = np.empty(number_states, dtype=np.float64)
    for state in range(number_states):
        midpoints[state] = 0.5 * (boundaries[state] + boundaries[state + 1])
    if not midpoint_last_interval:
        midpoints[number_states - 1] = boundaries[number_states - 1] + 1.5
    return midpoints


@numba.njit(cache=True)
def psmcplus_emission_vector(
    heterozygotes: int,
    masked_bases: int,
    bin_size: int,
    theta: float,
    midpoints: np.ndarray,
    mutation_factor: float,
) -> np.ndarray:
    """Return Poisson mutation-count probabilities across hidden states."""
    callable_bases = bin_size - masked_bases
    emissions = np.empty(midpoints.size, dtype=np.float64)
    for state in range(midpoints.size):
        mean = callable_bases * theta * midpoints[state] * mutation_factor
        if mean == 0.0:
            emissions[state] = 1.0 if heterozygotes == 0 else 0.0
        else:
            emissions[state] = math.exp(
                -mean + heterozygotes * math.log(mean) - math.lgamma(heterozygotes + 1.0)
            )
    return emissions


@numba.njit(cache=True)
def psmcplus_forward(
    heterozygotes: np.ndarray,
    masked_bases: np.ndarray,
    mutation_factors: np.ndarray,
    recombination_indices: np.ndarray,
    transitions: np.ndarray,
    initial_distribution: np.ndarray,
    bin_size: int,
    theta: float,
    midpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the scaled PSMC+ forward recursion."""
    sequence_length = heterozygotes.size
    number_states = initial_distribution.size
    forward = np.zeros((sequence_length, number_states), dtype=np.float64)
    scales = np.empty(sequence_length, dtype=np.float64)

    # The original inference path applies its baseline emission table at the
    # first bin and local mutation factors from the second bin onward.
    emissions = psmcplus_emission_vector(
        heterozygotes[0], masked_bases[0], bin_size, theta, midpoints, 1.0
    )
    total = 0.0
    for state in range(number_states):
        value = initial_distribution[state] * emissions[state]
        forward[0, state] = value
        total += value
    scales[0] = total
    for state in range(number_states):
        forward[0, state] /= total

    for position in range(1, sequence_length):
        emissions = psmcplus_emission_vector(
            heterozygotes[position],
            masked_bases[position],
            bin_size,
            theta,
            midpoints,
            mutation_factors[position],
        )
        transition = transitions[recombination_indices[position]]
        total = 0.0
        for destination in range(number_states):
            propagated = 0.0
            for current in range(number_states):
                propagated += forward[position - 1, current] * transition[current, destination]
            value = emissions[destination] * propagated
            forward[position, destination] = value
            total += value
        scales[position] = total
        for state in range(number_states):
            forward[position, state] /= total
    return forward, scales


@numba.njit(cache=True)
def psmcplus_backward(
    heterozygotes: np.ndarray,
    masked_bases: np.ndarray,
    mutation_factors: np.ndarray,
    recombination_indices: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    bin_size: int,
    theta: float,
    midpoints: np.ndarray,
) -> np.ndarray:
    """Run the scaled PSMC+ backward recursion."""
    sequence_length = heterozygotes.size
    number_states = transitions.shape[1]
    backward = np.zeros((sequence_length, number_states), dtype=np.float64)
    for state in range(number_states):
        backward[sequence_length - 1, state] = 1.0 / scales[sequence_length - 1]

    for position in range(sequence_length - 2, -1, -1):
        next_position = position + 1
        emissions = psmcplus_emission_vector(
            heterozygotes[next_position],
            masked_bases[next_position],
            bin_size,
            theta,
            midpoints,
            mutation_factors[next_position],
        )
        transition = transitions[recombination_indices[next_position]]
        for current in range(number_states):
            total = 0.0
            for destination in range(number_states):
                total += (
                    transition[current, destination]
                    * emissions[destination]
                    * backward[next_position, destination]
                )
            backward[position, current] = total / scales[position]
    return backward


@numba.njit(cache=True)
def psmcplus_expected_transition_counts(
    heterozygotes: np.ndarray,
    masked_bases: np.ndarray,
    mutation_factors: np.ndarray,
    recombination_indices: np.ndarray,
    transitions: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    bin_size: int,
    theta: float,
    midpoints: np.ndarray,
    upstream_origin_rate_index: bool,
) -> np.ndarray:
    """Accumulate expected transitions separately for each local rate class.

    The preserved implementation indexes a varying recombination map at the
    origin rather than destination bin only while accumulating EM evidence.
    ``upstream_origin_rate_index`` retains that observable fitting behavior for
    strict compatibility; ``False`` uses the transition that generated the
    destination bin.
    """
    number_rates = transitions.shape[0]
    number_states = transitions.shape[1]
    counts = np.zeros((number_rates, number_states, number_states), dtype=np.float64)
    for position in range(heterozygotes.size - 1):
        next_position = position + 1
        if upstream_origin_rate_index:
            rate_index = recombination_indices[position]
        else:
            rate_index = recombination_indices[next_position]
        emissions = psmcplus_emission_vector(
            heterozygotes[next_position],
            masked_bases[next_position],
            bin_size,
            theta,
            midpoints,
            mutation_factors[next_position],
        )
        for current in range(number_states):
            for destination in range(number_states):
                counts[rate_index, current, destination] += (
                    forward[position, current]
                    * transitions[rate_index, current, destination]
                    * emissions[destination]
                    * backward[next_position, destination]
                )
    return counts


@numba.njit(cache=True)
def psmcplus_marginal_recombination(
    heterozygotes: np.ndarray,
    masked_bases: np.ndarray,
    mutation_factors: np.ndarray,
    recombination_indices: np.ndarray,
    transitions: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    bin_size: int,
    theta: float,
    midpoints: np.ndarray,
) -> np.ndarray:
    """Return state-changing and same-state probability at each boundary."""
    number_states = transitions.shape[1]
    probabilities = np.empty((heterozygotes.size - 1, 2), dtype=np.float64)
    for position in range(heterozygotes.size - 1):
        next_position = position + 1
        rate_index = recombination_indices[next_position]
        emissions = psmcplus_emission_vector(
            heterozygotes[next_position],
            masked_bases[next_position],
            bin_size,
            theta,
            midpoints,
            mutation_factors[next_position],
        )
        changed = 0.0
        unchanged = 0.0
        for current in range(number_states):
            for destination in range(number_states):
                value = (
                    forward[position, current]
                    * transitions[rate_index, current, destination]
                    * emissions[destination]
                    * backward[next_position, destination]
                )
                if current == destination:
                    unchanged += value
                else:
                    changed += value
        total = changed + unchanged
        probabilities[position, 0] = changed / total
        probabilities[position, 1] = unchanged / total
    return probabilities


def psmcplus_stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Return the normalized stationary row distribution of a transition matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(transition, dtype=float).T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    distribution = np.real(eigenvectors[:, index])
    if distribution.sum() < 0.0:
        distribution = -distribution
    distribution /= distribution.sum()
    distribution[np.abs(distribution) < 1e-15] = 0.0
    return distribution


__all__ = [
    "psmcplus_backward",
    "psmcplus_conditional_transition",
    "psmcplus_emission_midpoints",
    "psmcplus_emission_vector",
    "psmcplus_expected_times",
    "psmcplus_expected_transition_counts",
    "psmcplus_forward",
    "psmcplus_marginal_recombination",
    "psmcplus_stationary_distribution",
    "psmcplus_time_boundaries",
    "psmcplus_transition_stack",
]
