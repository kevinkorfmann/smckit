"""Independent native fitting and decoding engine for panmictic PSMC+.

The implementation composes the native preprocessing and Numba HMM kernels.
It does not import the preserved source. Frozen upstream outputs are consumed
only by tests outside this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from smckit.backends._numba_psmcplus import (
    psmcplus_backward,
    psmcplus_conditional_transition,
    psmcplus_emission_midpoints,
    psmcplus_expected_times,
    psmcplus_expected_transition_counts,
    psmcplus_forward,
    psmcplus_marginal_recombination,
    psmcplus_stationary_distribution,
    psmcplus_time_boundaries,
    psmcplus_transition_stack,
)
from smckit.backends._psmcplus_preprocessing import (
    PSMCPlusSequence,
    prepare_psmcplus_sequence,
)


@dataclass(frozen=True)
class PSMCPlusSegmentLayout:
    """Mapping between optimizer parameters and the full lambda trajectory."""

    widths: tuple[int, ...]
    fixed: tuple[bool, ...]
    initial: np.ndarray

    @property
    def number_states(self) -> int:
        """Number of expanded time states."""
        return sum(self.widths)

    @property
    def free_segment_indices(self) -> tuple[int, ...]:
        """Indices of segment values optimized by the fitting engine."""
        return tuple(index for index, is_fixed in enumerate(self.fixed) if not is_fixed)

    @property
    def free_initial(self) -> np.ndarray:
        """Initial values for free segments only."""
        return self.initial[np.asarray(self.free_segment_indices, dtype=np.int64)]

    def expand(self, free_values: np.ndarray) -> np.ndarray:
        """Expand free and fixed segment values to one lambda per state."""
        free = np.asarray(free_values, dtype=np.float64)
        if free.size != len(self.free_segment_indices):
            raise ValueError(
                f"Expected {len(self.free_segment_indices)} free lambda values, "
                f"received {free.size}."
            )
        segment_values = self.initial.copy()
        for parameter, segment_index in zip(free, self.free_segment_indices, strict=True):
            segment_values[segment_index] = parameter
        expanded = np.empty(self.number_states, dtype=np.float64)
        offset = 0
        for width, value in zip(self.widths, segment_values, strict=True):
            expanded[offset : offset + width] = value
            offset += width
        return expanded


@dataclass(frozen=True)
class PSMCPlusNativeFit:
    """Complete numerical state returned by a native PSMC+ fit."""

    sequences: tuple[PSMCPlusSequence, ...]
    boundaries: np.ndarray
    lambda_values: np.ndarray
    theta: float
    rho: float
    likelihoods: np.ndarray
    final_log_likelihood: float
    optimization_success: tuple[bool, ...]
    optimization_messages: tuple[str, ...]
    optimization_evaluations: tuple[int, ...]
    iteration_lambda_values: tuple[np.ndarray, ...]
    iteration_rho: tuple[float, ...]

    @property
    def number_iterations(self) -> int:
        """Number of completed expectation/maximization iterations."""
        return int(self.likelihoods.size)

    @property
    def likelihood_change(self) -> float:
        """Improvement from the final recorded E step to the final parameters."""
        return float(self.final_log_likelihood - self.likelihoods[-1])


@dataclass(frozen=True)
class PSMCPlusNativeDecode:
    """Native state posterior and marginal-recombination decoding result."""

    sequence: PSMCPlusSequence
    boundaries: np.ndarray
    lambda_values: np.ndarray
    theta: float
    rho: float
    positions: np.ndarray
    posterior: np.ndarray
    log_likelihood: float
    marginal_positions: np.ndarray
    marginal_recombination: np.ndarray
    corrected_marginal_recombination: np.ndarray


def parse_psmcplus_segments(
    pattern: str | None,
    number_states: int,
    initial: Sequence[float] | str | None,
) -> PSMCPlusSegmentLayout:
    """Parse PSMC+'s ``count*width`` and fixed ``width*0`` grammar."""
    if number_states < 2:
        raise ValueError("number_states must be at least two.")
    text = pattern or f"{number_states}*1"
    widths: list[int] = []
    fixed: list[bool] = []
    for token in text.split(","):
        parts = token.strip().split("*")
        if len(parts) != 2:
            raise ValueError(f"Invalid PSMC+ lambda segment token: {token!r}.")
        try:
            count, width = (int(value) for value in parts)
        except ValueError as error:
            raise ValueError(f"Invalid PSMC+ lambda segment token: {token!r}.") from error
        if count < 1 or width < 0:
            raise ValueError(
                "PSMC+ lambda segment counts must be positive and widths non-negative."
            )
        if width == 0:
            widths.append(count)
            fixed.append(True)
        else:
            widths.extend([width] * count)
            fixed.extend([False] * count)
    if sum(widths) != number_states:
        raise ValueError(
            f"PSMC+ lambda segment pattern expands to {sum(widths)} states, not {number_states}."
        )

    if initial is None:
        initial_values = np.ones(len(widths), dtype=np.float64)
    else:
        raw = initial.split(",") if isinstance(initial, str) else initial
        try:
            initial_values = np.asarray([float(value) for value in raw], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("PSMC+ lambda initial values must be numeric.") from error
    if initial_values.size != len(widths):
        raise ValueError(
            f"PSMC+ lambda segment pattern requires {len(widths)} initial values, "
            f"received {initial_values.size}."
        )
    if np.any(~np.isfinite(initial_values)) or np.any(initial_values <= 0):
        raise ValueError("PSMC+ lambda initial values must be finite and positive.")
    return PSMCPlusSegmentLayout(tuple(widths), tuple(fixed), initial_values)


def _mutation_factor_sequence(sequence: PSMCPlusSequence) -> np.ndarray:
    factors = sequence.mutation_factors.copy()
    if factors[0] == 0.0:
        if factors.size < 2:
            raise ValueError("A constant zero PSMC+ mutation map has no valid emission rate.")
        factors[0] = 0.5 * (factors[0] + factors[1])
    return factors[sequence.mutation_indices]


def _empirical_or_supplied_rates(
    sequences: Sequence[PSMCPlusSequence],
    scaled_mutation_rate: float | str | None,
    scaled_recombination_rate: float | None,
    mutation_recombination_ratio: float,
) -> tuple[float, float]:
    total_length = sum(sequence.sequence_length for sequence in sequences)
    total_heterozygotes = sum(sequence.number_heterozygotes for sequence in sequences)
    total_masks = sum(sequence.number_masked_bases for sequence in sequences)
    if scaled_mutation_rate is None or scaled_mutation_rate == "empirical":
        theta = total_heterozygotes / (total_length - total_masks)
    else:
        try:
            theta = float(scaled_mutation_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("scaled_mutation_rate must be positive or 'empirical'.") from error
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("scaled_mutation_rate must be positive.")
    rho = (
        float(scaled_recombination_rate)
        if scaled_recombination_rate is not None
        else theta / mutation_recombination_ratio
    )
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("scaled_recombination_rate must be positive.")
    return theta, rho


def _model_matrices(
    boundaries: np.ndarray,
    lambda_values: np.ndarray,
    rho_per_bin: float,
    recombination_factors: np.ndarray,
    *,
    midpoint_transitions: bool,
    nonexponential_recombination: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_times = psmcplus_expected_times(
        boundaries,
        lambda_values,
        midpoint_transitions,
    )
    conditional = psmcplus_conditional_transition(
        boundaries,
        lambda_values,
        expected_times,
    )
    transitions = psmcplus_transition_stack(
        conditional,
        expected_times,
        rho_per_bin,
        recombination_factors,
        nonexponential_recombination,
    )
    return expected_times, conditional, transitions


def _downsample_recombination_factors(
    sequences: Sequence[PSMCPlusSequence],
    target: int,
) -> tuple[PSMCPlusSequence, ...]:
    """Share at most ``target`` representative local-rate classes across inputs.

    The original helper divides by zero when fewer distinct rates than the
    requested target are present and can create an empty final group. Native
    execution retains all rates in that case and uses non-empty balanced groups
    when actual compression is required.
    """
    if target < 1:
        raise ValueError("recombination map downsample target must be positive.")
    all_factors = np.sort(
        np.concatenate([sequence.recombination_factors for sequence in sequences])
    )
    if all_factors.size <= target:
        return tuple(sequences)
    window_size = all_factors.size // target
    groups = [
        all_factors[start : start + window_size]
        for start in range(0, all_factors.size, window_size)
    ]
    representatives = np.asarray([float(group.mean()) for group in groups], dtype=np.float64)
    downsampled: list[PSMCPlusSequence] = []
    for sequence in sequences:
        nearest = np.argmin(
            np.abs(sequence.recombination_factors[:, np.newaxis] - representatives),
            axis=1,
        )
        downsampled.append(
            dataclass_replace(
                sequence,
                recombination_indices=nearest[sequence.recombination_indices].astype(
                    np.int64,
                    copy=False,
                ),
                recombination_factors=representatives,
            )
        )
    return tuple(downsampled)


def _expectation(
    sequences: Sequence[PSMCPlusSequence],
    boundaries: np.ndarray,
    lambda_values: np.ndarray,
    theta: float,
    rho_per_bin: float,
    initial_distribution: np.ndarray,
    *,
    midpoint_transitions: bool,
    midpoint_emissions: bool,
    nonexponential_recombination: bool,
    cores: int,
    upstream_origin_rate_index: bool,
) -> tuple[np.ndarray, float, tuple[np.ndarray, ...]]:
    number_states = lambda_values.size
    evidence = np.zeros((number_states, number_states), dtype=np.float64)
    log_likelihood = 0.0
    midpoints = psmcplus_emission_midpoints(boundaries, midpoint_emissions)

    def one_sequence(sequence: PSMCPlusSequence) -> tuple[np.ndarray, float]:
        _, _, transitions = _model_matrices(
            boundaries,
            lambda_values,
            rho_per_bin,
            sequence.recombination_factors,
            midpoint_transitions=midpoint_transitions,
            nonexponential_recombination=nonexponential_recombination,
        )
        mutation_factors = _mutation_factor_sequence(sequence)
        forward, scales = psmcplus_forward(
            sequence.heterozygotes,
            sequence.masked_bases,
            mutation_factors,
            sequence.recombination_indices,
            transitions,
            initial_distribution,
            sequence.bin_size,
            theta,
            midpoints,
        )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise FloatingPointError("PSMC+ encountered a zero or invalid forward scale.")
        backward = psmcplus_backward(
            sequence.heterozygotes,
            sequence.masked_bases,
            mutation_factors,
            sequence.recombination_indices,
            transitions,
            scales,
            sequence.bin_size,
            theta,
            midpoints,
        )
        counts = psmcplus_expected_transition_counts(
            sequence.heterozygotes,
            sequence.masked_bases,
            mutation_factors,
            sequence.recombination_indices,
            transitions,
            forward,
            backward,
            sequence.bin_size,
            theta,
            midpoints,
            upstream_origin_rate_index,
        )
        return counts, float(np.log(scales).sum())

    if cores > 1 and len(sequences) > 1:
        with ThreadPoolExecutor(max_workers=min(cores, len(sequences))) as executor:
            results = list(executor.map(one_sequence, sequences))
    else:
        results = [one_sequence(sequence) for sequence in sequences]
    evidence_by_sequence: list[np.ndarray] = []
    for sequence_evidence, sequence_likelihood in results:
        evidence_by_sequence.append(sequence_evidence)
        evidence += sequence_evidence.sum(axis=0)
        log_likelihood += sequence_likelihood
    return evidence, log_likelihood, tuple(evidence_by_sequence)


def _log_transition_objective(evidence: np.ndarray, transition: np.ndarray) -> float:
    if np.any(~np.isfinite(transition)) or np.any(transition < 0.0) or np.any(transition > 1.0):
        return np.inf
    positive = evidence > 0.0
    if np.any(transition[positive] <= 0.0):
        return np.inf
    return -float(np.sum(evidence[positive] * np.log(transition[positive])))


def _final_likelihood(
    sequences: Sequence[PSMCPlusSequence],
    boundaries: np.ndarray,
    lambda_values: np.ndarray,
    theta: float,
    rho_per_bin: float,
    initial_distribution: np.ndarray,
    *,
    midpoint_transitions: bool,
    midpoint_emissions: bool,
    nonexponential_recombination: bool,
    cores: int,
) -> float:
    midpoints = psmcplus_emission_midpoints(boundaries, midpoint_emissions)

    def one_sequence(sequence: PSMCPlusSequence) -> float:
        _, _, transitions = _model_matrices(
            boundaries,
            lambda_values,
            rho_per_bin,
            sequence.recombination_factors,
            midpoint_transitions=midpoint_transitions,
            nonexponential_recombination=nonexponential_recombination,
        )
        _, scales = psmcplus_forward(
            sequence.heterozygotes,
            sequence.masked_bases,
            _mutation_factor_sequence(sequence),
            sequence.recombination_indices,
            transitions,
            initial_distribution,
            sequence.bin_size,
            theta,
            midpoints,
        )
        return float(np.log(scales).sum())

    if cores > 1 and len(sequences) > 1:
        with ThreadPoolExecutor(max_workers=min(cores, len(sequences))) as executor:
            return float(sum(executor.map(one_sequence, sequences)))
    return float(sum(one_sequence(sequence) for sequence in sequences))


def fit_psmcplus_native(
    input_paths: Sequence[str | Path],
    *,
    mutation_map_paths: Sequence[str | Path] | None = None,
    recombination_map_paths: Sequence[str | Path] | None = None,
    recombination_map_downsamples: int = 200,
    number_states: int = 50,
    spread_1: float = 0.1,
    spread_2: float = 50.0,
    bin_size: int = 100,
    scaled_mutation_rate: float | str | None = None,
    scaled_recombination_rate: float | None = None,
    estimate_rho: bool = True,
    mutation_recombination_ratio: float = 1.5,
    lambda_initial: Sequence[float] | str | None = None,
    lambda_segments: str | None = None,
    lambda_lower_bound: float = 0.1,
    lambda_upper_bound: float = 50.0,
    iterations: int = 20,
    likelihood_threshold: float = 1.0,
    parameter_tolerance: float = 1e-4,
    objective_tolerance: float = 1e-4,
    optimization_method: str = "Powell",
    midpoint_transitions: bool = False,
    midpoint_emissions: bool = False,
    final_time_factor: float | None = None,
    nonexponential_recombination: bool = False,
    cores: int = 1,
) -> PSMCPlusNativeFit:
    """Fit the panmictic PSMC+ model with the independent native engine."""
    inputs = tuple(Path(path).expanduser().resolve() for path in input_paths)
    if not inputs:
        raise ValueError("Native PSMC+ requires at least one multihetsep input.")
    mutation_maps = tuple(Path(path).expanduser().resolve() for path in (mutation_map_paths or ()))
    recombination_maps = tuple(
        Path(path).expanduser().resolve() for path in (recombination_map_paths or ())
    )
    if mutation_maps and len(mutation_maps) != len(inputs):
        raise ValueError("Native PSMC+ requires one mutation map per input.")
    if recombination_maps and len(recombination_maps) != len(inputs):
        raise ValueError("Native PSMC+ requires one recombination map per input.")
    if iterations < 1:
        raise ValueError("iterations must be at least one.")
    if likelihood_threshold < 0:
        raise ValueError("likelihood_threshold cannot be negative.")
    if cores < 1:
        raise ValueError("cores must be positive.")
    layout = parse_psmcplus_segments(lambda_segments, number_states, lambda_initial)
    sequences = tuple(
        prepare_psmcplus_sequence(
            path,
            bin_size=bin_size,
            mutation_map_path=mutation_maps[index] if mutation_maps else None,
            recombination_map_path=recombination_maps[index] if recombination_maps else None,
        )
        for index, path in enumerate(inputs)
    )
    if recombination_maps:
        sequences = _downsample_recombination_factors(
            sequences,
            recombination_map_downsamples,
        )
    theta, rho = _empirical_or_supplied_rates(
        sequences,
        scaled_mutation_rate,
        scaled_recombination_rate,
        mutation_recombination_ratio,
    )
    rho_per_bin = rho * bin_size
    boundaries = psmcplus_time_boundaries(
        number_states,
        spread_1,
        spread_2,
        -1.0 if final_time_factor is None else final_time_factor,
    )
    lambda_values = layout.expand(layout.free_initial)
    _, _, initial_transition_stack = _model_matrices(
        boundaries,
        lambda_values,
        rho_per_bin,
        np.ones(1, dtype=np.float64),
        midpoint_transitions=midpoint_transitions,
        nonexponential_recombination=nonexponential_recombination,
    )
    initial_distribution = psmcplus_stationary_distribution(initial_transition_stack[0])

    likelihoods: list[float] = []
    successes: list[bool] = []
    messages: list[str] = []
    evaluations: list[int] = []
    iteration_lambda_values: list[np.ndarray] = []
    iteration_rho: list[float] = []
    old_likelihood = 0.0
    likelihood_change = np.inf
    local_rate_estimation = estimate_rho and any(
        sequence.recombination_factors.size > 1
        or not np.allclose(sequence.recombination_factors, 1.0)
        for sequence in sequences
    )
    for _iteration in range(iterations):
        if likelihood_change <= likelihood_threshold:
            break
        evidence, log_likelihood, evidence_by_sequence = _expectation(
            sequences,
            boundaries,
            lambda_values,
            theta,
            rho_per_bin,
            initial_distribution,
            midpoint_transitions=midpoint_transitions,
            midpoint_emissions=midpoint_emissions,
            nonexponential_recombination=nonexponential_recombination,
            cores=cores,
            upstream_origin_rate_index=not local_rate_estimation,
        )
        likelihoods.append(log_likelihood)
        likelihood_change = (
            -log_likelihood if len(likelihoods) == 1 else log_likelihood - old_likelihood
        )
        old_likelihood = log_likelihood

        free_initial = np.asarray(
            [
                lambda_values[sum(layout.widths[:segment_index])]
                for segment_index in layout.free_segment_indices
            ],
            dtype=np.float64,
        )
        optimize_rho = estimate_rho
        initial_parameters = np.append(free_initial, rho_per_bin) if optimize_rho else free_initial
        lambda_bounds = [(lambda_lower_bound, lambda_upper_bound)] * free_initial.size
        if lambda_bounds:
            lambda_bounds[-1] = (lambda_lower_bound, min(lambda_upper_bound, 10.0))
        bounds = [*lambda_bounds, *([(1e-16, 0.5)] if optimize_rho else [])]

        def objective(parameters: np.ndarray) -> float:
            candidate_free = parameters[:-1] if optimize_rho else parameters
            candidate_rho = float(parameters[-1]) if optimize_rho else rho_per_bin
            candidate_lambda = layout.expand(candidate_free)
            if local_rate_estimation:
                objective_value = 0.0
                for sequence, sequence_evidence in zip(
                    sequences,
                    evidence_by_sequence,
                    strict=True,
                ):
                    _, _, candidate_stack = _model_matrices(
                        boundaries,
                        candidate_lambda,
                        candidate_rho,
                        sequence.recombination_factors,
                        midpoint_transitions=midpoint_transitions,
                        nonexponential_recombination=nonexponential_recombination,
                    )
                    for rate_index in range(candidate_stack.shape[0]):
                        objective_value += _log_transition_objective(
                            sequence_evidence[rate_index],
                            candidate_stack[rate_index],
                        )
                return objective_value
            _, _, candidate_stack = _model_matrices(
                boundaries,
                candidate_lambda,
                candidate_rho,
                np.ones(1, dtype=np.float64),
                midpoint_transitions=midpoint_transitions,
                nonexponential_recombination=nonexponential_recombination,
            )
            return _log_transition_objective(evidence, candidate_stack[0])

        if initial_parameters.size:
            optimizer_options = None
            if not optimize_rho:
                optimizer_options = {
                    "xtol": parameter_tolerance,
                    "ftol": objective_tolerance,
                }
            result = minimize(
                objective,
                initial_parameters,
                method="Powell" if optimize_rho else optimization_method,
                bounds=bounds,
                options=optimizer_options,
            )
            candidate_free = result.x[:-1] if optimize_rho else result.x
            if optimize_rho:
                rho_per_bin = float(result.x[-1])
            lambda_values = layout.expand(candidate_free)
            successes.append(bool(result.success))
            messages.append(str(result.message))
            evaluations.append(int(result.nfev))
        else:
            successes.append(True)
            messages.append("All lambda parameters fixed; no maximization required.")
            evaluations.append(0)
        iteration_lambda_values.append(lambda_values.copy())
        iteration_rho.append(rho_per_bin / bin_size)

    final_log_likelihood = _final_likelihood(
        sequences,
        boundaries,
        lambda_values,
        theta,
        rho_per_bin,
        initial_distribution,
        midpoint_transitions=midpoint_transitions,
        midpoint_emissions=midpoint_emissions,
        nonexponential_recombination=nonexponential_recombination,
        cores=cores,
    )
    return PSMCPlusNativeFit(
        sequences=sequences,
        boundaries=boundaries,
        lambda_values=lambda_values,
        theta=theta,
        rho=rho_per_bin / bin_size,
        likelihoods=np.asarray(likelihoods, dtype=np.float64),
        final_log_likelihood=final_log_likelihood,
        optimization_success=tuple(successes),
        optimization_messages=tuple(messages),
        optimization_evaluations=tuple(evaluations),
        iteration_lambda_values=tuple(iteration_lambda_values),
        iteration_rho=tuple(iteration_rho),
    )


def decode_psmcplus_native(
    input_path: str | Path,
    *,
    mutation_map_path: str | Path | None = None,
    recombination_map_path: str | Path | None = None,
    recombination_map_downsamples: int = 200,
    number_states: int = 50,
    spread_1: float = 0.1,
    spread_2: float = 50.0,
    bin_size: int = 100,
    scaled_mutation_rate: float | str | None = None,
    scaled_recombination_rate: float | None = None,
    mutation_recombination_ratio: float = 1.5,
    lambda_initial: Sequence[float] | str | None = None,
    lambda_segments: str | None = None,
    downsample: int = 10,
    midpoint_transitions: bool = False,
    midpoint_emissions: bool = False,
    final_time_factor: float | None = None,
    nonexponential_recombination: bool = False,
) -> PSMCPlusNativeDecode:
    """Decode one sequence with a fixed panmictic PSMC+ demographic model."""
    if downsample < 1:
        raise ValueError("downsample must be positive.")
    layout = parse_psmcplus_segments(lambda_segments, number_states, lambda_initial)
    sequence = prepare_psmcplus_sequence(
        input_path,
        bin_size=bin_size,
        mutation_map_path=mutation_map_path,
        recombination_map_path=recombination_map_path,
    )
    if recombination_map_path is not None:
        sequence = _downsample_recombination_factors(
            (sequence,),
            recombination_map_downsamples,
        )[0]
    theta, rho = _empirical_or_supplied_rates(
        (sequence,),
        scaled_mutation_rate,
        scaled_recombination_rate,
        mutation_recombination_ratio,
    )
    rho_per_bin = rho * bin_size
    boundaries = psmcplus_time_boundaries(
        number_states,
        spread_1,
        spread_2,
        -1.0 if final_time_factor is None else final_time_factor,
    )
    lambda_values = layout.expand(layout.free_initial)
    _, _, transitions = _model_matrices(
        boundaries,
        lambda_values,
        rho_per_bin,
        sequence.recombination_factors,
        midpoint_transitions=midpoint_transitions,
        nonexponential_recombination=nonexponential_recombination,
    )
    _, _, baseline = _model_matrices(
        boundaries,
        lambda_values,
        rho_per_bin,
        np.ones(1, dtype=np.float64),
        midpoint_transitions=midpoint_transitions,
        nonexponential_recombination=nonexponential_recombination,
    )
    initial_distribution = psmcplus_stationary_distribution(baseline[0])
    mutation_factors = _mutation_factor_sequence(sequence)
    midpoints = psmcplus_emission_midpoints(boundaries, midpoint_emissions)
    forward, scales = psmcplus_forward(
        sequence.heterozygotes,
        sequence.masked_bases,
        mutation_factors,
        sequence.recombination_indices,
        transitions,
        initial_distribution,
        bin_size,
        theta,
        midpoints,
    )
    backward = psmcplus_backward(
        sequence.heterozygotes,
        sequence.masked_bases,
        mutation_factors,
        sequence.recombination_indices,
        transitions,
        scales,
        bin_size,
        theta,
        midpoints,
    )
    posterior = forward * backward
    posterior /= posterior.sum(axis=1, keepdims=True)
    corrected_marginal = psmcplus_marginal_recombination(
        sequence.heterozygotes,
        sequence.masked_bases,
        mutation_factors,
        sequence.recombination_indices,
        transitions,
        forward,
        backward,
        bin_size,
        theta,
        midpoints,
    )
    upstream_marginal = psmcplus_marginal_recombination(
        sequence.heterozygotes,
        sequence.masked_bases,
        np.ones_like(mutation_factors),
        sequence.recombination_indices,
        transitions,
        forward,
        backward,
        bin_size,
        theta,
        midpoints,
    )
    retained = sequence.number_bins // downsample
    selected = np.arange(retained, dtype=np.int64) * downsample
    return PSMCPlusNativeDecode(
        sequence=sequence,
        boundaries=boundaries,
        lambda_values=lambda_values,
        theta=theta,
        rho=rho,
        positions=selected * bin_size,
        posterior=posterior[selected],
        log_likelihood=float(np.log(scales).sum()),
        marginal_positions=np.arange(sequence.number_bins - 1, dtype=np.int64) * bin_size,
        marginal_recombination=upstream_marginal,
        corrected_marginal_recombination=corrected_marginal,
    )


__all__ = [
    "PSMCPlusNativeFit",
    "PSMCPlusNativeDecode",
    "PSMCPlusSegmentLayout",
    "decode_psmcplus_native",
    "fit_psmcplus_native",
    "parse_psmcplus_segments",
]
