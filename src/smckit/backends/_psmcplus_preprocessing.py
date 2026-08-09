"""Independent, memory-efficient preprocessing for native PSMC+.

The preserved implementation expands every input to one value per base before
binning.  The native path computes the same bin counts and interval-weighted
rate averages directly, so its memory use scales with bins and annotations
rather than chromosome length.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAP_LENGTH_TOLERANCE = 5_000_000


@dataclass(frozen=True)
class PSMCPlusSequence:
    """Dense binned observations and local-rate factors for one sequence."""

    source_path: str
    bin_size: int
    sequence_length: int
    number_heterozygotes: int
    number_masked_bases: int
    heterozygotes: np.ndarray
    masked_bases: np.ndarray
    mutation_indices: np.ndarray
    mutation_factors: np.ndarray
    recombination_indices: np.ndarray
    recombination_factors: np.ndarray

    @property
    def number_bins(self) -> int:
        """Number of complete bins retained by PSMC+."""
        return int(self.heterozygotes.size)

    @property
    def maximum_heterozygotes(self) -> int:
        """Largest heterozygote count in a retained bin."""
        return int(self.heterozygotes.max(initial=0))

    def mutation_factor_sequence(self) -> np.ndarray:
        """Expand the compact mutation-factor representation by bin."""
        return self.mutation_factors[self.mutation_indices]

    def recombination_factor_sequence(self) -> np.ndarray:
        """Expand the compact recombination-factor representation by bin."""
        return self.recombination_factors[self.recombination_indices]


def _parse_multihetsep(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    chromosome: str | None = None
    positions: list[int] = []
    called: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or (row[0].lstrip().startswith("#")):
                continue
            if len(row) < 3:
                raise ValueError(
                    f"{path}:{line_number}: expected at least three tab-separated columns."
                )
            if chromosome is None:
                chromosome = row[0]
            elif row[0] != chromosome:
                raise ValueError(f"{path}:{line_number}: multiple chromosomes are not supported.")
            try:
                position = int(row[1])
                callable_gap = int(row[2])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: position and callable count must be integers."
                ) from error
            if position < 1:
                raise ValueError(
                    f"{path}:{line_number}: positions must be one-based and positive."
                )
            if positions and position <= positions[-1]:
                raise ValueError(f"{path}:{line_number}: positions must be strictly increasing.")
            previous = positions[-1] if positions else 0
            if callable_gap < 1 or callable_gap > position - previous:
                raise ValueError(
                    f"{path}:{line_number}: callable count must be between one and the "
                    "distance from the previous heterozygote."
                )
            positions.append(position)
            called.append(callable_gap)
    if not positions or chromosome is None:
        raise ValueError(f"{path}: multihetsep input contains no observations.")
    return chromosome, np.asarray(positions, dtype=np.int64), np.asarray(called, dtype=np.int64)


def _add_interval_counts(counts: np.ndarray, start: int, stop: int, bin_size: int) -> None:
    """Add half-open interval overlap counts without allocating per-base arrays."""
    retained_stop = min(stop, counts.size * bin_size)
    if start >= retained_stop:
        return
    start_bin = start // bin_size
    stop_bin = (retained_stop - 1) // bin_size
    if start_bin == stop_bin:
        counts[start_bin] += retained_stop - start
        return
    counts[start_bin] += (start_bin + 1) * bin_size - start
    if stop_bin > start_bin + 1:
        counts[start_bin + 1 : stop_bin] += bin_size
    counts[stop_bin] += retained_stop - stop_bin * bin_size


def _bin_observations(
    positions: np.ndarray,
    called: np.ndarray,
    bin_size: int,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    zero_based = positions - 1
    sequence_length = int(zero_based[-1] + 2 * bin_size)
    number_bins = sequence_length // bin_size
    heterozygotes = np.bincount(
        zero_based // bin_size,
        minlength=number_bins,
    )[:number_bins].astype(np.int64, copy=False)
    masked_bases = np.zeros(number_bins, dtype=np.int64)
    previous_position = 0
    for position, callable_gap in zip(positions, called, strict=True):
        missing = int(position - callable_gap - previous_position)
        if missing:
            _add_interval_counts(
                masked_bases,
                previous_position,
                previous_position + missing,
                bin_size,
            )
        previous_position = int(position)
    if np.any(heterozygotes + masked_bases > bin_size):
        raise ValueError("Some input bases are labelled as both heterozygous and masked.")
    return sequence_length, heterozygotes, masked_bases, int(masked_bases.sum())


def _parse_rate_map(path: Path, chromosome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[int] = []
    stops: list[int] = []
    values: list[float] = []
    previous_stop = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected chromosome, start, end, and factor."
                )
            if row[0] != chromosome:
                raise ValueError(
                    f"{path}:{line_number}: chromosome {row[0]!r} does not match {chromosome!r}."
                )
            try:
                start, stop = int(row[1]), int(row[2])
                value = float(row[3])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid rate-map value.") from error
            if start != previous_stop:
                raise ValueError(
                    f"{path}:{line_number}: rate maps must be contiguous from coordinate zero."
                )
            if stop <= start:
                raise ValueError(f"{path}:{line_number}: interval end must exceed its start.")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{path}:{line_number}: factors must be finite and non-negative.")
            starts.append(start)
            stops.append(stop)
            values.append(value)
            previous_stop = stop
    if not values:
        raise ValueError(f"{path}: rate map contains no intervals.")
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(stops, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def _bin_rate_map(
    path: Path | None,
    *,
    chromosome: str,
    sequence_length: int,
    number_bins: int,
    bin_size: int,
    decimate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.zeros(number_bins, dtype=np.int64), np.ones(1, dtype=np.float64)
    starts, stops, values = _parse_rate_map(path, chromosome)
    map_length = int(stops[-1])
    if sequence_length - map_length > _MAP_LENGTH_TOLERANCE:
        raise ValueError(
            f"{path}: map is more than {_MAP_LENGTH_TOLERANCE} bases shorter than the sequence."
        )

    # PSMC+ expands the map to one float per base and calls ``np.average`` on
    # each bin.  Computing a weighted sum is mathematically equivalent, but it
    # can round on the other side of the upstream two-decimal truncation (for
    # example, a bin filled with 0.4 averages to 0.3999999999999999).  Retain
    # the upstream reduction order while keeping memory proportional to bins
    # and map boundaries rather than chromosome length.
    full_bin_values = np.full(number_bins, np.nan, dtype=np.float64)
    partial_runs: dict[int, list[tuple[int, int, float]]] = {}
    covered = np.zeros(number_bins, dtype=np.int64)
    retained_length = number_bins * bin_size

    def record_interval(start: int, stop: int, value: float) -> None:
        if start >= stop:
            return
        first_full_bin = (start + bin_size - 1) // bin_size
        after_last_full_bin = stop // bin_size
        if first_full_bin >= after_last_full_bin:
            first_bin = start // bin_size
            last_bin = (stop - 1) // bin_size
            for bin_index in range(first_bin, last_bin + 1):
                local_start = max(start, bin_index * bin_size) - bin_index * bin_size
                local_stop = min(stop, (bin_index + 1) * bin_size) - bin_index * bin_size
                partial_runs.setdefault(bin_index, []).append((local_start, local_stop, value))
                covered[bin_index] += local_stop - local_start
            return

        prefix_stop = first_full_bin * bin_size
        if start < prefix_stop:
            bin_index = start // bin_size
            partial_runs.setdefault(bin_index, []).append(
                (start - bin_index * bin_size, bin_size, value)
            )
            covered[bin_index] += prefix_stop - start

        full_bin_values[first_full_bin:after_last_full_bin] = value
        covered[first_full_bin:after_last_full_bin] = bin_size

        suffix_start = after_last_full_bin * bin_size
        if suffix_start < stop:
            bin_index = suffix_start // bin_size
            partial_runs.setdefault(bin_index, []).append((0, stop - suffix_start, value))
            covered[bin_index] += stop - suffix_start

    for start, stop, value in zip(starts, stops, values, strict=True):
        interval_stop = min(int(stop), retained_length)
        interval_start = int(start)
        if interval_start >= interval_stop:
            break
        record_interval(interval_start, interval_stop, float(value))

    if map_length < retained_length:
        record_interval(map_length, retained_length, 1.0)
    if np.any(covered != bin_size):
        raise ValueError(f"{path}: rate map does not cover all retained sequence bins.")

    factors_by_bin = np.empty(number_bins, dtype=np.float64)
    constant_average_cache: dict[float, float] = {}
    for bin_index in range(number_bins):
        value = full_bin_values[bin_index]
        if np.isfinite(value):
            numeric_value = float(value)
            if numeric_value not in constant_average_cache:
                constant_average_cache[numeric_value] = float(
                    np.average(np.full(bin_size, numeric_value, dtype=np.float64))
                )
            factors_by_bin[bin_index] = constant_average_cache[numeric_value]
            continue

        expanded_bin = np.empty(bin_size, dtype=np.float64)
        for local_start, local_stop, run_value in partial_runs[bin_index]:
            expanded_bin[local_start:local_stop] = run_value
        factors_by_bin[bin_index] = float(np.average(expanded_bin))
    if decimate and not np.all(factors_by_bin == factors_by_bin[0]):
        factors_by_bin = np.trunc(100.0 * factors_by_bin) / 100.0
        factors_by_bin[factors_by_bin == 0.0] = 1e-5
    factors, indices = np.unique(factors_by_bin, return_inverse=True)
    return indices.astype(np.int64, copy=False), factors


def prepare_psmcplus_sequence(
    input_path: str | Path,
    *,
    bin_size: int,
    mutation_map_path: str | Path | None = None,
    recombination_map_path: str | Path | None = None,
) -> PSMCPlusSequence:
    """Parse and bin one PSMC+ input without constructing chromosome-sized arrays."""
    if bin_size < 1:
        raise ValueError("bin_size must be positive.")
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PSMC+ multihetsep input does not exist: {source}")
    chromosome, positions, called = _parse_multihetsep(source)
    sequence_length, heterozygotes, masked_bases, number_masked = _bin_observations(
        positions,
        called,
        bin_size,
    )
    mutation_map = (
        Path(mutation_map_path).expanduser().resolve() if mutation_map_path is not None else None
    )
    recombination_map = (
        Path(recombination_map_path).expanduser().resolve()
        if recombination_map_path is not None
        else None
    )
    for map_path, label in (
        (mutation_map, "mutation map"),
        (recombination_map, "recombination map"),
    ):
        if map_path is not None and not map_path.is_file():
            raise FileNotFoundError(f"PSMC+ {label} does not exist: {map_path}")
    mutation_indices, mutation_factors = _bin_rate_map(
        mutation_map,
        chromosome=chromosome,
        sequence_length=sequence_length,
        number_bins=heterozygotes.size,
        bin_size=bin_size,
        decimate=False,
    )
    recombination_indices, recombination_factors = _bin_rate_map(
        recombination_map,
        chromosome=chromosome,
        sequence_length=sequence_length,
        number_bins=heterozygotes.size,
        bin_size=bin_size,
        decimate=True,
    )
    return PSMCPlusSequence(
        source_path=str(source),
        bin_size=bin_size,
        sequence_length=sequence_length,
        number_heterozygotes=int(positions.size),
        number_masked_bases=number_masked,
        heterozygotes=heterozygotes,
        masked_bases=masked_bases,
        mutation_indices=mutation_indices,
        mutation_factors=mutation_factors,
        recombination_indices=recombination_indices,
        recombination_factors=recombination_factors,
    )
