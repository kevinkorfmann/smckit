"""Read diCal2 input formats (.param, .demo, .config) and VCF.

diCal2 uses three configuration files plus sequence data:

- ``.param`` — mutation rate theta, recombination rate rho, and the
  allele mutation matrix.
- ``.demo`` — demographic model with epoch boundaries, population sizes,
  migration matrices, and pulse events.
- ``.config`` — sample configuration: sequence length, number of alleles,
  number of populations, and population assignments for each haplotype.

Reference: Steinrücken, Kamm & Song (2019), PNAS 116(34):17115–17120.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smckit._core import SmcData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures for diCal2 input
# ---------------------------------------------------------------------------


@dataclass
class DiCal2Params:
    """Mutation and recombination parameters."""

    theta: float
    rho: float
    mutation_matrix: np.ndarray  # (n_alleles, n_alleles)


@dataclass
class DiCal2Epoch:
    """A single epoch in the demographic model."""

    start: float
    end: float
    partition: list[list[int]]  # partition[ancient_deme] = [present_demes]
    pop_sizes: np.ndarray | None  # (n_ancient_demes,) — None for pulse epochs
    migration_matrix: np.ndarray | None  # (n_ancient_demes, n_ancient_demes)
    pulse_migration: np.ndarray | None  # (n_ancient_demes, n_ancient_demes)
    growth_rates: np.ndarray | None  # (n_ancient_demes,)
    pop_size_param_ids: list[int | None] | None = None
    migration_param_ids: list[list[int | None]] | None = None
    growth_rate_param_ids: list[int | None] | None = None
    pulse_migration_param_ids: list[list[int | None]] | None = None


@dataclass
class DiCal2Demo:
    """Complete demographic model."""

    epoch_boundaries: np.ndarray  # time boundaries
    epochs: list[DiCal2Epoch]
    n_present_demes: int
    boundary_param_ids: list[int | None] | None = None


@dataclass
class DiCal2Config:
    """Sample configuration."""

    seq_length: int
    n_alleles: int
    n_populations: int
    haplotype_populations: list[int]  # population index for each haplotype
    haplotypes_to_include: list[bool]
    haplotype_multiplicities: np.ndarray  # (n_haplotypes, n_populations)
    sample_sizes: np.ndarray  # (n_populations,) — samples per pop


@dataclass
class DiCal2Contig:
    """One independent diCal2 sequence/VCF likelihood contribution."""

    sequences: np.ndarray
    seg_positions: np.ndarray | None = None
    reference_length: int | None = None
    reference_alleles: np.ndarray | None = None
    source_path: str | None = None
    reference_file: str | None = None
    bed_file: str | None = None
    vcf_offset: int = 0


# ---------------------------------------------------------------------------
# .param file parser
# ---------------------------------------------------------------------------


def _compact_dical2_config(config: DiCal2Config) -> DiCal2Config:
    """Drop excluded haplotypes so config rows match filtered sequence rows."""
    include_mask = np.asarray(config.haplotypes_to_include, dtype=bool)
    multiplicities = np.asarray(config.haplotype_multiplicities, dtype=np.int64)
    compact_multiplicities = multiplicities[include_mask].copy()
    compact_populations = [
        int(pop)
        for pop, include in zip(config.haplotype_populations, include_mask, strict=False)
        if include
    ]
    compact_sample_sizes = (
        compact_multiplicities.sum(axis=0)
        if len(compact_multiplicities)
        else np.zeros(config.n_populations, dtype=np.int64)
    )
    return DiCal2Config(
        seq_length=int(config.seq_length),
        n_alleles=int(config.n_alleles),
        n_populations=int(config.n_populations),
        haplotype_populations=compact_populations,
        haplotypes_to_include=[True] * len(compact_populations),
        haplotype_multiplicities=compact_multiplicities,
        sample_sizes=np.asarray(compact_sample_sizes, dtype=np.int64),
    )


def read_dical2_param(path: str | Path) -> DiCal2Params:
    """Read a diCal2 ``.param`` file.

    Format::

        theta
        rho
        m00  m01  ...
        m10  m11  ...
        ...

    Parameters
    ----------
    path : str or Path
        Path to the ``.param`` file.

    Returns
    -------
    DiCal2Params
    """
    path = Path(path)
    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if len(lines) < 3:
        raise ValueError(f"Expected at least 3 non-comment lines in {path}")

    theta = float(lines[0])
    rho = float(lines[1])

    rows = []
    for ln in lines[2:]:
        rows.append([float(x) for x in ln.split()])
    mutation_matrix = np.array(rows, dtype=np.float64)

    return DiCal2Params(theta=theta, rho=rho, mutation_matrix=mutation_matrix)


# ---------------------------------------------------------------------------
# .demo file parser
# ---------------------------------------------------------------------------


def _parse_partition(text: str) -> list[list[int]]:
    """Parse a partition string like ``{{0},{1,2}}`` into a list of lists.

    Strips the outermost braces and splits on top-level commas separating
    inner ``{...}`` groups.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()

    result: list[list[int]] = []
    depth = 0
    current = ""
    for ch in text:
        if ch == "{":
            depth += 1
            if depth == 1:
                current = ""
                continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                tokens = [t.strip() for t in current.split(",") if t.strip()]
                result.append([int(t) for t in tokens])
                current = ""
                continue
        if depth >= 1:
            current += ch
    return result


_PLACEHOLDER_RE = re.compile(r"\?\d+")


def _parse_value(token: str, default: float = 1.0) -> tuple[float, bool]:
    """Parse a numeric token; ``?N`` is a parameter placeholder.

    Returns
    -------
    value : float
        Numeric value (default if placeholder).
    is_placeholder : bool
        True iff the token was a ``?N`` placeholder.
    """
    token = token.strip()
    if _PLACEHOLDER_RE.fullmatch(token):
        return default, True
    return float(token), False


def read_dical2_demo(
    path: str | Path,
    n_present_demes: int | None = None,
) -> DiCal2Demo:
    """Read a diCal2 ``.demo`` file.

    Supports ``?N`` parameter placeholders (replaced by default values of
    1.0 — these mark parameters that would normally be optimized). Lines
    starting with ``#`` are comments.

    Parameters
    ----------
    path : str or Path
        Path to the ``.demo`` file.
    n_present_demes : int, optional
        Number of present-day demes (inferred from first partition if omitted).

    Returns
    -------
    DiCal2Demo
    """
    path = Path(path)
    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Empty .demo file: {path}")

    idx = 0

    # First line: epoch boundaries [t1, t2, ...]
    boundary_text = lines[idx]
    idx += 1
    boundary_text = boundary_text.strip("[] ")
    boundaries: list[float] = []
    boundary_param_ids: list[int | None] = []
    if boundary_text:
        placeholder_rank = 1
        for tok in boundary_text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            val, is_placeholder = _parse_value(tok, default=float(placeholder_rank))
            boundaries.append(val)
            boundary_param_ids.append(int(tok[1:]) if is_placeholder else None)
            if is_placeholder:
                placeholder_rank += 1

    # Preserve file order; diCal2 demos specify epochs from present to past.
    epoch_bounds = [0.0] + boundaries + [np.inf]
    n_epochs = len(epoch_bounds) - 1

    epochs: list[DiCal2Epoch] = []
    for e in range(n_epochs):
        if idx >= len(lines):
            break

        # Partition line
        partition = _parse_partition(lines[idx])
        idx += 1
        n_ancient = len(partition)

        if n_present_demes is None:
            all_demes = set()
            for group in partition:
                all_demes.update(group)
            n_present_demes = len(all_demes)

        is_pulse = epoch_bounds[e] == epoch_bounds[e + 1]

        # Population sizes (or null for pulse epochs)
        pop_sizes: np.ndarray | None = None
        pop_size_param_ids: list[int | None] | None = None
        if idx < len(lines):
            if lines[idx].lower() == "null":
                idx += 1
            else:
                pop_values = []
                pop_param_ids = []
                for x in lines[idx].split():
                    value, is_placeholder = _parse_value(x, default=1.0)
                    pop_values.append(value)
                    pop_param_ids.append(int(x.strip()[1:]) if is_placeholder else None)
                pop_sizes = np.array(
                    pop_values,
                    dtype=np.float64,
                )
                pop_size_param_ids = pop_param_ids
                idx += 1

        # Pulse migration matrix (or null)
        pulse_migration: np.ndarray | None = None
        pulse_migration_param_ids: list[list[int | None]] | None = None
        if idx < len(lines):
            if lines[idx].lower() == "null":
                idx += 1
            else:
                pulse_rows = []
                pulse_param_rows = []
                # n_ancient rows for the pulse matrix
                for _ in range(n_ancient):
                    if idx < len(lines):
                        row_values = []
                        row_ids = []
                        for value in lines[idx].split():
                            parsed, is_placeholder = _parse_value(value, default=0.0)
                            row_values.append(parsed)
                            row_ids.append(int(value.strip()[1:]) if is_placeholder else None)
                        pulse_rows.append(row_values)
                        pulse_param_rows.append(row_ids)
                        idx += 1
                if pulse_rows:
                    pulse_migration = np.array(pulse_rows, dtype=np.float64)
                    pulse_migration_param_ids = pulse_param_rows

        # Continuous migration matrix
        migration_matrix: np.ndarray | None = None
        migration_param_ids: list[list[int | None]] | None = None
        if idx < len(lines):
            if lines[idx].lower() == "null":
                idx += 1
            else:
                mig_rows = []
                mig_param_rows = []
                for _ in range(n_ancient):
                    if idx < len(lines):
                        row_vals = []
                        row_ids = []
                        for x in lines[idx].split():
                            val, is_placeholder = _parse_value(x, default=0.0)
                            row_vals.append(val)
                            row_ids.append(int(x.strip()[1:]) if is_placeholder else None)
                        mig_rows.append(row_vals)
                        mig_param_rows.append(row_ids)
                        idx += 1
                if mig_rows:
                    migration_matrix = np.array(mig_rows, dtype=np.float64)
                    migration_param_ids = mig_param_rows
                    # Ensure rows sum to 0 (proper rate matrix)
                    for i in range(migration_matrix.shape[0]):
                        off_diag = (
                            migration_matrix[i, :i].sum() + migration_matrix[i, i + 1 :].sum()
                        )
                        migration_matrix[i, i] = -off_diag

        epochs.append(
            DiCal2Epoch(
                start=epoch_bounds[e],
                end=epoch_bounds[e + 1],
                partition=partition,
                pop_sizes=None if is_pulse else pop_sizes,
                pop_size_param_ids=None if is_pulse else pop_size_param_ids,
                migration_matrix=None if is_pulse else migration_matrix,
                migration_param_ids=None if is_pulse else migration_param_ids,
                pulse_migration=pulse_migration,
                growth_rates=None,
                growth_rate_param_ids=None,
                pulse_migration_param_ids=pulse_migration_param_ids,
            )
        )

    return DiCal2Demo(
        epoch_boundaries=np.array(epoch_bounds, dtype=np.float64),
        boundary_param_ids=[None] + boundary_param_ids + [None],
        epochs=epochs,
        n_present_demes=n_present_demes,
    )


# ---------------------------------------------------------------------------
# .rates file parser
# ---------------------------------------------------------------------------


def read_dical2_rates(
    path: str | Path,
    demo: DiCal2Demo,
) -> DiCal2Demo:
    """Read a diCal2 ``.rates`` file and attach growth rates to a demo."""
    path = Path(path)
    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if len(lines) != len(demo.epochs):
        raise ValueError(
            f"Rates file {path} has {len(lines)} data rows, expected {len(demo.epochs)}"
        )

    new_epochs: list[DiCal2Epoch] = []
    for epoch, line in zip(demo.epochs, lines):
        new_epoch = DiCal2Epoch(
            start=epoch.start,
            end=epoch.end,
            partition=[list(group) for group in epoch.partition],
            pop_sizes=None if epoch.pop_sizes is None else epoch.pop_sizes.copy(),
            pop_size_param_ids=(
                None if epoch.pop_size_param_ids is None else list(epoch.pop_size_param_ids)
            ),
            migration_matrix=(
                None if epoch.migration_matrix is None else epoch.migration_matrix.copy()
            ),
            migration_param_ids=(
                None
                if epoch.migration_param_ids is None
                else [list(row) for row in epoch.migration_param_ids]
            ),
            pulse_migration=(
                None if epoch.pulse_migration is None else epoch.pulse_migration.copy()
            ),
            pulse_migration_param_ids=(
                None
                if epoch.pulse_migration_param_ids is None
                else [list(row) for row in epoch.pulse_migration_param_ids]
            ),
            growth_rates=None,
            growth_rate_param_ids=None,
        )
        if epoch.pop_sizes is None:
            new_epochs.append(new_epoch)
            continue

        values = []
        param_ids = []
        tokens = line.split()
        if len(tokens) != len(epoch.partition):
            raise ValueError(
                f"Rates row for epoch [{epoch.start}, {epoch.end}) has {len(tokens)} values, "
                f"expected {len(epoch.partition)}"
            )
        for token in tokens:
            value, is_placeholder = _parse_value(token, default=0.0)
            values.append(value)
            param_ids.append(int(token.strip()[1:]) if is_placeholder else None)
        new_epoch.growth_rates = np.array(values, dtype=np.float64)
        new_epoch.growth_rate_param_ids = param_ids
        new_epochs.append(new_epoch)

    return DiCal2Demo(
        epoch_boundaries=demo.epoch_boundaries.copy(),
        boundary_param_ids=(
            None if demo.boundary_param_ids is None else list(demo.boundary_param_ids)
        ),
        epochs=new_epochs,
        n_present_demes=demo.n_present_demes,
    )


# ---------------------------------------------------------------------------
# .config file parser
# ---------------------------------------------------------------------------


def read_dical2_config(path: str | Path) -> DiCal2Config:
    """Read a diCal2 ``.config`` file.

    Format::

        seq_length  n_alleles  n_populations
        pop0_indicator  pop1_indicator ...   (per haplotype)
        ...

    Parameters
    ----------
    path : str or Path
        Path to the ``.config`` file.

    Returns
    -------
    DiCal2Config
    """
    path = Path(path)
    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Empty .config file: {path}")

    header = lines[0].split()
    seq_length = int(header[0])
    n_alleles = int(header[1])
    n_populations = int(header[2])

    haplotype_populations: list[int] = []
    haplotypes_to_include: list[bool] = []
    multiplicities: list[list[int]] = []
    for ln in lines[1:]:
        indicators = [int(x) for x in ln.split()]
        if len(indicators) != n_populations:
            raise ValueError(f"Config row has {len(indicators)} columns, expected {n_populations}")
        include = any(indicators)
        haplotypes_to_include.append(include)
        multiplicities.append(indicators)
        if not include:
            haplotype_populations.append(-1)
        else:
            try:
                pop_idx = indicators.index(1)
            except ValueError:
                pop_idx = -1
            haplotype_populations.append(pop_idx)

    sample_sizes = np.zeros(n_populations, dtype=np.int64)
    mult_arr = np.array(multiplicities, dtype=np.int64)
    sample_sizes = mult_arr.sum(axis=0) if len(mult_arr) else sample_sizes

    return DiCal2Config(
        seq_length=seq_length,
        n_alleles=n_alleles,
        n_populations=n_populations,
        haplotype_populations=haplotype_populations,
        haplotypes_to_include=haplotypes_to_include,
        haplotype_multiplicities=mult_arr,
        sample_sizes=sample_sizes,
    )


# ---------------------------------------------------------------------------
# Sequence reader (simple tab-separated allele matrix)
# ---------------------------------------------------------------------------


def read_dical2_sequences(
    path: str | Path,
    n_alleles: int = 2,
) -> np.ndarray:
    """Read a diCal2 sequence file (one row per haplotype, space-separated alleles).

    Parameters
    ----------
    path : str or Path
        Path to the sequence file.
    n_alleles : int
        Number of distinct alleles (for validation).

    Returns
    -------
    sequences : (n_haplotypes, seq_length) int8 array
    """
    path = Path(path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([int(x) for x in line.split()])
    sequences = np.array(rows, dtype=np.int8)
    if np.any(sequences >= n_alleles) or np.any(sequences < 0):
        raise ValueError(f"Allele values must be in [0, {n_alleles})")
    return sequences


def read_dical2_vcf(
    vcf_file: str | Path,
    reference_file: str | Path | None,
    config: DiCal2Config,
    filter_pass_string: str = ".",
    *,
    bed_file: str | Path | None = None,
    vcf_offset: int = 0,
    accept_unphased_as_missing: bool = False,
    vcf_ignore_double_entries: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Read a diCal2-style VCF plus reference into a haplotype matrix.

    Matches diCal2's upstream representation more closely: only segregating
    sites for the selected haplotypes are retained in the returned matrix.
    Missing alleles remain encoded as ``-1`` at segregating sites. The
    ``reference_alleles`` return value records the upstream reference-state
    allele carried at each physical locus after VCF preprocessing.
    """
    vcf_file = Path(vcf_file)
    if reference_file is None:
        reference_from_header = None
        with vcf_file.open() as fh:
            for line in fh:
                if line.startswith("##reference=file://"):
                    reference_from_header = line.strip().split("file://", 1)[1]
                    break
                if line.startswith("#CHROM"):
                    break
        if reference_from_header is None:
            raise ValueError("VCF input requires reference_file or a ##reference=file:// header.")
        candidate = Path(reference_from_header)
        reference_file = candidate if candidate.is_absolute() else vcf_file.parent / candidate
    reference_lines = []
    for line in Path(reference_file).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        reference_lines.append(stripped.upper())
    reference = "".join(reference_lines)
    if not reference:
        raise ValueError(f"Reference file {reference_file} contains no sequence")

    include_mask = np.array(config.haplotypes_to_include, dtype=bool)
    missing_bases = {"N", "U", "W", "S", "M", "K", "R", "Y", "B", "D", "H", "V", "."}
    if config.n_alleles < 4:
        reference_alleles = np.array(
            [-1 if base in missing_bases else 0 for base in reference],
            dtype=np.int8,
        )
    else:
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        reference_alleles = np.array(
            [base_to_idx.get(base, -1) for base in reference],
            dtype=np.int8,
        )

    bed_regions: list[tuple[int, int]] = []
    if bed_file is not None:
        previous_end = 0
        for line_number, line in enumerate(Path(bed_file).read_text().splitlines(), start=1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"BED row {line_number} in {bed_file} must contain exactly 3 columns."
                )
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end < start or end > len(reference):
                raise ValueError(
                    f"BED interval [{start}, {end}) is outside reference length {len(reference)}."
                )
            if bed_regions and start < previous_end:
                raise ValueError("BED intervals must be sorted and non-overlapping.")
            bed_regions.append((start, end))
            reference_alleles[start:end] = -1
            previous_end = end

    def _is_masked(position: int) -> bool:
        return any(start <= position < end for start, end in bed_regions)

    selected_variants: list[np.ndarray] = []
    seg_positions: list[int] = []
    previous_pos = -1
    saw_header = False

    with vcf_file.open() as fh:
        n_haps_total = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.rstrip().split("\t")
                if len(parts) < 10 or parts[3:9] != [
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                ]:
                    raise ValueError("VCF header does not contain the required diCal2 columns.")
                n_haps_total = None
                saw_header = True
                continue
            if not line.strip():
                continue
            if not saw_header:
                raise ValueError("VCF has variant records before its #CHROM header.")
            fields = line.rstrip().split("\t")
            if len(fields) < 10:
                raise ValueError("VCF record has fewer than 10 columns.")
            pos = int(fields[1]) - 1 - int(vcf_offset)
            if pos < 0 or pos >= len(reference):
                raise ValueError(
                    "VCF position corrected for vcf_offset is outside reference length "
                    f"{len(reference)}."
                )
            if pos <= previous_pos:
                if pos == previous_pos and vcf_ignore_double_entries:
                    continue
                if pos == previous_pos:
                    raise ValueError(f"VCF contains a duplicate entry at position {fields[1]}.")
                raise ValueError("VCF positions must be sorted after applying vcf_offset.")
            previous_pos = pos
            if _is_masked(pos):
                continue
            ref_allele = fields[3].upper()
            alt_field = fields[4].upper()
            filt = fields[6]
            if filt != filter_pass_string:
                reference_alleles[pos] = -1
                continue
            # The pinned Java reader treats ALT="." as one missing alternate
            # allele rather than as an empty alternate list.
            alt_alleles = alt_field.split(",")
            if len(alt_alleles) > config.n_alleles - 1:
                reference_alleles[pos] = -1
                continue
            if len(ref_allele) != 1 or any(len(alt) != 1 for alt in alt_alleles):
                raise ValueError("diCal2 VCF input does not support structural alleles.")
            if reference[pos] != ref_allele:
                raise ValueError(
                    f"Reference allele mismatch at position {pos + 1 + int(vcf_offset)}: "
                    f"VCF has {ref_allele}, "
                    f"reference has {reference[pos]}"
                )

            if config.n_alleles == 4:
                base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
                allele_map = {"0": base_to_idx.get(ref_allele, -1)}
                for idx, alt in enumerate(alt_alleles, start=1):
                    if alt not in base_to_idx and alt not in missing_bases:
                        raise ValueError(
                            f"Invalid alternative allele in VCF at position {fields[1]}: {alt}."
                        )
                    allele_map[str(idx)] = base_to_idx.get(alt, -1)
            else:
                allele_map = {"0": -1 if ref_allele in missing_bases else 0}
                for idx, alt in enumerate(alt_alleles, start=1):
                    if alt not in {"A", "C", "G", "T"} and alt not in missing_bases:
                        raise ValueError(
                            f"Invalid alternative allele in VCF at position {fields[1]}: {alt}."
                        )
                    allele_map[str(idx)] = -1 if alt in missing_bases else idx
            if len(set(allele_map.values())) != len(allele_map):
                raise ValueError("Reference and alternative VCF alleles must be distinct.")
            if fields[8] != "GT":
                raise ValueError("diCal2 requires the VCF FORMAT column to equal 'GT'.")

            selected_column: list[int] = []
            running_first_haplotype = 0
            for sample in fields[9:]:
                if len(sample) == 1:
                    ploidy = 1
                elif len(sample) == 3:
                    ploidy = 2
                else:
                    raise ValueError(
                        "VCF genotype must contain one haploid allele or two single-digit "
                        f"alleles and a divider; got {sample!r}."
                    )
                last_haplotype = running_first_haplotype + ploidy
                if last_haplotype > len(include_mask):
                    raise ValueError(
                        "VCF exposes more haplotypes than the diCal2 config specifies."
                    )
                selected = include_mask[running_first_haplotype:last_haplotype]
                running_first_haplotype = last_haplotype
                if not np.any(selected):
                    continue

                def _allele(value: str) -> int:
                    if value == ".":
                        return -1
                    if value not in allele_map:
                        raise ValueError(f"Invalid allele {value!r} in VCF genotype {sample!r}.")
                    return int(allele_map[value])

                if ploidy == 1:
                    sample_alleles = [_allele(sample)]
                else:
                    sample_alleles = [_allele(sample[0]), _allele(sample[2])]
                    if sample_alleles[0] != sample_alleles[1] and sample[1] != "|":
                        if not accept_unphased_as_missing:
                            raise ValueError(
                                f"VCF genotype {sample!r} is not phased; set "
                                "accept_unphased_as_missing=True to convert both alleles "
                                "to missing."
                            )
                        sample_alleles = [-1, -1]
                selected_column.extend(
                    allele
                    for allele, include in zip(sample_alleles, selected, strict=True)
                    if include
                )

            n_haps_total = running_first_haplotype
            if n_haps_total != len(include_mask):
                raise ValueError(
                    f"VCF exposes {n_haps_total} haplotypes but config has {len(include_mask)}"
                )
            column = np.asarray(selected_column, dtype=np.int8)
            uniq = set(int(x) for x in column)
            if len(uniq) <= 1:
                reference_alleles[pos] = np.int8(next(iter(uniq)))
                continue
            selected_variants.append(column)
            seg_positions.append(pos)
            reference_alleles[pos] = -1

    if not selected_variants:
        raise ValueError(
            "No segregating sites left in the input vcf after pre-processing and filtering."
        )

    seqs = np.stack(selected_variants, axis=1).astype(np.int8, copy=False)
    return seqs, np.asarray(seg_positions, dtype=np.int64), len(reference), reference_alleles


def _dical2_contig_inputs(
    sequences: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
) -> tuple[list[str | Path | np.ndarray], bool]:
    if isinstance(sequences, (str, Path, np.ndarray)):
        return [sequences], False
    values = list(sequences)
    if not values:
        raise ValueError("At least one diCal2 sequence input is required.")
    return values, True


def _dical2_per_contig_values(value, count: int, name: str) -> list:
    if value is None:
        return [None] * count
    if isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        return [value] * count
    values = list(value)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must contain one value or one value per contig ({count}).")
    return values


# ---------------------------------------------------------------------------
# Combined reader → SmcData
# ---------------------------------------------------------------------------


def read_dical2(
    sequences: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
    param_file: str | Path | None = None,
    demo_file: str | Path | None = None,
    rates_file: str | Path | None = None,
    config_file: str | Path | None = None,
    reference_file: str | Path | Sequence[str | Path] | None = None,
    filter_pass_string: str = ".",
    bed_files: str | Path | Sequence[str | Path] | None = None,
    vcf_offsets: int | Sequence[int] = 0,
    theta: float = 0.0005,
    rho: float = 0.0005,
    n_alleles: int = 2,
    accept_unphased_as_missing: bool = False,
    vcf_ignore_double_entries: bool = False,
) -> SmcData:
    """Read diCal2 inputs into SmcData.

    Parameters
    ----------
    sequences : path, ndarray, or sequence of paths/ndarrays
        Haplotype matrix (n_haplotypes, seq_length) of allele indices, or
        path to a sequence file. Multiple entries are independent contigs.
    param_file : path, optional
        Path to a ``.param`` file. If None, *theta* and *rho* are used
        with a default symmetric mutation matrix.
    demo_file : path, optional
        Path to a ``.demo`` file. If None, a single-population panmictic
        model is assumed.
    rates_file : path, optional
        Path to a ``.rates`` file with exponential growth rates matching
        the demo epochs.
    config_file : path, optional
        Path to a ``.config`` file. If None, all haplotypes are placed
        in a single population.
    reference_file : path or sequence of paths, optional
        External VCF reference(s). A single value is reused for every contig.
        If omitted, each VCF must contain a ``##reference=file://`` header.
    bed_files : path or sequence of paths, optional
        Zero-based, half-open BED masks, one value or one per VCF contig.
    vcf_offsets : int or sequence of int
        Coordinate offset subtracted from VCF positions, one value or one per
        VCF contig.
    accept_unphased_as_missing : bool
        Match diCal2's ``--acceptUnphasedAsMissing`` switch. Unphased
        heterozygotes fail by default; when enabled, both alleles are missing.
    vcf_ignore_double_entries : bool
        Match diCal2's ``--vcfIgnoreDoubleEntries`` switch by retaining the
        first VCF record at a duplicated physical position.
    theta : float
        Mutation rate (used only when *param_file* is None).
    rho : float
        Recombination rate (used only when *param_file* is None).
    n_alleles : int
        Number of allele types (default 2 = biallelic).

    Returns
    -------
    SmcData
    """
    # Load parameters
    if param_file is not None:
        params = read_dical2_param(param_file)
    else:
        # Default symmetric mutation matrix
        mut_mat = np.ones((n_alleles, n_alleles), dtype=np.float64)
        np.fill_diagonal(mut_mat, 0.0)
        params = DiCal2Params(theta=theta, rho=rho, mutation_matrix=mut_mat)

    # Load demography
    if demo_file is not None:
        demo = read_dical2_demo(demo_file)
        if rates_file is not None:
            demo = read_dical2_rates(rates_file, demo)
    else:
        demo = None

    # Load config
    if config_file is not None:
        config = read_dical2_config(config_file)
    else:
        config = None

    # Load independent sequence/VCF contributions.
    sequence_inputs, multiple_inputs = _dical2_contig_inputs(sequences)
    reference_inputs = _dical2_per_contig_values(
        reference_file, len(sequence_inputs), "reference_file"
    )
    bed_inputs = _dical2_per_contig_values(bed_files, len(sequence_inputs), "bed_files")
    offset_inputs = _dical2_per_contig_values(vcf_offsets, len(sequence_inputs), "vcf_offsets")
    contigs: list[DiCal2Contig] = []
    for sequence_input, reference_input, bed_input, offset_input in zip(
        sequence_inputs, reference_inputs, bed_inputs, offset_inputs, strict=True
    ):
        source_path = None
        if isinstance(sequence_input, (str, Path)):
            seq_path = Path(sequence_input)
            source_path = str(seq_path)
            if seq_path.suffix.lower() == ".vcf":
                if config is None:
                    raise ValueError("config_file is required when reading a VCF")
                seqs, seg_positions, reference_length, reference_alleles = read_dical2_vcf(
                    seq_path,
                    reference_input,
                    config,
                    filter_pass_string=filter_pass_string,
                    bed_file=bed_input,
                    vcf_offset=int(offset_input or 0),
                    accept_unphased_as_missing=accept_unphased_as_missing,
                    vcf_ignore_double_entries=vcf_ignore_double_entries,
                )
            else:
                if bed_input is not None or int(offset_input or 0) != 0:
                    raise ValueError("BED masks and VCF offsets require VCF sequence input.")
                seqs = read_dical2_sequences(seq_path, n_alleles=n_alleles)
                seg_positions = None
                reference_length = None
                reference_alleles = None
        else:
            if reference_input is not None or bed_input is not None or int(offset_input or 0) != 0:
                raise ValueError("Reference, BED, and offset controls require VCF sequence input.")
            seqs = np.asarray(sequence_input, dtype=np.int8)
            seg_positions = None
            reference_length = None
            reference_alleles = None
        if seqs.ndim != 2:
            raise ValueError("Each diCal2 sequence input must be a two-dimensional matrix.")
        contigs.append(
            DiCal2Contig(
                sequences=seqs,
                seg_positions=seg_positions,
                reference_length=reference_length,
                reference_alleles=reference_alleles,
                source_path=source_path,
                reference_file=(None if reference_input is None else str(Path(reference_input))),
                bed_file=None if bed_input is None else str(Path(bed_input)),
                vcf_offset=int(offset_input or 0),
            )
        )

    n_hap = int(contigs[0].sequences.shape[0])
    if any(contig.sequences.shape[0] != n_hap for contig in contigs):
        raise ValueError("All diCal2 contigs must contain the same number of haplotypes.")
    seqs = np.concatenate([contig.sequences for contig in contigs], axis=1)
    seq_len = int(seqs.shape[1])

    if config is None:
        config = DiCal2Config(
            seq_length=seq_len,
            n_alleles=n_alleles,
            n_populations=1,
            haplotype_populations=[0] * n_hap,
            haplotypes_to_include=[True] * n_hap,
            haplotype_multiplicities=np.ones((n_hap, 1), dtype=np.int64),
            sample_sizes=np.array([n_hap], dtype=np.int64),
        )
    elif len(config.haplotype_populations) != n_hap:
        include_count = int(np.sum(np.asarray(config.haplotypes_to_include, dtype=bool)))
        if include_count != n_hap:
            raise ValueError(
                "diCal2 config and sequences disagree on haplotype count after filtering."
            )
        config = _compact_dical2_config(config)

    data = SmcData(
        sequences=seqs,
        window_size=1,
        params={
            "theta": params.theta,
            "rho": params.rho,
        },
        uns={
            "mutation_matrix": params.mutation_matrix,
            "demo": demo,
            "config": config,
            "n_haplotypes": n_hap,
            "seq_length": seq_len,
            "contigs": contigs,
            "n_contigs": len(contigs),
            "seg_positions": contigs[0].seg_positions if len(contigs) == 1 else None,
            "reference_length": contigs[0].reference_length if len(contigs) == 1 else None,
            "reference_alleles": contigs[0].reference_alleles if len(contigs) == 1 else None,
            "n_alleles": n_alleles,
            "filter_pass_string": filter_pass_string,
            "accept_unphased_as_missing": bool(accept_unphased_as_missing),
            "vcf_ignore_double_entries": bool(vcf_ignore_double_entries),
            "source_paths": {
                "sequences": (
                    [contig.source_path for contig in contigs]
                    if multiple_inputs
                    else contigs[0].source_path
                ),
                "param_file": None if param_file is None else str(Path(param_file)),
                "demo_file": None if demo_file is None else str(Path(demo_file)),
                "rates_file": None if rates_file is None else str(Path(rates_file)),
                "config_file": None if config_file is None else str(Path(config_file)),
                "reference_file": (
                    [contig.reference_file for contig in contigs]
                    if multiple_inputs
                    else contigs[0].reference_file
                ),
                "bed_files": (
                    [contig.bed_file for contig in contigs]
                    if multiple_inputs
                    else contigs[0].bed_file
                ),
                "vcf_offsets": (
                    [contig.vcf_offset for contig in contigs]
                    if multiple_inputs
                    else contigs[0].vcf_offset
                ),
            },
        },
    )
    return data


def write_dical2_output(
    value: SmcData | dict,
    path: str | Path,
) -> Path:
    """Write captured upstream stdout or parser-compatible native EM rows.

    The native form preserves the complete machine-readable output described
    by the diCal2 manual: one tab-separated row per E-step containing log
    likelihood, elapsed milliseconds, ordered parameters, and the original
    generation/step/particle identifier. Comment lines identify the independent
    smckit implementation. Exact upstream execution retains captured Java
    stdout byte-for-byte apart from adding a final newline when absent.
    """
    result = value.results.get("dical2") if isinstance(value, SmcData) else value
    if not isinstance(result, dict):
        raise ValueError("Value does not contain a diCal2 result mapping.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    upstream = result.get("upstream")
    if isinstance(upstream, dict) and isinstance(upstream.get("stdout"), str):
        text = upstream["stdout"]
        if text and not text.endswith("\n"):
            text += "\n"
    else:
        em_path = result.get("em_path")
        if not isinstance(em_path, list) or not em_path:
            em_path = [
                {
                    "log_likelihood": result["log_likelihood"],
                    "elapsed_ms": 0,
                    "params": result.get("best_params", result.get("ordered_params", [])),
                    "id": "[smckit-native-best]",
                }
            ]
        rows: list[str] = []
        for record in em_path:
            if not isinstance(record, dict):
                raise ValueError("Native diCal2 em_path entries must be mappings.")
            params = np.asarray(record.get("params", []), dtype=np.float64)
            if params.ndim != 1 or params.size == 0:
                raise ValueError(
                    "Native diCal2 result lacks one-dimensional parameters in an EM row."
                )
            run_id = str(record.get("id", "[smckit-native]"))
            if not (run_id.startswith("[") and run_id.endswith("]")):
                run_id = f"[{run_id}]"
            rows.append(
                "\t".join(
                    [
                        f"{float(record['log_likelihood']):.17g}",
                        f"{float(record.get('elapsed_ms', 0)):.17g}",
                        *(f"{float(parameter):.17g}" for parameter in params),
                        run_id,
                    ]
                )
            )
        text = (
            "# smckit native diCal2-compatible EM output\n"
            "# The original CLI's result artifact is stdout; no additional result files "
            "are generated.\n"
            "# LogLikelihood\tTime\tcoordinates...\t[idString]\n" + "\n".join(rows) + "\n"
        )
    target.write_text(text, encoding="utf-8")
    return target
