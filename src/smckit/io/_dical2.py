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
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# .param file parser
# ---------------------------------------------------------------------------


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
                    pop_param_ids.append(
                        int(x.strip()[1:]) if is_placeholder else None
                    )
                pop_sizes = np.array(
                    pop_values,
                    dtype=np.float64,
                )
                pop_size_param_ids = pop_param_ids
                idx += 1

        # Pulse migration matrix (or null)
        pulse_migration: np.ndarray | None = None
        if idx < len(lines):
            if lines[idx].lower() == "null":
                idx += 1
            else:
                pulse_rows = []
                # n_ancient rows for the pulse matrix
                for _ in range(n_ancient):
                    if idx < len(lines):
                        pulse_rows.append(
                            [_parse_value(x, default=0.0)[0] for x in lines[idx].split()]
                        )
                        idx += 1
                if pulse_rows:
                    pulse_migration = np.array(pulse_rows, dtype=np.float64)

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
                            migration_matrix[i, :i].sum()
                            + migration_matrix[i, i + 1 :].sum()
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
                pulse_migration=pulse_migration if is_pulse else None,
                growth_rates=None,
                growth_rate_param_ids=None,
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
                None
                if epoch.migration_matrix is None
                else epoch.migration_matrix.copy()
            ),
            migration_param_ids=(
                None
                if epoch.migration_param_ids is None
                else [list(row) for row in epoch.migration_param_ids]
            ),
            pulse_migration=(
                None if epoch.pulse_migration is None else epoch.pulse_migration.copy()
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
            raise ValueError(
                f"Config row has {len(indicators)} columns, expected {n_populations}"
            )
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
    reference_file: str | Path,
    config: DiCal2Config,
    filter_pass_string: str = ".",
) -> np.ndarray:
    """Read a diCal2-style VCF plus reference into a haplotype matrix.

    Missing reference bases are encoded as ``-1`` and sites with any missing
    called allele are masked out across all selected haplotypes.
    """
    vcf_file = Path(vcf_file)
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
    seqs = np.zeros((int(include_mask.sum()), len(reference)), dtype=np.int8)
    for idx, base in enumerate(reference):
        if base in {"N", "U", "W", "S", "M", "K", "R", "Y", "B", "D", "H", "V", "."}:
            seqs[:, idx] = -1

    with vcf_file.open() as fh:
        n_haps_total = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.rstrip().split("\t")
                n_haps_total = 2 * (len(parts) - 9)
                if n_haps_total != len(include_mask):
                    raise ValueError(
                        f"VCF exposes {n_haps_total} haplotypes but config has {len(include_mask)}"
                    )
                continue
            if not line.strip():
                continue
            fields = line.rstrip().split("\t")
            pos = int(fields[1]) - 1
            if pos < 0 or pos >= len(reference):
                raise ValueError(
                    f"VCF position {pos + 1} is outside reference length {len(reference)}"
                )
            ref_allele = fields[3].upper()
            alt_field = fields[4].upper()
            filt = fields[6]
            if filt != filter_pass_string:
                seqs[:, pos] = -1
                continue
            if alt_field == ".":
                continue
            alt_alleles = alt_field.split(",")
            if len(alt_alleles) > config.n_alleles - 1:
                seqs[:, pos] = -1
                continue
            if reference[pos] not in {ref_allele, "N"}:
                raise ValueError(
                    f"Reference allele mismatch at position {pos + 1}: VCF has {ref_allele}, "
                    f"reference has {reference[pos]}"
                )

            allele_map = {"0": 0}
            for idx, _alt in enumerate(alt_alleles, start=1):
                allele_map[str(idx)] = idx if config.n_alleles > 2 else idx

            full_column: list[int] = []
            has_missing = False
            for sample in fields[9:]:
                gt = sample.split(":", 1)[0]
                if gt in {"./.", ".|."}:
                    full_column.extend([-1, -1])
                    has_missing = True
                    continue
                if "|" in gt:
                    a, b = gt.split("|")
                elif "/" in gt:
                    a, b = gt.split("/")
                    if a != b:
                        full_column.extend([-1, -1])
                        has_missing = True
                        continue
                else:
                    a = b = gt
                if a == "." or b == ".":
                    full_column.extend([-1, -1])
                    has_missing = True
                    continue
                full_column.extend([allele_map.get(a, -1), allele_map.get(b, -1)])
                if full_column[-1] < 0 or full_column[-2] < 0:
                    has_missing = True

            column = np.array(full_column, dtype=np.int8)[include_mask]
            uniq = set(int(x) for x in column)
            if len(uniq) <= 1:
                if len(uniq) == 1 and next(iter(uniq)) >= 0:
                    seqs[:, pos] = next(iter(uniq))
                continue
            if has_missing:
                seqs[:, pos] = -1
            else:
                seqs[:, pos] = column
    return seqs


# ---------------------------------------------------------------------------
# Combined reader → SmcData
# ---------------------------------------------------------------------------


def read_dical2(
    sequences: str | Path | np.ndarray,
    param_file: str | Path | None = None,
    demo_file: str | Path | None = None,
    rates_file: str | Path | None = None,
    config_file: str | Path | None = None,
    reference_file: str | Path | None = None,
    filter_pass_string: str = ".",
    theta: float = 0.0005,
    rho: float = 0.0005,
    n_alleles: int = 2,
) -> SmcData:
    """Read diCal2 inputs into SmcData.

    Parameters
    ----------
    sequences : path or ndarray
        Haplotype matrix (n_haplotypes, seq_length) of allele indices, or
        path to a sequence file.
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

    # Load sequences
    if isinstance(sequences, (str, Path)):
        seq_path = Path(sequences)
        if seq_path.suffix.lower() == ".vcf":
            if reference_file is None:
                raise ValueError("reference_file is required when reading a VCF")
            if config is None:
                raise ValueError("config_file is required when reading a VCF")
            seqs = read_dical2_vcf(
                seq_path,
                reference_file,
                config,
                filter_pass_string=filter_pass_string,
            )
        else:
            seqs = read_dical2_sequences(seq_path, n_alleles=n_alleles)
    else:
        seqs = np.asarray(sequences, dtype=np.int8)

    n_hap, seq_len = seqs.shape

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
            "n_alleles": n_alleles,
            "source_paths": {
                "sequences": None if not isinstance(sequences, (str, Path)) else str(Path(sequences)),
                "param_file": None if param_file is None else str(Path(param_file)),
                "demo_file": None if demo_file is None else str(Path(demo_file)),
                "rates_file": None if rates_file is None else str(Path(rates_file)),
                "config_file": None if config_file is None else str(Path(config_file)),
                "reference_file": None if reference_file is None else str(Path(reference_file)),
            },
        },
    )
    return data
