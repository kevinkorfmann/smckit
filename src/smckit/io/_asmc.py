"""I/O for ASMC (Ascertained Sequentially Markovian Coalescent) formats.

Reads decoding quantities, Oxford haplotype files, sample files, and genetic maps.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smckit._core import SmcData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoding quantities container
# ---------------------------------------------------------------------------


@dataclass
class DecodingQuantities:
    """Precomputed HMM parameters for ASMC decoding.

    Loaded from a ``.decodingQuantities.gz`` file produced by PrepareDecoding.

    Parameters
    ----------
    states : int
        Number of discrete coalescence time states.
    csfs_samples : int
        Number of CSFS sample frequency bins.
    initial_state_prob : (states,) float32
        Initial state probabilities.
    expected_times : (states,) float32
        Expected coalescence time per state.
    discretization : (states+1,) float32
        Time interval boundaries.
    time_vector : (T,) float32
        Time vector from demographic model.
    column_ratios : (states,) float32
        Column ratio coefficients for forward algorithm.
    B_vectors : dict[float, (states,) float32]
        Transition B vectors keyed by rounded genetic distance.
    U_vectors : dict[float, (states,) float32]
        Transition U vectors keyed by rounded genetic distance.
    D_vectors : dict[float, (states,) float32]
        Transition D (diagonal) vectors keyed by rounded genetic distance.
    row_ratio_vectors : dict[float, (states,) float32]
        Row ratio vectors for backward algorithm, keyed by rounded genetic distance.
    classic_emission : (2, states) float32
        Classic emission table (row 0 = homozygous, row 1 = heterozygous).
    compressed_emission : (2, states) float32
        Compressed/ascertained emission table.
    csfs_map : list of (3, states) float32
        CSFS per undistinguished count (rows: obs=0, obs=1, obs=2).
    folded_csfs_map : list of (2, states) float32
        Folded CSFS per undistinguished count.
    ascertained_csfs_map : list of (3, states) float32
        Ascertained CSFS per undistinguished count.
    folded_ascertained_csfs_map : list of (2, states) float32
        Folded ascertained CSFS per undistinguished count.
    homozygous_emission_map : dict[int, (states,) float32]
        Homozygous emissions keyed by physical distance.
    """

    states: int = 0
    csfs_samples: int = 0
    initial_state_prob: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    expected_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    discretization: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    time_vector: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    column_ratios: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    B_vectors: dict[float, np.ndarray] = field(default_factory=dict)
    U_vectors: dict[float, np.ndarray] = field(default_factory=dict)
    D_vectors: dict[float, np.ndarray] = field(default_factory=dict)
    row_ratio_vectors: dict[float, np.ndarray] = field(default_factory=dict)

    classic_emission: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    compressed_emission: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )

    csfs_map: list[np.ndarray] = field(default_factory=list)
    folded_csfs_map: list[np.ndarray] = field(default_factory=list)
    ascertained_csfs_map: list[np.ndarray] = field(default_factory=list)
    folded_ascertained_csfs_map: list[np.ndarray] = field(default_factory=list)

    homozygous_emission_map: dict[int, np.ndarray] = field(default_factory=dict)


def read_decoding_quantities(path: str | Path) -> DecodingQuantities:
    """Read precomputed ASMC decoding quantities from a gzipped text file.

    Parameters
    ----------
    path : str or Path
        Path to ``.decodingQuantities.gz`` file.

    Returns
    -------
    DecodingQuantities
    """
    path = Path(path)
    dq = DecodingQuantities()

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as f:
        current_type = None

        for line in f:
            tokens = line.split()
            if not tokens:
                continue

            first = tokens[0].lower()

            if first == "transitiontype":
                current_type = "transitiontype"
                next(f, None)  # skip value line
                continue
            elif first == "states":
                current_type = "states"
                val_line = next(f, "").strip()
                dq.states = int(val_line)
                continue
            elif first == "csfssamples":
                current_type = "csfssamples"
                val_line = next(f, "").strip()
                dq.csfs_samples = int(val_line)
                dq.csfs_map = [None] * (dq.csfs_samples - 1)
                dq.folded_csfs_map = [None] * (dq.csfs_samples - 1)
                dq.ascertained_csfs_map = [None] * (dq.csfs_samples - 1)
                dq.folded_ascertained_csfs_map = [None] * (dq.csfs_samples - 1)
                continue
            elif first == "timevector":
                current_type = "timevector"
                val_line = next(f, "").strip()
                dq.time_vector = np.array([float(x) for x in val_line.split()], dtype=np.float32)
                continue
            elif first == "sizevector":
                current_type = "sizevector"
                next(f, None)  # skip
                continue
            elif first == "expectedtimes":
                current_type = "expectedtimes"
                val_line = next(f, "").strip()
                vals = [float(x) for x in val_line.split()]
                dq.expected_times = np.array(vals, dtype=np.float32)
                continue
            elif first == "discretization":
                current_type = "discretization"
                val_line = next(f, "").strip()
                vals = [float(x) for x in val_line.split()]
                dq.discretization = np.array(vals, dtype=np.float32)
                continue
            elif first == "classicemission":
                current_type = "classicemission"
                rows = []
                for _ in range(2):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.classic_emission = np.array(rows, dtype=np.float32)
                continue
            elif first == "compressedascertainedemission":
                current_type = "compressedascertainedemission"
                rows = []
                for _ in range(2):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.compressed_emission = np.array(rows, dtype=np.float32)
                continue
            elif first == "csfs":
                idx = int(tokens[1])
                rows = []
                for _ in range(3):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.csfs_map[idx] = np.array(rows, dtype=np.float32)
                continue
            elif first == "foldedcsfs":
                idx = int(tokens[1])
                rows = []
                for _ in range(2):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.folded_csfs_map[idx] = np.array(rows, dtype=np.float32)
                continue
            elif first == "ascertainedcsfs":
                idx = int(tokens[1])
                rows = []
                for _ in range(3):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.ascertained_csfs_map[idx] = np.array(rows, dtype=np.float32)
                continue
            elif first == "foldedascertainedcsfs":
                idx = int(tokens[1])
                rows = []
                for _ in range(2):
                    val_line = next(f, "").strip()
                    rows.append([float(x) for x in val_line.split()])
                dq.folded_ascertained_csfs_map[idx] = np.array(rows, dtype=np.float32)
                continue
            elif first == "homozygousemissions":
                current_type = "homozygousemissions"
                continue
            elif first == "initialstateprob":
                current_type = "initialstateprob"
                continue
            elif first == "columnratios":
                current_type = "columnratios"
                continue
            elif first == "rowratios":
                current_type = "rowratios"
                continue
            elif first == "uvectors":
                current_type = "uvectors"
                continue
            elif first == "bvectors":
                current_type = "bvectors"
                continue
            elif first == "dvectors":
                current_type = "dvectors"
                continue
            else:
                # Data line for current section
                if current_type == "columnratios":
                    dq.column_ratios = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens):
                        dq.column_ratios[i] = float(v)
                elif current_type == "initialstateprob":
                    dq.initial_state_prob = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens):
                        dq.initial_state_prob[i] = float(v)
                elif current_type == "rowratios":
                    key = float(tokens[0])
                    vals = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens[1:]):
                        vals[i] = float(v)
                    dq.row_ratio_vectors[key] = vals
                elif current_type == "uvectors":
                    key = float(tokens[0])
                    vals = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens[1:]):
                        vals[i] = float(v)
                    dq.U_vectors[key] = vals
                elif current_type == "bvectors":
                    key = float(tokens[0])
                    vals = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens[1:]):
                        vals[i] = float(v)
                    dq.B_vectors[key] = vals
                elif current_type == "dvectors":
                    key = float(tokens[0])
                    vals = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens[1:]):
                        vals[i] = float(v)
                    dq.D_vectors[key] = vals
                elif current_type == "homozygousemissions":
                    key = int(tokens[0])
                    vals = np.zeros(dq.states, dtype=np.float32)
                    for i, v in enumerate(tokens[1:]):
                        vals[i] = float(v)
                    dq.homozygous_emission_map[key] = vals

    logger.info(
        "Loaded decoding quantities: states=%d, csfs_samples=%d, %d transition distances",
        dq.states,
        dq.csfs_samples,
        len(dq.D_vectors),
    )
    return dq


# ---------------------------------------------------------------------------
# Haplotype data (Oxford .hap format)
# ---------------------------------------------------------------------------


def read_hap(
    path: str | Path,
    fold_to_minor: bool = True,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray]:
    """Read Oxford .hap/.hap.gz phased haplotype file.

    Format: each line is ``chr snpID pos alleleA alleleB hap0 hap1 ...``

    When ``fold_to_minor=True``, alleles are flipped at sites where the
    derived allele count exceeds the ancestral count, matching the C++
    ASMC behavior. This ensures the minor allele is always coded as 1.

    Parameters
    ----------
    path : str or Path
    fold_to_minor : bool
        Flip alleles so the minor allele is always 1.

    Returns
    -------
    haplotypes : (n_haps, n_sites) uint8 array
        Haplotype matrix (0/1 per haplotype per site).
    snp_ids : list of str
        SNP identifiers.
    positions : list of int
        Physical positions in base pairs.
    flipped : (n_sites,) bool array
        True at sites where alleles were flipped during folding.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open

    hap_rows = []
    snp_ids = []
    positions = []

    with opener(path, "rt") as f:
        expected_haplotypes: int | None = None
        for line_number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 7:
                raise ValueError(
                    f"Malformed ASMC haplotype row {line_number} in {path}: "
                    "expected at least 7 columns."
                )
            # chr snpID pos alleleA alleleB hap0 hap1 ...
            snp_ids.append(parts[1])
            positions.append(int(parts[2]))
            try:
                genotypes = [int(x) for x in parts[5:]]
            except ValueError as exc:
                raise ValueError(
                    f"Non-binary ASMC haplotype value on row {line_number} in {path}."
                ) from exc
            if any(value not in (0, 1) for value in genotypes):
                raise ValueError(
                    f"ASMC haplotype row {line_number} in {path} contains values "
                    "other than 0 or 1."
                )
            if expected_haplotypes is None:
                expected_haplotypes = len(genotypes)
            elif len(genotypes) != expected_haplotypes:
                raise ValueError(
                    f"Inconsistent haplotype count on row {line_number} in {path}: "
                    f"expected {expected_haplotypes}, found {len(genotypes)}."
                )
            hap_rows.append(genotypes)

    if not hap_rows:
        raise ValueError(f"ASMC haplotype file is empty: {path}")

    # hap_rows is (n_sites, n_haps) -> transpose to (n_haps, n_sites)
    haplotypes = np.array(hap_rows, dtype=np.uint8).T
    n_haps, n_sites = haplotypes.shape

    flipped = np.zeros(n_sites, dtype=np.bool_)
    if fold_to_minor:
        derived_counts = haplotypes.sum(axis=0)
        ancestral_counts = n_haps - derived_counts
        flip_mask = derived_counts > ancestral_counts
        flipped[flip_mask] = True
        # Flip alleles: 0 <-> 1
        haplotypes[:, flip_mask] = 1 - haplotypes[:, flip_mask]

    return haplotypes, snp_ids, positions, flipped


def read_samples(path: str | Path) -> list[dict[str, str]]:
    """Read Oxford .sample/.samples file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    samples : list of dict
        Each dict has keys ``fam_id``, ``ind_id``, ``missing``, ``sex``.
    """
    path = Path(path)
    samples = []

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        # Skip header lines (2 header lines in Oxford format)
        next(f, None)
        next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append(
                    {
                        "fam_id": parts[0],
                        "ind_id": parts[1],
                        "missing": parts[2] if len(parts) > 2 else "0",
                        "sex": parts[3] if len(parts) > 3 else "NA",
                    }
                )
    return samples


def read_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read genetic map file.

    Format: each line is ``chr snpID geneticPos(cM) physicalPos(bp)``

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    genetic_positions : (n_sites,) float32 in Morgans
    physical_positions : (n_sites,) int32 in base pairs
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open

    gen_pos = []
    phys_pos = []

    with opener(path, "rt") as f:
        for line in f:
            parts = line.split()
            # Upstream parses map coordinates into float32 before converting
            # cM to Morgans; preserving that order matters for long WGS HMMs.
            gen_pos.append(
                np.float32(parts[2]) / np.float32(100.0)
            )
            phys_pos.append(int(parts[3]))

    return (
        np.array(gen_pos, dtype=np.float32),
        np.array(phys_pos, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Undistinguished counts (for CSFS emission model)
# ---------------------------------------------------------------------------


def compute_undistinguished_counts(
    haplotypes: np.ndarray,
    csfs_samples: int,
    fold_to_minor: bool = True,
    *,
    strategy: str = "expected",
    random_seed: int | None = None,
    allow_undersampled: bool = False,
) -> np.ndarray:
    """Compute undistinguished allele counts for CSFS emissions.

    For each site, compute the number of derived alleles among the
    "undistinguished" (non-focal) lineages, for each of the 3 observation
    types (0 = both ancestral, 1 = heterozygous, 2 = both derived).

    ``strategy="expected"`` uses the rounded hypergeometric expectation and
    is deterministic. ``strategy="sample"`` draws from the same
    hypergeometric distribution used by upstream ASMC; ``random_seed``
    controls reproducibility.

    Parameters
    ----------
    haplotypes : (n_haps, n_sites) uint8
    csfs_samples : int
        Total number of CSFS samples (from decoding quantities).
    fold_to_minor : bool
        Whether to fold allele counts to minor allele.
    strategy : {"expected", "sample"}
        Deterministic expected count or a stochastic hypergeometric draw.
    random_seed : int, optional
        Seed used by ``strategy="sample"``.
    allow_undersampled : bool
        Return unavailable counts instead of failing when the CSFS sample
        requirement exceeds the input. This supports compressed decoding.

    Returns
    -------
    undistinguished : (n_sites, 3) int32
        Undistinguished counts for obs types 0, 1, 2. Value of -1 means
        the count is invalid (out of range).
    """
    if strategy not in {"expected", "sample"}:
        raise ValueError("strategy must be one of: expected, sample")

    n_haps, n_sites = haplotypes.shape
    if n_haps < 2:
        raise ValueError("ASMC requires at least two haplotypes.")
    if csfs_samples < 2:
        # Compressed test fixtures can omit CSFS tables entirely.
        return np.full((n_sites, 3), -1, dtype=np.int32)
    if csfs_samples > n_haps:
        if allow_undersampled:
            return np.full((n_sites, 3), -1, dtype=np.int32)
        raise ValueError(
            f"CSFS requires {csfs_samples} haplotypes but input contains only {n_haps}."
        )
    rng = np.random.default_rng(random_seed)

    # Derived allele count per site
    derived_counts = haplotypes.sum(axis=0).astype(np.int32)
    total_samples = np.full(n_sites, n_haps, dtype=np.int32)

    # Fold to minor allele if needed
    if fold_to_minor:
        ancestral_counts = total_samples - derived_counts
        flip = derived_counts > ancestral_counts
        derived_counts[flip] = ancestral_counts[flip]

    undistinguished = np.zeros((n_sites, 3), dtype=np.int32)

    for site in range(n_sites):
        d = derived_counts[site]
        n = total_samples[site]

        for distinguished in range(3):
            # Hypergeometric parameters:
            # Population size = n - 2 (remove the focal pair)
            # Successes in population = d - distinguished
            # Sample size = csfs_samples - 2
            successes = d - distinguished
            pop_size = n - 2
            sample_size = csfs_samples - 2

            if successes < 0 or successes > pop_size:
                undistinguished[site, distinguished] = -1
                continue

            if strategy == "sample":
                sample = int(
                    rng.hypergeometric(
                        successes,
                        pop_size - successes,
                        sample_size,
                    )
                )
            else:
                # Deterministic expected value: E[X] = n * K / N.
                sample = round(sample_size * successes / pop_size)

            if fold_to_minor and (sample + distinguished > csfs_samples // 2):
                sample = sample_size - sample

            undistinguished[site, distinguished] = sample

    return undistinguished


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def read_asmc(
    file_root: str | Path,
    dq_file: str | Path,
    *,
    map_file: str | Path | None = None,
    fold_to_minor: bool = True,
    undistinguished_strategy: str = "expected",
    random_seed: int | None = None,
) -> SmcData:
    """Load ASMC input data from standard file set.

    Expects files: ``{file_root}.hap.gz``, ``{file_root}.samples``,
    ``{file_root}.map.gz``, and a decoding quantities file.

    Parameters
    ----------
    file_root : str or Path
        Common prefix for hap/samples/map files.
    dq_file : str or Path
        Path to ``.decodingQuantities.gz`` file.
    map_file : str or Path, optional
        Explicit genetic map path instead of ``{file_root}.map[.gz]``.
    fold_to_minor : bool
        Match the default upstream behavior by coding the minor allele as 1.
        Set to false when ancestral alleles are already coded as 1.
    undistinguished_strategy : {"expected", "sample"}
        Method for constructing CSFS undistinguished counts.
    random_seed : int, optional
        Seed used for stochastic hypergeometric sampling.

    Returns
    -------
    SmcData
        With ``uns`` populated for ASMC decoding.
    """
    file_root = Path(file_root)

    # Find haplotype file
    hap_path = _find_file(file_root, [".hap.gz", ".hap", ".haps.gz", ".haps"])
    samples_path = _find_file(file_root, [".samples", ".sample"])
    map_path = (
        Path(map_file) if map_file is not None else _find_file(file_root, [".map.gz", ".map"])
    )
    if not map_path.is_file():
        raise FileNotFoundError(f"ASMC genetic map does not exist: {map_path}")

    haplotypes, snp_ids, hap_positions, flipped = read_hap(
        hap_path,
        fold_to_minor=fold_to_minor,
    )
    samples = read_samples(samples_path)
    genetic_positions, physical_positions = read_map(map_path)
    dq = read_decoding_quantities(dq_file)
    if haplotypes.shape[1] != genetic_positions.size:
        raise ValueError(
            "ASMC haplotype and genetic-map site counts differ: "
            f"{haplotypes.shape[1]} != {genetic_positions.size}."
        )
    if not np.array_equal(np.asarray(hap_positions), physical_positions):
        raise ValueError("ASMC haplotype and map physical positions do not match.")
    if len(samples) * 2 != haplotypes.shape[0]:
        raise ValueError(
            "ASMC sample and haplotype counts differ: "
            f"{len(samples)} diploid samples imply {2 * len(samples)} haplotypes, "
            f"found {haplotypes.shape[0]}."
        )
    if np.any(np.diff(genetic_positions) < 0):
        raise ValueError("ASMC genetic positions must be nondecreasing.")
    if np.any(np.diff(physical_positions) < 0):
        raise ValueError("ASMC physical positions must be nondecreasing.")

    # Compute recombination rates between markers
    rec_rates = np.zeros_like(genetic_positions)
    genetic_distances = np.diff(genetic_positions)
    physical_distances = np.diff(physical_positions)
    informative = genetic_distances >= np.finfo(np.float32).eps
    np.divide(
        genetic_distances,
        physical_distances,
        out=rec_rates[1:],
        where=informative & (physical_distances != 0),
    )
    if rec_rates.size > 1:
        rec_rates[0] = rec_rates[1]

    # Compute undistinguished counts for CSFS
    undist_counts = compute_undistinguished_counts(
        haplotypes,
        dq.csfs_samples,
        fold_to_minor=fold_to_minor,
        strategy=undistinguished_strategy,
        random_seed=random_seed,
        allow_undersampled=True,
    )
    data_has_csfs_sample_size = dq.csfs_samples <= haplotypes.shape[0]

    data = SmcData()
    data.sequences = haplotypes
    data.uns["haplotypes"] = haplotypes
    data.uns["snp_ids"] = snp_ids
    data.uns["samples"] = samples
    data.uns["genetic_positions"] = genetic_positions
    data.uns["physical_positions"] = physical_positions
    data.uns["rec_rates"] = rec_rates
    data.uns["decoding_quantities"] = dq
    data.uns["undistinguished_counts"] = undist_counts
    data.uns["site_was_flipped"] = flipped
    data.uns["fold_to_minor"] = fold_to_minor
    data.uns["undistinguished_strategy"] = undistinguished_strategy
    data.uns["random_seed"] = random_seed
    data.uns["csfs_sample_size_available"] = data_has_csfs_sample_size
    data.uns["input_file_root"] = str(file_root)
    data.uns["decoding_quantities_path"] = str(Path(dq_file))
    data.uns["map_path"] = str(map_path)
    data.uns["source_paths"] = [
        str(hap_path),
        str(samples_path),
        str(map_path),
        str(Path(dq_file)),
    ]

    logger.info(
        "Loaded ASMC data: %d haplotypes, %d sites",
        haplotypes.shape[0],
        haplotypes.shape[1],
    )
    return data


def _find_file(root: Path, suffixes: list[str]) -> Path:
    """Find a file matching root + one of the given suffixes."""
    for suffix in suffixes:
        candidate = Path(str(root) + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find file with root '{root}' and suffixes {suffixes}")


def read_asmc_posterior_sums(path: str | Path) -> np.ndarray:
    """Read an ASMC ``*.sumOverPairs[.gz]`` matrix."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:
        values = np.loadtxt(handle, dtype=np.float64, ndmin=2)
    return np.asarray(values)


def write_asmc_posterior_sums(
    path: str | Path,
    values: np.ndarray,
) -> Path:
    """Write an original-compatible ASMC posterior-sum matrix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("ASMC posterior sums must be a two-dimensional matrix.")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt") as handle:
        np.savetxt(handle, matrix, fmt="%.9g", delimiter="\t")
    return path


def merge_asmc_posterior_sums(
    paths: list[str | Path],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Merge ASMC job outputs and optionally normalize each site."""
    if not paths:
        raise ValueError("At least one ASMC posterior-sum path is required.")
    matrices = [read_asmc_posterior_sums(path) for path in paths]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("ASMC posterior-sum matrices must have identical shapes.")
    merged = np.sum(matrices, axis=0, dtype=np.float64)
    if normalize:
        totals = merged.sum(axis=1, keepdims=True)
        np.divide(merged, totals, out=merged, where=totals != 0)
    return merged
