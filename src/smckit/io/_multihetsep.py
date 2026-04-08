"""Read MSMC2 multihetsep input format and MSMC2 output files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from smckit._core import SmcData

# Valid called alleles (case-insensitive)
_CALLED = set("ACGTacgt0")
_MISSING = set("Nn?")


def _parse_allele_observation(
    allele_str: str, i: int, j: int, skip_ambiguous: bool = False
) -> float:
    """Compute observation for haplotype pair (i, j) from an allele string.

    Parameters
    ----------
    allele_str : str
        Allele string, possibly comma-separated for ambiguous phasing.
    i : int
        Index of first haplotype.
    j : int
        Index of second haplotype.

    skip_ambiguous : bool
        If True, ambiguous phasing is treated as missing, matching MSMC2's
        ``-s/--skipAmbiguous`` behavior.

    Returns
    -------
    float
        Observation value: 0 (missing), 1 (homozygous), 2 (heterozygous),
        a float average for ambiguous phasing, or -1 when
        ``skip_ambiguous=True`` and the site is skipped as ambiguous.
    """
    configs = allele_str.split(",")
    obs_values = []
    for config in configs:
        if i >= len(config) or j >= len(config):
            return 0.0
        ai = config[i]
        aj = config[j]
        if ai in _MISSING or aj in _MISSING:
            return 0.0
        if ai.upper() == aj.upper():
            obs_values.append(1.0)
        else:
            obs_values.append(2.0)

    uniq = sorted(set(obs_values))
    if len(uniq) == 1:
        return uniq[0]
    if skip_ambiguous:
        return -1.0
    return sum(obs_values) / len(obs_values)


def _detect_n_haplotypes(allele_columns: list[str]) -> int:
    """Detect the number of haplotypes from allele column strings.

    Each allele column encodes haplotypes as characters. Multiple columns
    are possible but each column has the same number of haplotypes.
    The number of haplotypes is the length of the first configuration
    (before any comma) of the first allele column.
    """
    for col in allele_columns:
        first_config = col.split(",")[0]
        if first_config:
            return len(first_config)
    return 0


def _all_pairs(n_haplotypes: int) -> list[tuple[int, int]]:
    """Generate all unique haplotype pair indices."""
    pairs = []
    for i in range(n_haplotypes):
        for j in range(i + 1, n_haplotypes):
            pairs.append((i, j))
    return pairs


def read_multihetsep(
    path: str | Path | list[str | Path],
    pair_indices: list[tuple[int, int]] | None = None,
    skip_ambiguous: bool = False,
) -> SmcData:
    """Read MSMC2 multihetsep input file(s) into a SmcData object.

    The multihetsep format is a tab-separated text file with columns:

    1. chr (string): chromosome name
    2. pos (int): position (rightmost in segment, 1-based)
    3. nCalledSites (int): number of called sites in this segment
    4. alleles (string): one or more allele columns for haplotype groups

    Parameters
    ----------
    path : str, Path, or list thereof
        Path to multihetsep file, or a list of paths for multiple chromosomes.
    pair_indices : list of (int, int) tuples, optional
        Haplotype pairs to analyze. Each tuple ``(i, j)`` specifies the
        zero-based indices of two haplotypes. If ``None``, all unique pairs
        from the detected haplotypes are used.
    skip_ambiguous : bool
        If True, sites with ambiguous phasing are treated as missing on a
        per-pair basis, matching MSMC2's ``-s/--skipAmbiguous`` mode.

    Returns
    -------
    SmcData
        Data container with:

        - ``uns["segments"]``: list of dicts per chromosome, each with
          ``"chr"``, ``"positions"``, ``"n_called"``, and ``"obs"``
          (dict mapping pair tuple to int8 observation array).
        - ``uns["pairs"]``: list of (i, j) pair tuples.
        - ``uns["n_haplotypes"]``: number of haplotypes detected.
    """
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    segments: list[dict[str, Any]] = []
    n_haplotypes: int | None = None

    for filepath in paths:
        chr_segments: dict[str, dict[str, list]] = {}

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                chrom = parts[0]
                pos = int(parts[1])
                n_called = int(parts[2])
                allele_cols = parts[3:]

                # Detect haplotype count from first valid line
                if n_haplotypes is None:
                    n_haplotypes = _detect_n_haplotypes(allele_cols)
                    if n_haplotypes == 0:
                        raise ValueError(
                            f"Could not detect haplotype count from {filepath}"
                        )

                if chrom not in chr_segments:
                    chr_segments[chrom] = {
                        "positions": [],
                        "n_called": [],
                        "allele_cols": [],
                    }

                chr_segments[chrom]["positions"].append(pos)
                chr_segments[chrom]["n_called"].append(n_called)
                chr_segments[chrom]["allele_cols"].append(allele_cols)

        for chrom in chr_segments:
            seg = chr_segments[chrom]
            segments.append({
                "chr": chrom,
                "source_path": str(filepath),
                "positions": np.array(seg["positions"], dtype=np.int64),
                "n_called": np.array(seg["n_called"], dtype=np.int64),
                "_allele_cols": seg["allele_cols"],
            })

    if n_haplotypes is None or n_haplotypes == 0:
        raise ValueError("No valid data found in input file(s)")

    # Determine pairs
    if pair_indices is not None:
        pairs = [tuple(p) for p in pair_indices]
    else:
        pairs = _all_pairs(n_haplotypes)

    if not pairs:
        raise ValueError(
            f"No valid haplotype pairs for {n_haplotypes} haplotype(s). "
            "Need at least 2 haplotypes or explicit pair_indices."
        )

    # Validate pair indices
    for i, j in pairs:
        if i < 0 or j < 0 or i >= n_haplotypes or j >= n_haplotypes:
            raise ValueError(
                f"Pair ({i}, {j}) out of range for {n_haplotypes} haplotypes"
            )

    # Compute observations per pair per segment
    for seg in segments:
        allele_cols_list = seg.pop("_allele_cols")
        n_sites = len(allele_cols_list)
        obs_dict: dict[tuple[int, int], np.ndarray] = {}

        for pair in pairs:
            obs = np.empty(n_sites, dtype=np.float64)
            for site_idx, allele_cols in enumerate(allele_cols_list):
                # Concatenate all allele columns into one string per config
                # Each column represents a group of haplotypes; the full allele
                # string is the concatenation across columns.
                if len(allele_cols) == 1:
                    combined = allele_cols[0]
                else:
                    # For multiple allele columns, each may have comma-separated
                    # configs. Combine by concatenating corresponding configs.
                    split_cols = [col.split(",") for col in allele_cols]
                    n_configs = max(len(sc) for sc in split_cols)
                    combined_configs = []
                    for cfg_idx in range(n_configs):
                        parts = []
                        for sc in split_cols:
                            idx = min(cfg_idx, len(sc) - 1)
                            parts.append(sc[idx])
                        combined_configs.append("".join(parts))
                    combined = ",".join(combined_configs)

                obs[site_idx] = _parse_allele_observation(
                    combined, pair[0], pair[1], skip_ambiguous=skip_ambiguous
                )

            # Round to int8 for non-ambiguous cases, keep float for ambiguous
            if np.all(np.equal(np.mod(obs, 1), 0)):
                obs_dict[pair] = obs.astype(np.int8)
            else:
                obs_dict[pair] = obs

        seg["obs"] = obs_dict

    data = SmcData()
    data.uns["segments"] = segments
    data.uns["pairs"] = pairs
    data.uns["n_haplotypes"] = n_haplotypes
    data.uns["source_paths"] = [str(p) for p in paths]

    return data


def read_msmc_output(path: str | Path) -> dict[str, np.ndarray]:
    """Read MSMC2 ``.final.txt`` output file.

    The MSMC2 output format is a tab-separated file with columns:

    - time_index (int)
    - left_time_boundary (float)
    - right_time_boundary (float)
    - lambda (float)

    Parameters
    ----------
    path : str or Path
        Path to MSMC2 ``.final.txt`` file.

    Returns
    -------
    dict
        Dictionary with keys ``"time_index"``, ``"left_boundary"``,
        ``"right_boundary"``, and ``"lambda"``, each containing a NumPy array.
    """
    path = Path(path)

    time_indices: list[int] = []
    left_boundaries: list[float] = []
    right_boundaries: list[float] = []
    lambdas: list[float] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("time_index"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            time_indices.append(int(parts[0]))
            left_boundaries.append(float(parts[1]))
            right_boundaries.append(float(parts[2]))
            lambdas.append(float(parts[3]))

    if not time_indices:
        raise ValueError(f"No data found in {path}")

    return {
        "time_index": np.array(time_indices, dtype=np.int64),
        "left_boundary": np.array(left_boundaries, dtype=np.float64),
        "right_boundary": np.array(right_boundaries, dtype=np.float64),
        "lambda": np.array(lambdas, dtype=np.float64),
    }


def read_msmc_combined_output(
    path: str | Path,
    mu: float = 1.25e-8,
) -> dict[str, np.ndarray]:
    """Read MSMC2 combined cross-population output file.

    This reads the 6-column combined output from MSMC2 that contains
    within-population and cross-population coalescence rates for a pair
    of populations. Time boundaries are rescaled from coalescent units
    to generations using the mutation rate.

    Parameters
    ----------
    path : str or Path
        Path to combined MSMC2 output file (e.g.
        ``twopops.combined.msmc2.final.txt``).
    mu : float
        Per-base per-generation mutation rate. Default ``1.25e-8`` (human).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"left_boundary"`` : left time boundaries in generations
        - ``"right_boundary"`` : right time boundaries in generations
        - ``"lambda_00"`` : within-pop1 coalescence rates (scaled by mu)
        - ``"lambda_01"`` : cross-population coalescence rates (scaled by mu)
        - ``"lambda_11"`` : within-pop2 coalescence rates (scaled by mu)
    """
    path = Path(path)

    left_boundaries: list[float] = []
    right_boundaries: list[float] = []
    lambdas_00: list[float] = []
    lambdas_01: list[float] = []
    lambdas_11: list[float] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("time_index"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            left_boundaries.append(float(parts[1]) / mu)
            right_boundaries.append(float(parts[2]) / mu)
            lambdas_00.append(float(parts[3]) * mu)
            lambdas_01.append(float(parts[4]) * mu)
            lambdas_11.append(float(parts[5]) * mu)

    if not left_boundaries:
        raise ValueError(f"No data found in {path}")

    # Force time to start at 0 and end at inf
    if left_boundaries[0] != 0.0:
        left_boundaries[0] = 0.0
    if right_boundaries[-1] != float("inf"):
        right_boundaries[-1] = float("inf")

    return {
        "left_boundary": np.array(left_boundaries, dtype=np.float64),
        "right_boundary": np.array(right_boundaries, dtype=np.float64),
        "lambda_00": np.array(lambdas_00, dtype=np.float64),
        "lambda_01": np.array(lambdas_01, dtype=np.float64),
        "lambda_11": np.array(lambdas_11, dtype=np.float64),
        "source_path": str(path),
    }
