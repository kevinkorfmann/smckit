"""ASMC: Ascertained Sequentially Markovian Coalescent.

Reimplementation of Palamara et al. (2018) Nature Genetics.
Infers pairwise coalescence times along the genome using a linear-time
HMM with B/U/D transition decomposition.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from smckit._core import SmcData
from smckit._provenance import sha256_file
from smckit.io._asmc import (
    DecodingQuantities,
    compute_undistinguished_counts,
    write_asmc_posterior_sums,
)
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    standard_upstream_metadata,
    warn_if_native_not_trusted,
)
from smckit.upstream import bootstrap as bootstrap_upstream
from smckit.upstream import status as upstream_status

logger = logging.getLogger(__name__)

# Rounding defaults (matching C++ implementation)
PRECISION = 2
MIN_GENETIC = 1e-10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def round_morgans(value: float, precision: int = PRECISION, min_val: float = MIN_GENETIC) -> float:
    """Round a genetic distance to a fixed number of significant figures.

    Matches the C++ ``asmc::roundMorgans`` function used to key
    precomputed transition vectors.

    Parameters
    ----------
    value : float
        Genetic distance in Morgans.
    precision : int
        One less than the number of significant figures.
    min_val : float
        Minimum returned value.

    Returns
    -------
    float
        Rounded value.
    """
    if value <= min_val:
        return min_val
    correction = 10.0 - float(precision)
    L10 = max(0.0, math.floor(math.log10(value)) + correction)
    factor = 10.0 ** (10.0 - L10)
    scaled = value * factor
    return math.floor(scaled + 0.5) / factor


def round_physical(value: int, precision: int = PRECISION) -> int:
    """Round a physical distance exactly as upstream ASMC."""
    if precision < 0:
        raise ValueError("precision must be non-negative")
    if value < -1:
        raise ValueError("physical distance must be at least -1")
    if value <= 1:
        return 1
    exponent = max(0, math.floor(math.log10(value)) - precision)
    factor = 10**exponent
    return int(math.floor(value / factor + 0.5)) * factor


# ---------------------------------------------------------------------------
# Pair observations
# ---------------------------------------------------------------------------


@dataclass
class PairObservations:
    """Encoded observations for a pair of haplotypes.

    Parameters
    ----------
    obs_is_zero : (L,) float32
        1.0 where both haplotypes carry the same allele (XOR=0, AND=0), else 0.
    obs_is_two : (L,) float32
        1.0 where both carry the minor/derived allele (AND=1), else 0.
    hap_i : int
        Index of first haplotype.
    hap_j : int
        Index of second haplotype.
    """

    obs_is_zero: np.ndarray
    obs_is_two: np.ndarray
    hap_i: int = 0
    hap_j: int = 0


def encode_pair(
    haplotypes: np.ndarray,
    i: int,
    j: int,
) -> PairObservations:
    """Encode a haplotype pair into ASMC observation vectors.

    Parameters
    ----------
    haplotypes : (n_haps, n_sites) uint8
    i, j : int
        Haplotype indices.

    Returns
    -------
    PairObservations
    """
    hap_i = haplotypes[i].astype(np.bool_)
    hap_j = haplotypes[j].astype(np.bool_)

    xor_bits = np.logical_xor(hap_i, hap_j)  # heterozygous
    and_bits = np.logical_and(hap_i, hap_j)  # both derived

    # C++ obsIsZero = !obsBits = !(XOR): 1 when NOT heterozygous
    # (includes both-ancestral AND both-derived)
    obs_is_zero = (~xor_bits).astype(np.float32)
    obs_is_two = and_bits.astype(np.float32)

    return PairObservations(
        obs_is_zero=obs_is_zero,
        obs_is_two=obs_is_two,
        hap_i=i,
        hap_j=j,
    )


# ---------------------------------------------------------------------------
# Emission preparation
# ---------------------------------------------------------------------------


def prepare_emissions(
    dq: DecodingQuantities,
    genetic_positions: np.ndarray,
    n_sites: int,
    *,
    use_csfs: bool = True,
    skip_csfs_distance: float = 0.0,
    fold_data: bool = True,
    decoding_sequence: bool = False,
    undistinguished_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-site emission vectors.

    Returns three arrays of shape (n_sites, states) that encode the
    emission model as::

        emission(pos, k) = e1[pos,k]
                         + e0m1[pos,k] * obs_is_zero[pos]
                         + e2m0[pos,k] * obs_is_two[pos]

    Parameters
    ----------
    dq : DecodingQuantities
    genetic_positions : (n_sites,) float32
    n_sites : int
    use_csfs : bool
        Whether to use CSFS emissions at eligible sites.
    skip_csfs_distance : float
        Minimum genetic distance (Morgans) between CSFS sites.
    fold_data : bool
        Whether to use folded CSFS tables.
    decoding_sequence : bool
        True for whole-genome sequencing, False for array data.
    undistinguished_counts : (n_sites, 3) int, optional
        Per-site undistinguished allele counts for obs types 0, 1, 2.

    Returns
    -------
    emission1 : (n_sites, states)
    emission0minus1 : (n_sites, states)
    emission2minus0 : (n_sites, states)
    """
    states = dq.states
    e1 = np.zeros((n_sites, states), dtype=np.float32)
    e0m1 = np.zeros((n_sites, states), dtype=np.float32)
    e2m0 = np.zeros((n_sites, states), dtype=np.float32)

    # Determine which sites use CSFS
    use_csfs_at = np.zeros(n_sites, dtype=np.bool_)
    if use_csfs and skip_csfs_distance < float("inf"):
        use_csfs_at[0] = True
        last_csfs_pos = 0.0
        for pos in range(1, n_sites):
            if genetic_positions[pos] - last_csfs_pos >= skip_csfs_distance:
                use_csfs_at[pos] = True
                last_csfs_pos = genetic_positions[pos]

    # Select emission tables
    if decoding_sequence:
        classic_e0 = dq.classic_emission[0]
        classic_e1 = dq.classic_emission[1]
    else:
        classic_e0 = dq.compressed_emission[0]
        classic_e1 = dq.compressed_emission[1]

    for pos in range(n_sites):
        if use_csfs_at[pos] and undistinguished_counts is not None:
            uc = undistinguished_counts[pos]
            if fold_data:
                csfs = (
                    dq.folded_ascertained_csfs_map if not decoding_sequence else dq.folded_csfs_map
                )
                # folded: 2 rows (obs=0, obs=1)
                uc1 = uc[1]
                if uc1 >= 0 and csfs[uc1] is not None:
                    e1[pos] = csfs[uc1][1]
                # else e1 stays 0

                uc0 = uc[0]
                if uc0 >= 0 and csfs[uc0] is not None:
                    e0m1[pos] = csfs[uc0][0] - e1[pos]
                else:
                    e0m1[pos] = -e1[pos]

                uc2 = uc[2]
                if uc2 >= 0 and csfs[uc2] is not None:
                    e2m0[pos] = (
                        csfs[uc2][0] - csfs[uc0][0]
                        if (uc0 >= 0 and csfs[uc0] is not None)
                        else csfs[uc2][0]
                    )
                else:
                    e2m0[pos] = -(csfs[uc0][0] if (uc0 >= 0 and csfs[uc0] is not None) else 0)
            else:
                csfs = dq.ascertained_csfs_map if not decoding_sequence else dq.csfs_map
                uc1 = uc[1]
                if uc1 >= 0 and csfs[uc1] is not None:
                    e1[pos] = csfs[uc1][1]

                uc0 = uc[0]
                e0_val = np.zeros(states, dtype=np.float32)
                if uc0 >= 0 and csfs[uc0] is not None:
                    e0_val = csfs[uc0][0]
                e0m1[pos] = e0_val - e1[pos]

                uc2 = uc[2]
                if uc2 >= 0 and csfs[uc2] is not None:
                    # Handle monomorphic derived folding
                    if uc2 == dq.csfs_samples - 2:
                        e2m0[pos] = csfs[0][0] - e0_val
                    else:
                        e2m0[pos] = csfs[uc2][2] - e0_val
                else:
                    e2m0[pos] = -e0_val
        else:
            # Non-CSFS site: use classic/compressed emission
            e1[pos] = classic_e1
            e0m1[pos] = classic_e0 - classic_e1
            # emission2 = emission0 for non-CSFS sites
            e2m0[pos] = 0.0

    return e1, e0m1, e2m0


# ---------------------------------------------------------------------------
# Forward algorithm (linear time via B/U/D decomposition)
# ---------------------------------------------------------------------------


def _get_transition_vectors(
    dq: DecodingQuantities,
    rec_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Look up B, U, D, row_ratio vectors for a given genetic distance.

    Falls back to nearest available key if exact match is missing.
    """
    if rec_dist in dq.D_vectors:
        return (
            dq.B_vectors[rec_dist],
            dq.U_vectors[rec_dist],
            dq.D_vectors[rec_dist],
            dq.row_ratio_vectors[rec_dist],
        )

    # Nearest key fallback
    keys = np.array(list(dq.D_vectors.keys()))
    idx = np.argmin(np.abs(keys - rec_dist))
    nearest = keys[idx]
    return (
        dq.B_vectors[nearest],
        dq.U_vectors[nearest],
        dq.D_vectors[nearest],
        dq.row_ratio_vectors[nearest],
    )


def _forward_transition(
    dq: DecodingQuantities,
    previous: np.ndarray,
    rec_dist: float,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: float,
    obs_is_two: float,
) -> np.ndarray:
    """Apply one ASMC forward transition without scaling."""
    states = dq.states
    B, U, D, _ = _get_transition_vectors(dq, rec_dist)
    alpha_c = np.cumsum(previous[::-1], dtype=np.float32)[::-1]
    result = np.empty(states, dtype=np.float32)
    upward = np.float32(0.0)
    for state in range(states):
        if state > 0:
            upward = U[state - 1] * previous[state - 1] + dq.column_ratios[state - 1] * upward
        transitioned = upward + D[state] * previous[state]
        if state < states - 1:
            transitioned += B[state] * alpha_c[state + 1]
        emission = (
            emission1[state]
            + emission0minus1[state] * obs_is_zero
            + emission2minus0[state] * obs_is_two
        )
        result[state] = emission * transitioned
    return result


def _backward_transition(
    dq: DecodingQuantities,
    next_beta: np.ndarray,
    rec_dist: float,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: float,
    obs_is_two: float,
) -> np.ndarray:
    """Apply one ASMC backward transition without scaling."""
    states = dq.states
    B, U, D, row_ratios = _get_transition_vectors(dq, rec_dist)
    emission = emission1 + emission0minus1 * obs_is_zero + emission2minus0 * obs_is_two
    vec = emission * next_beta
    upward = np.zeros(states, dtype=np.float32)
    for state in range(states - 2, -1, -1):
        upward[state] = U[state] * vec[state + 1] + row_ratios[state] * upward[state + 1]
    result = np.empty(states, dtype=np.float32)
    lower = np.float32(0.0)
    for state in range(states):
        if state > 0:
            lower += B[state - 1] * vec[state - 1]
        result[state] = lower + D[state] * vec[state] + upward[state]
    return result


@njit(cache=True)
def _forward_transition_numba(
    previous: np.ndarray,
    transition_index: int,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: np.float32,
    obs_is_two: np.float32,
    b_vectors: np.ndarray,
    u_vectors: np.ndarray,
    d_vectors: np.ndarray,
    column_ratios: np.ndarray,
    output: np.ndarray,
    cumulative: np.ndarray,
) -> None:
    states = previous.size
    cumulative[states - 1] = previous[states - 1]
    for state in range(states - 2, -1, -1):
        cumulative[state] = cumulative[state + 1] + previous[state]
    upward = np.float32(0.0)
    for state in range(states):
        if state > 0:
            upward = (
                u_vectors[transition_index, state - 1] * previous[state - 1]
                + column_ratios[state - 1] * upward
            )
        transitioned = upward + d_vectors[transition_index, state] * previous[state]
        if state < states - 1:
            transitioned += b_vectors[transition_index, state] * cumulative[state + 1]
        emission = (
            emission1[state]
            + emission0minus1[state] * obs_is_zero
            + emission2minus0[state] * obs_is_two
        )
        output[state] = emission * transitioned


@njit(cache=True)
def _backward_transition_numba(
    next_beta: np.ndarray,
    transition_index: int,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: np.float32,
    obs_is_two: np.float32,
    b_vectors: np.ndarray,
    u_vectors: np.ndarray,
    d_vectors: np.ndarray,
    row_ratios: np.ndarray,
    output: np.ndarray,
    vec: np.ndarray,
    upward: np.ndarray,
) -> None:
    states = next_beta.size
    for state in range(states):
        emission = (
            emission1[state]
            + emission0minus1[state] * obs_is_zero
            + emission2minus0[state] * obs_is_two
        )
        vec[state] = emission * next_beta[state]
        upward[state] = np.float32(0.0)
    for state in range(states - 2, -1, -1):
        upward[state] = (
            u_vectors[transition_index, state] * vec[state + 1]
            + row_ratios[transition_index, state] * upward[state + 1]
        )
    lower = np.float32(0.0)
    for state in range(states):
        if state > 0:
            lower += b_vectors[transition_index, state - 1] * vec[state - 1]
        output[state] = lower + d_vectors[transition_index, state] * vec[state] + upward[state]


@njit(cache=True)
def _decode_pair_numba(
    initial_state_prob: np.ndarray,
    column_ratios: np.ndarray,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: np.ndarray,
    obs_is_two: np.ndarray,
    total_transition_indices: np.ndarray,
    marker_transition_indices: np.ndarray,
    forward_inter_transition_indices: np.ndarray,
    backward_marker_transition_indices: np.ndarray,
    backward_inter_transition_indices: np.ndarray,
    homozygous_emissions: np.ndarray,
    homozygous_indices: np.ndarray,
    b_vectors: np.ndarray,
    u_vectors: np.ndarray,
    d_vectors: np.ndarray,
    row_ratios: np.ndarray,
    scaling_skip: int,
    decoding_sequence: bool,
) -> np.ndarray:
    """Compiled single-pair ASMC forward/backward decoder."""
    sites, states = emission1.shape
    alpha = np.zeros((sites, states), dtype=np.float32)
    beta = np.zeros((sites, states), dtype=np.float32)
    cumulative = np.empty(states, dtype=np.float32)
    scratch = np.empty(states, dtype=np.float32)
    vec = np.empty(states, dtype=np.float32)
    upward = np.empty(states, dtype=np.float32)

    for state in range(states):
        emission = (
            emission1[0, state]
            + emission0minus1[0, state] * obs_is_zero[0]
            + emission2minus0[0, state] * obs_is_two[0]
        )
        alpha[0, state] = initial_state_prob[state] * emission
    total = np.float32(0.0)
    for state in range(states):
        total += alpha[0, state]
    if total != 0:
        for state in range(states):
            alpha[0, state] /= total

    for site in range(1, sites):
        if decoding_sequence:
            homozygous = homozygous_emissions[homozygous_indices[site]]
            _forward_transition_numba(
                alpha[site - 1],
                forward_inter_transition_indices[site],
                homozygous,
                homozygous,
                homozygous,
                np.float32(0.0),
                np.float32(0.0),
                b_vectors,
                u_vectors,
                d_vectors,
                column_ratios,
                scratch,
                cumulative,
            )
            # ASMC 1.4's batched sequence implementation stores the
            # inter-marker intermediate in the destination slot and then
            # assigns that slot back onto the preceding alpha map before the
            # marker step. Propagation still uses the marker alpha, but the
            # posterior buffer retains this inter-marker value at site-1.
            alpha[site - 1] = scratch
            _forward_transition_numba(
                scratch,
                marker_transition_indices[site],
                emission1[site],
                emission0minus1[site],
                emission2minus0[site],
                obs_is_zero[site],
                obs_is_two[site],
                b_vectors,
                u_vectors,
                d_vectors,
                column_ratios,
                alpha[site],
                cumulative,
            )
        else:
            _forward_transition_numba(
                alpha[site - 1],
                total_transition_indices[site],
                emission1[site],
                emission0minus1[site],
                emission2minus0[site],
                obs_is_zero[site],
                obs_is_two[site],
                b_vectors,
                u_vectors,
                d_vectors,
                column_ratios,
                alpha[site],
                cumulative,
            )
        if site % scaling_skip == 0:
            total = np.float32(0.0)
            for state in range(states):
                total += alpha[site, state]
            if total != 0:
                for state in range(states):
                    alpha[site, state] /= total

    for state in range(states):
        beta[sites - 1, state] = np.float32(1.0 / states)
    for site in range(sites - 2, -1, -1):
        if decoding_sequence:
            homozygous = homozygous_emissions[homozygous_indices[site + 1]]
            _backward_transition_numba(
                beta[site + 1],
                backward_inter_transition_indices[site],
                homozygous,
                homozygous,
                homozygous,
                np.float32(0.0),
                np.float32(0.0),
                b_vectors,
                u_vectors,
                d_vectors,
                row_ratios,
                scratch,
                vec,
                upward,
            )
            # The corresponding Eigen map assignment in upstream overwrites
            # the following site's stored beta with the inter-marker result.
            beta[site + 1] = scratch
            _backward_transition_numba(
                scratch,
                backward_marker_transition_indices[site],
                emission1[site + 1],
                emission0minus1[site + 1],
                emission2minus0[site + 1],
                obs_is_zero[site + 1],
                obs_is_two[site + 1],
                b_vectors,
                u_vectors,
                d_vectors,
                row_ratios,
                beta[site],
                vec,
                upward,
            )
        else:
            _backward_transition_numba(
                beta[site + 1],
                total_transition_indices[site + 1],
                emission1[site + 1],
                emission0minus1[site + 1],
                emission2minus0[site + 1],
                obs_is_zero[site + 1],
                obs_is_two[site + 1],
                b_vectors,
                u_vectors,
                d_vectors,
                row_ratios,
                beta[site],
                vec,
                upward,
            )
        if site % scaling_skip == 0:
            total = np.float32(0.0)
            for state in range(states):
                total += beta[site, state]
            if total != 0:
                for state in range(states):
                    beta[site, state] /= total

    posterior = np.empty((sites, states), dtype=np.float32)
    for site in range(sites):
        total = np.float32(0.0)
        for state in range(states):
            posterior[site, state] = alpha[site, state] * beta[site, state]
            total += posterior[site, state]
        if total != 0:
            for state in range(states):
                posterior[site, state] /= total
    return posterior


def _nearest_transition_indices(
    transition_keys: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Map rounded distances to the nearest decoding-quantity transition."""
    right = np.searchsorted(transition_keys, distances)
    right = np.clip(right, 0, transition_keys.size - 1)
    left = np.clip(right - 1, 0, transition_keys.size - 1)
    choose_left = np.abs(distances - transition_keys[left]) <= np.abs(
        transition_keys[right] - distances
    )
    return np.where(choose_left, left, right).astype(np.int32)


def _compiled_decode_tables(
    dq: DecodingQuantities,
    genetic_positions: np.ndarray,
    physical_positions: np.ndarray,
    rec_rates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize compact transition indices used by the Numba kernel."""
    transition_keys = np.asarray(sorted(dq.D_vectors), dtype=np.float64)
    if transition_keys.size == 0:
        raise ValueError("ASMC decoding quantities contain no transition vectors.")
    b_vectors = np.stack([dq.B_vectors[key] for key in transition_keys]).astype(np.float32)
    u_vectors = np.stack([dq.U_vectors[key] for key in transition_keys]).astype(np.float32)
    d_vectors = np.stack([dq.D_vectors[key] for key in transition_keys]).astype(np.float32)
    row_ratios = np.stack([dq.row_ratio_vectors[key] for key in transition_keys]).astype(
        np.float32
    )
    sites = genetic_positions.size
    total_distances = np.full(sites, MIN_GENETIC, dtype=np.float64)
    total_distances[1:] = [round_morgans(float(value)) for value in np.diff(genetic_positions)]
    marker_distances = np.asarray(
        [round_morgans(float(value)) for value in rec_rates],
        dtype=np.float64,
    )
    forward_inter = np.full(sites, MIN_GENETIC, dtype=np.float64)
    forward_inter[1:] = [
        round_morgans(total_distances[index] - marker_distances[index])
        for index in range(1, sites)
    ]
    backward_inter = np.full(sites, MIN_GENETIC, dtype=np.float64)
    backward_inter[:-1] = [
        round_morgans(total_distances[index + 1] - marker_distances[index])
        for index in range(sites - 1)
    ]

    homozygous_keys = np.asarray(
        sorted(dq.homozygous_emission_map),
        dtype=np.int64,
    )
    if homozygous_keys.size:
        homozygous_emissions = np.stack(
            [dq.homozygous_emission_map[int(key)] for key in homozygous_keys]
        ).astype(np.float32)
        gaps = np.ones(sites, dtype=np.int64)
        gaps[1:] = [round_physical(int(value - 1)) for value in np.diff(physical_positions)]
        homozygous_lookup = {int(key): index for index, key in enumerate(homozygous_keys)}
        missing = sorted({int(gap) for gap in gaps[1:] if int(gap) not in homozygous_lookup})
        if missing:
            raise ValueError(
                "Decoding quantities lack sequence emissions for physical gaps: "
                + ", ".join(map(str, missing[:10]))
            )
        homozygous_indices = np.asarray(
            [homozygous_lookup.get(int(gap), 0) for gap in gaps],
            dtype=np.int32,
        )
    else:
        homozygous_emissions = np.empty((0, dq.states), dtype=np.float32)
        homozygous_indices = np.zeros(sites, dtype=np.int32)

    return {
        "b_vectors": b_vectors,
        "u_vectors": u_vectors,
        "d_vectors": d_vectors,
        "row_ratios": row_ratios,
        "total_indices": _nearest_transition_indices(
            transition_keys,
            total_distances,
        ),
        "marker_indices": _nearest_transition_indices(
            transition_keys,
            marker_distances,
        ),
        "forward_inter_indices": _nearest_transition_indices(
            transition_keys,
            forward_inter,
        ),
        "backward_marker_indices": _nearest_transition_indices(
            transition_keys,
            marker_distances,
        ),
        "backward_inter_indices": _nearest_transition_indices(
            transition_keys,
            backward_inter,
        ),
        "homozygous_emissions": homozygous_emissions,
        "homozygous_indices": homozygous_indices,
    }


def forward(
    dq: DecodingQuantities,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: np.ndarray,
    obs_is_two: np.ndarray,
    genetic_positions: np.ndarray,
    from_pos: int = 0,
    to_pos: int | None = None,
    scaling_skip: int = 1,
    physical_positions: np.ndarray | None = None,
    rec_rates: np.ndarray | None = None,
    decoding_sequence: bool = False,
) -> np.ndarray:
    """Scaled forward algorithm with B/U/D transition decomposition.

    Runs in O(states * L) time instead of O(states^2 * L).

    Parameters
    ----------
    dq : DecodingQuantities
    emission1, emission0minus1, emission2minus0 : (L, states)
    obs_is_zero, obs_is_two : (L,) float32
    genetic_positions : (L,) float32 in Morgans
    from_pos, to_pos : int
        Subsequence range.
    scaling_skip : int
        Apply scaling every this many positions.
    physical_positions : (L,) int, optional
        Marker positions required for sequence decoding.
    rec_rates : (L,) float, optional
        Per-base recombination rates in Morgans required for sequence decoding.
    decoding_sequence : bool
        Include the homozygous inter-marker sequence emissions used by ASMC.

    Returns
    -------
    alpha : (L, states) float32
        Scaled forward probabilities.
    """
    if to_pos is None:
        to_pos = len(genetic_positions)
    if scaling_skip < 1:
        raise ValueError("scaling_skip must be at least 1")
    if decoding_sequence and (physical_positions is None or rec_rates is None):
        raise ValueError("Sequence decoding requires physical_positions and per-base rec_rates.")

    states = dq.states
    # Use float32 to match C++ SIMD precision exactly
    alpha = np.zeros((to_pos, states), dtype=np.float32)

    init_prob = dq.initial_state_prob  # float32

    # Initialize at from_pos
    em_vec = (
        emission1[from_pos]
        + emission0minus1[from_pos] * obs_is_zero[from_pos]
        + emission2minus0[from_pos] * obs_is_two[from_pos]
    )
    alpha[from_pos] = init_prob * em_vec

    # Scale initial
    s = alpha[from_pos].sum()
    if s != 0:
        alpha[from_pos] /= s

    last_gen_pos = genetic_positions[from_pos]

    for pos in range(from_pos + 1, to_pos):
        rec_dist = round_morgans(float(genetic_positions[pos] - last_gen_pos))

        prev = alpha[pos - 1]
        if decoding_sequence:
            rate = round_morgans(float(rec_rates[pos]))
            gap = round_physical(int(physical_positions[pos] - physical_positions[pos - 1] - 1))
            try:
                homozygous = dq.homozygous_emission_map[gap]
            except KeyError as exc:
                raise ValueError(
                    f"Decoding quantities lack sequence emission for physical gap {gap}."
                ) from exc
            inter_marker_dist = round_morgans(rec_dist - rate)
            prev = _forward_transition(
                dq,
                prev,
                inter_marker_dist,
                homozygous,
                homozygous,
                homozygous,
                0.0,
                0.0,
            )
            alpha[pos] = _forward_transition(
                dq,
                prev,
                rate,
                emission1[pos],
                emission0minus1[pos],
                emission2minus0[pos],
                float(obs_is_zero[pos]),
                float(obs_is_two[pos]),
            )
        else:
            alpha[pos] = _forward_transition(
                dq,
                prev,
                rec_dist,
                emission1[pos],
                emission0minus1[pos],
                emission2minus0[pos],
                float(obs_is_zero[pos]),
                float(obs_is_two[pos]),
            )

        # Scale (unconditionally, matching C++ behavior)
        if pos % scaling_skip == 0:
            s = alpha[pos].sum()
            if s != 0:
                alpha[pos] /= s

        last_gen_pos = genetic_positions[pos]

    return alpha


# ---------------------------------------------------------------------------
# Backward algorithm (linear time via B/U/D decomposition)
# ---------------------------------------------------------------------------


def backward(
    dq: DecodingQuantities,
    emission1: np.ndarray,
    emission0minus1: np.ndarray,
    emission2minus0: np.ndarray,
    obs_is_zero: np.ndarray,
    obs_is_two: np.ndarray,
    genetic_positions: np.ndarray,
    from_pos: int = 0,
    to_pos: int | None = None,
    scaling_skip: int = 1,
    physical_positions: np.ndarray | None = None,
    rec_rates: np.ndarray | None = None,
    decoding_sequence: bool = False,
) -> np.ndarray:
    """Scaled backward algorithm with B/U/D transition decomposition.

    Parameters
    ----------
    dq : DecodingQuantities
    emission1, emission0minus1, emission2minus0 : (L, states)
    obs_is_zero, obs_is_two : (L,) float32
    genetic_positions : (L,) float32 in Morgans
    from_pos, to_pos : int
    scaling_skip : int

    Returns
    -------
    beta : (L, states) float32
        Scaled backward probabilities.
    """
    if to_pos is None:
        to_pos = len(genetic_positions)
    if scaling_skip < 1:
        raise ValueError("scaling_skip must be at least 1")
    if decoding_sequence and (physical_positions is None or rec_rates is None):
        raise ValueError("Sequence decoding requires physical_positions and per-base rec_rates.")

    states = dq.states
    # Use float32 to match C++ SIMD precision exactly
    beta = np.zeros((to_pos, states), dtype=np.float32)

    # Initialize at to_pos - 1
    beta[to_pos - 1, :] = 1.0
    s = beta[to_pos - 1].sum()
    if s != 0:
        beta[to_pos - 1] /= s

    last_gen_pos = genetic_positions[to_pos - 1]

    for pos in range(to_pos - 2, from_pos - 1, -1):
        rec_dist = round_morgans(float(last_gen_pos - genetic_positions[pos]))

        next_beta = beta[pos + 1]
        if decoding_sequence:
            # Preserve upstream ASMC's backward ordering and rate index.
            rate = round_morgans(float(rec_rates[pos]))
            gap = round_physical(int(physical_positions[pos + 1] - physical_positions[pos] - 1))
            try:
                homozygous = dq.homozygous_emission_map[gap]
            except KeyError as exc:
                raise ValueError(
                    f"Decoding quantities lack sequence emission for physical gap {gap}."
                ) from exc
            inter_marker_dist = round_morgans(rec_dist - rate)
            next_beta = _backward_transition(
                dq,
                next_beta,
                inter_marker_dist,
                homozygous,
                homozygous,
                homozygous,
                0.0,
                0.0,
            )
            beta[pos] = _backward_transition(
                dq,
                next_beta,
                rate,
                emission1[pos + 1],
                emission0minus1[pos + 1],
                emission2minus0[pos + 1],
                float(obs_is_zero[pos + 1]),
                float(obs_is_two[pos + 1]),
            )
        else:
            beta[pos] = _backward_transition(
                dq,
                next_beta,
                rec_dist,
                emission1[pos + 1],
                emission0minus1[pos + 1],
                emission2minus0[pos + 1],
                float(obs_is_zero[pos + 1]),
                float(obs_is_two[pos + 1]),
            )

        # Scale (unconditionally, matching C++ behavior)
        if pos % scaling_skip == 0:
            s = beta[pos].sum()
            if s != 0:
                beta[pos] /= s

        last_gen_pos = genetic_positions[pos]

    return beta


# ---------------------------------------------------------------------------
# Posterior computation
# ---------------------------------------------------------------------------


def compute_posteriors(
    alpha: np.ndarray,
    beta: np.ndarray,
    from_pos: int = 0,
    to_pos: int | None = None,
) -> np.ndarray:
    """Compute normalized posterior probabilities from alpha and beta.

    Parameters
    ----------
    alpha, beta : (L, states) float32
    from_pos, to_pos : int

    Returns
    -------
    posteriors : (L, states) float32
        P(state=k | data) at each position.
    """
    if to_pos is None:
        to_pos = alpha.shape[0]

    posteriors = alpha[from_pos:to_pos] * beta[from_pos:to_pos]
    row_sums = posteriors.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    posteriors /= row_sums
    return posteriors


def posterior_mean_tmrca(
    posteriors: np.ndarray,
    expected_times: np.ndarray,
) -> np.ndarray:
    """Compute posterior mean TMRCA at each site.

    Parameters
    ----------
    posteriors : (L, states)
    expected_times : (states,)

    Returns
    -------
    means : (L,) float32
    """
    return posteriors @ expected_times


def posterior_map_tmrca(
    posteriors: np.ndarray,
    expected_times: np.ndarray,
    initial_state_prob: np.ndarray,
) -> np.ndarray:
    """Compute upstream ASMC ``perPairMAP`` state indices at each site.

    Upstream ASMC stores ``perPairMAP`` as the posterior argmax state index.
    This differs from the segment-level ``getMAP`` helper in the C++ code,
    which divides by the initial-state prior before taking the argmax.

    Parameters
    ----------
    posteriors : (L, states)
    expected_times : (states,)
    initial_state_prob : (states,)

    Returns
    -------
    map_states : (L,) int32
    """
    return posteriors.argmax(axis=1).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class AsmcResult:
    """Results from an ASMC decoding run."""

    expected_times: np.ndarray  # (states,) time per state
    discretization: np.ndarray  # (states+1,) time boundaries
    sum_of_posteriors: np.ndarray  # (n_sites, states) aggregated posteriors
    sum_of_posteriors_major_minor: dict[str, np.ndarray] = field(default_factory=dict)
    per_pair_posteriors: list[np.ndarray] = field(default_factory=list)
    per_pair_posterior_means: list[np.ndarray] = field(default_factory=list)
    per_pair_maps: list[np.ndarray] = field(default_factory=list)
    per_pair_indices: list[tuple[int, int]] = field(default_factory=list)
    per_pair_labels: list[tuple[str, str]] = field(default_factory=list)
    min_posterior_means: np.ndarray | None = None
    argmin_posterior_means: np.ndarray | None = None
    min_maps: np.ndarray | None = None
    argmin_maps: np.ndarray | None = None
    n_pairs_decoded: int = 0


def _all_haplotype_pairs(n_haplotypes: int) -> list[tuple[int, int]]:
    """Return pairs in the same order as upstream ``decodePairs()``."""
    return [(hap_b, hap_a) for hap_a in range(n_haplotypes) for hap_b in range(hap_a)]


def _validate_pairs(
    pairs: list[tuple[int, int]],
    n_haplotypes: int,
) -> list[tuple[int, int]]:
    validated: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each ASMC pair must contain exactly two haplotype indices.")
        first, second = (int(pair[0]), int(pair[1]))
        if first == second:
            raise ValueError("ASMC cannot decode a haplotype against itself.")
        if first < 0 or second < 0 or first >= n_haplotypes or second >= n_haplotypes:
            raise IndexError(f"ASMC pair {(first, second)} is outside 0..{n_haplotypes - 1}.")
        validated.append((first, second))
    if not validated:
        raise ValueError("At least one ASMC pair is required.")
    return validated


def _select_job_pairs(
    pairs: list[tuple[int, int]],
    jobs: int,
    job_index: int,
) -> list[tuple[int, int]]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if job_index < 1 or job_index > jobs:
        raise ValueError("job_index must be between 1 and jobs inclusive")
    start = len(pairs) * (job_index - 1) // jobs
    end = len(pairs) * job_index // jobs
    selected = pairs[start:end]
    if not selected:
        raise ValueError(f"ASMC job {job_index}/{jobs} contains no pairs; reduce the job count.")
    return selected


def _pair_label(data: SmcData, haplotype: int) -> str:
    samples = data.uns.get("samples", [])
    individual = haplotype // 2
    if individual < len(samples):
        sample = samples[individual]
        identifier = sample.get("ind_id") or sample.get("fam_id") or str(individual)
    else:
        identifier = str(individual)
    return f"{identifier}_{haplotype % 2 + 1}"


def _native_haplotypes_and_counts(
    data: SmcData,
    *,
    fold_data: bool,
    dq: DecodingQuantities,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return data coded for the selected folded/ancestral emission semantics."""
    haplotypes = np.asarray(data.uns["haplotypes"], dtype=np.uint8).copy()
    loaded_folded = bool(data.uns.get("fold_to_minor", True))
    flipped = np.asarray(
        data.uns.get("site_was_flipped", np.zeros(haplotypes.shape[1], dtype=bool)),
        dtype=bool,
    )
    if loaded_folded:
        haplotypes[:, flipped] = 1 - haplotypes[:, flipped]
    if fold_data:
        fold_mask = haplotypes.sum(axis=0) > haplotypes.shape[0] / 2
        haplotypes[:, fold_mask] = 1 - haplotypes[:, fold_mask]

    strategy = str(data.uns.get("undistinguished_strategy", "expected"))
    counts = compute_undistinguished_counts(
        haplotypes,
        dq.csfs_samples,
        fold_to_minor=fold_data,
        strategy=strategy,
        random_seed=random_seed,
        allow_undersampled=True,
    )
    return haplotypes, counts


def _write_native_asmc_outputs(
    output_prefix: str | Path,
    *,
    posterior_sums: np.ndarray | None,
    major_minor_sums: dict[str, np.ndarray],
    posterior_means: list[np.ndarray],
    maps: list[np.ndarray],
    pair_indices: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, str]] = []
    if posterior_sums is not None:
        path = write_asmc_posterior_sums(
            Path(f"{prefix}.sumOverPairs.gz"),
            posterior_sums,
        )
        written.append((path, "asmc-posterior-sums"))
    for genotype, values in major_minor_sums.items():
        path = write_asmc_posterior_sums(
            Path(f"{prefix}.{genotype}.sumOverPairs.gz"),
            values,
        )
        written.append((path, f"asmc-posterior-sums-{genotype}"))
    if posterior_means:
        path = write_asmc_posterior_sums(
            Path(f"{prefix}.perPairPosteriorMeans.gz"),
            np.asarray(posterior_means),
        )
        written.append((path, "asmc-per-pair-posterior-means"))
    if maps:
        path = write_asmc_posterior_sums(
            Path(f"{prefix}.perPairMAP.gz"),
            np.asarray(maps, dtype=np.int32),
        )
        written.append((path, "asmc-per-pair-map"))
    pair_path = Path(f"{prefix}.pairs.tsv")
    with pair_path.open("w", encoding="utf-8") as handle:
        handle.write("pair_index\thaplotype_a\thaplotype_b\n")
        for pair_index, (first, second) in enumerate(pair_indices):
            handle.write(f"{pair_index}\t{first}\t{second}\n")
    written.append((pair_path, "asmc-pair-index"))
    return [
        {"path": str(path), "sha256": sha256_file(path), "kind": kind} for path, kind in written
    ]


def asmc(
    data: SmcData,
    *,
    pairs: list[tuple[int, int]] | None = None,
    mode: str = "array",
    fold_data: bool = True,
    use_ancestral: bool = False,
    skip_csfs_distance: float = 0.0,
    scaling_skip: int = 1,
    posterior_sums: bool = True,
    major_minor_posterior_sums: bool = False,
    store_per_pair_posterior: bool = False,
    store_per_pair_posterior_mean: bool = True,
    store_per_pair_map: bool = False,
    jobs: int = 1,
    job_index: int = 1,
    from_pos: int = 0,
    to_pos: int | None = None,
    cm_burn_in: float = 0.5,
    random_seed: int | None = None,
    output_prefix: str | Path | None = None,
    implementation: str = "auto",
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run ASMC pairwise coalescence time inference.

    Parameters
    ----------
    data : SmcData
        Input data from ``smckit.io.read_asmc()``.
    pairs : list of (int, int), optional
        Haplotype index pairs to decode. If None, decodes all unique pairs.
    mode : {"array", "sequence"}
        "array" uses compressed/ascertained emissions;
        "sequence" uses classic emissions.
    fold_data : bool
        Use folded CSFS tables.
    use_ancestral : bool
        Treat input allele 1 as ancestral, equivalent to upstream
        ``--useAncestral``. This implies ``fold_data=False``.
    skip_csfs_distance : float
        Minimum distance between CSFS sites (Morgans).
    scaling_skip : int
        Apply scaling every this many positions.
    posterior_sums : bool
        Aggregate posterior state probabilities over all selected pairs.
    major_minor_posterior_sums : bool
        Also partition posterior sums into 00, 01, and 11 carrier classes.
    store_per_pair_posterior : bool
        Retain each complete site-by-state posterior matrix.
    store_per_pair_posterior_mean : bool
        Store per-pair posterior mean TMRCA.
    store_per_pair_map : bool
        Store per-pair MAP state indices, matching upstream ASMC ``perPairMAP``.
    jobs, job_index : int
        Deterministically partition all-pairs decoding into mergeable jobs.
    from_pos, to_pos : int
        Return this half-open marker interval. The HMM includes
        ``cm_burn_in`` flanking sequence on each side.
    cm_burn_in : float
        Genetic burn-in distance in centimorgans for interval decoding.
    random_seed : int, optional
        Seed for stochastic undistinguished-count construction.
    output_prefix : str or Path, optional
        Persist original-compatible posterior and per-pair output files.
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo decoder.
        ``"upstream"`` executes the preserved ASMC binary. ``"auto"`` resolves
        using the capability registry.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["asmc"]``.
    """
    started = time.perf_counter()
    implementation = normalize_implementation(implementation)
    if mode not in {"array", "sequence"}:
        raise ValueError("mode must be one of: array, sequence")
    if use_ancestral:
        fold_data = False
    if skip_csfs_distance < 0:
        raise ValueError("skip_csfs_distance must be non-negative")
    if scaling_skip < 1:
        raise ValueError("scaling_skip must be at least 1")
    requested_capabilities: set[str] = set()
    if mode == "sequence":
        requested_capabilities.add("sequence")
    if pairs is not None:
        requested_capabilities.add("pair_selection")
    if use_ancestral or not fold_data:
        requested_capabilities.add("ancestral")
    if np.isinf(skip_csfs_distance):
        requested_capabilities.add("compressed")
    elif skip_csfs_distance > 0:
        requested_capabilities.add("csfs_spacing")
    if major_minor_posterior_sums:
        requested_capabilities.add("major_minor_sums")
    if store_per_pair_posterior:
        requested_capabilities.add("full_posteriors")
    if store_per_pair_posterior_mean or store_per_pair_map:
        requested_capabilities.add("per_pair_outputs")
    if jobs != 1 or job_index != 1:
        requested_capabilities.add("job_partitioning")
    if from_pos != 0 or to_pos is not None:
        requested_capabilities.add("interval_decode")
    if output_prefix is not None:
        requested_capabilities.add("output")
    if upstream_options:
        requested_capabilities.add("upstream_options")
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("asmc"),
        method_name="asmc",
        requested_capabilities=requested_capabilities or None,
    )
    warn_if_native_not_trusted("asmc", implementation_used)
    if implementation_used == "upstream":
        return _asmc_upstream(
            data,
            pairs=pairs,
            mode=mode,
            fold_data=fold_data,
            skip_csfs_distance=skip_csfs_distance,
            posterior_sums=posterior_sums,
            major_minor_posterior_sums=major_minor_posterior_sums,
            store_per_pair_posterior=store_per_pair_posterior,
            store_per_pair_posterior_mean=store_per_pair_posterior_mean,
            store_per_pair_map=store_per_pair_map,
            jobs=jobs,
            job_index=job_index,
            from_pos=from_pos,
            to_pos=to_pos,
            cm_burn_in=cm_burn_in,
            output_prefix=output_prefix,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )
    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported asmc native_options keys: {unsupported}")

    genetic_positions = np.asarray(data.uns["genetic_positions"])
    physical_positions = np.asarray(data.uns["physical_positions"])
    if "rec_rates" in data.uns:
        rec_rates = np.asarray(data.uns["rec_rates"])
    else:
        rec_rates = np.zeros_like(genetic_positions, dtype=np.float64)
        genetic_distance = np.diff(genetic_positions)
        physical_distance = np.diff(physical_positions)
        np.divide(
            genetic_distance,
            physical_distance,
            out=rec_rates[1:],
            where=physical_distance > 0,
        )
        if rec_rates.size > 1:
            rec_rates[0] = rec_rates[1]
    dq: DecodingQuantities = data.uns["decoding_quantities"]
    use_csfs = not np.isinf(skip_csfs_distance) and dq.csfs_samples >= 2
    if use_csfs and dq.csfs_samples > np.asarray(data.uns["haplotypes"]).shape[0]:
        raise ValueError(
            f"ASMC CSFS requires {dq.csfs_samples} haplotypes but input contains "
            f"{np.asarray(data.uns['haplotypes']).shape[0]}; use "
            "skip_csfs_distance=float('inf') for compressed decoding."
        )
    haplotypes, undist_counts = _native_haplotypes_and_counts(
        data,
        fold_data=fold_data,
        dq=dq,
        random_seed=random_seed,
    )
    n_haps, n_sites = haplotypes.shape
    decoding_sequence = mode == "sequence"
    if to_pos is None or to_pos == 0:
        to_pos = n_sites
    if from_pos < 0 or from_pos >= to_pos or to_pos > n_sites:
        raise ValueError(
            f"Require 0 <= from_pos < to_pos <= {n_sites}, got {from_pos} < {to_pos}."
        )
    if cm_burn_in < 0:
        raise ValueError("cm_burn_in must be non-negative")

    decode_from = from_pos
    decode_to = to_pos
    if cm_burn_in > 0:
        while (
            decode_from > 0
            and 100.0 * (genetic_positions[from_pos] - genetic_positions[decode_from]) < cm_burn_in
        ):
            decode_from -= 1
        while (
            decode_to < n_sites
            and 100.0 * (genetic_positions[decode_to - 1] - genetic_positions[to_pos - 1])
            < cm_burn_in
        ):
            decode_to += 1

    logger.info("ASMC: %d haplotypes, %d sites, %d states", n_haps, n_sites, dq.states)

    emission1, emission0minus1, emission2minus0 = prepare_emissions(
        dq,
        genetic_positions,
        n_sites,
        use_csfs=use_csfs,
        skip_csfs_distance=skip_csfs_distance,
        fold_data=fold_data,
        decoding_sequence=decoding_sequence,
        undistinguished_counts=undist_counts,
    )
    decode_slice = slice(decode_from, decode_to)
    compiled_tables = _compiled_decode_tables(
        dq,
        genetic_positions[decode_slice],
        physical_positions[decode_slice],
        rec_rates[decode_slice],
    )

    if pairs is None:
        pairs = _all_haplotype_pairs(n_haps)
    pairs = _validate_pairs(pairs, n_haps)
    pairs = _select_job_pairs(pairs, jobs, job_index)

    result_sites = to_pos - from_pos
    sum_of_posteriors = np.zeros((result_sites, dq.states), dtype=np.float32)
    major_minor_sums = (
        {genotype: np.zeros_like(sum_of_posteriors) for genotype in ("00", "01", "11")}
        if major_minor_posterior_sums
        else {}
    )
    per_pair_posteriors: list[np.ndarray] = []
    per_pair_means: list[np.ndarray] = []
    per_pair_maps: list[np.ndarray] = []
    per_pair_indices: list[tuple[int, int]] = []
    per_pair_labels: list[tuple[str, str]] = []

    for pair_idx, (i, j) in enumerate(pairs):
        if (pair_idx + 1) % 100 == 0 or pair_idx == 0:
            logger.info("  Decoding pair %d/%d (haps %d, %d)", pair_idx + 1, len(pairs), i, j)

        obs = encode_pair(haplotypes, i, j)

        decoded = _decode_pair_numba(
            np.asarray(dq.initial_state_prob, dtype=np.float32),
            np.asarray(dq.column_ratios, dtype=np.float32),
            np.asarray(emission1[decode_slice], dtype=np.float32),
            np.asarray(emission0minus1[decode_slice], dtype=np.float32),
            np.asarray(emission2minus0[decode_slice], dtype=np.float32),
            np.asarray(obs.obs_is_zero[decode_slice], dtype=np.float32),
            np.asarray(obs.obs_is_two[decode_slice], dtype=np.float32),
            compiled_tables["total_indices"],
            compiled_tables["marker_indices"],
            compiled_tables["forward_inter_indices"],
            compiled_tables["backward_marker_indices"],
            compiled_tables["backward_inter_indices"],
            compiled_tables["homozygous_emissions"],
            compiled_tables["homozygous_indices"],
            compiled_tables["b_vectors"],
            compiled_tables["u_vectors"],
            compiled_tables["d_vectors"],
            compiled_tables["row_ratios"],
            scaling_skip=scaling_skip,
            decoding_sequence=decoding_sequence,
        )
        result_from = from_pos - decode_from
        result_to = result_from + result_sites
        posteriors = decoded[result_from:result_to]
        if posterior_sums or major_minor_posterior_sums:
            sum_of_posteriors += posteriors
        if major_minor_posterior_sums:
            obs_zero = obs.obs_is_zero[from_pos:to_pos].astype(bool)
            obs_two = obs.obs_is_two[from_pos:to_pos].astype(bool)
            masks = {
                "00": obs_zero & ~obs_two,
                "01": ~obs_zero,
                "11": obs_two,
            }
            for genotype, mask in masks.items():
                major_minor_sums[genotype] += posteriors * mask[:, None]
        per_pair_indices.append((i, j))
        per_pair_labels.append((_pair_label(data, i), _pair_label(data, j)))

        if store_per_pair_posterior:
            per_pair_posteriors.append(posteriors.copy())

        if store_per_pair_posterior_mean:
            per_pair_means.append(posterior_mean_tmrca(posteriors, dq.expected_times))

        if store_per_pair_map:
            per_pair_maps.append(
                posterior_map_tmrca(posteriors, dq.expected_times, dq.initial_state_prob)
            )

    result = AsmcResult(
        expected_times=dq.expected_times,
        discretization=dq.discretization,
        sum_of_posteriors=sum_of_posteriors,
        sum_of_posteriors_major_minor=major_minor_sums,
        per_pair_posteriors=per_pair_posteriors,
        per_pair_posterior_means=per_pair_means,
        per_pair_maps=per_pair_maps,
        per_pair_indices=per_pair_indices,
        per_pair_labels=per_pair_labels,
        n_pairs_decoded=len(pairs),
    )
    if per_pair_means:
        means_matrix = np.asarray(per_pair_means)
        result.min_posterior_means = means_matrix.min(axis=0)
        result.argmin_posterior_means = means_matrix.argmin(axis=0)
    if per_pair_maps:
        maps_matrix = np.asarray(per_pair_maps)
        result.min_maps = maps_matrix.min(axis=0)
        result.argmin_maps = maps_matrix.argmin(axis=0)

    artifacts: list[dict[str, Any]] = []
    if output_prefix is not None:
        artifacts = _write_native_asmc_outputs(
            output_prefix,
            posterior_sums=sum_of_posteriors if posterior_sums else None,
            major_minor_sums=major_minor_sums,
            posterior_means=per_pair_means,
            maps=per_pair_maps,
            pair_indices=per_pair_indices,
        )

    effective_args = {
        "pairs": per_pair_indices,
        "mode": mode,
        "fold_data": fold_data,
        "use_ancestral": use_ancestral,
        "skip_csfs_distance": skip_csfs_distance,
        "scaling_skip": scaling_skip,
        "posterior_sums": posterior_sums,
        "major_minor_posterior_sums": major_minor_posterior_sums,
        "store_per_pair_posterior": store_per_pair_posterior,
        "store_per_pair_posterior_mean": store_per_pair_posterior_mean,
        "store_per_pair_map": store_per_pair_map,
        "jobs": jobs,
        "job_index": job_index,
        "from_pos": from_pos,
        "to_pos": to_pos,
        "cm_burn_in": cm_burn_in,
        "output_prefix": str(output_prefix) if output_prefix is not None else None,
    }

    data.results["asmc"] = annotate_result(
        {
            "expected_times": result.expected_times,
            "discretization": result.discretization,
            "sum_of_posteriors": result.sum_of_posteriors,
            "sum_of_posteriors_major_minor": result.sum_of_posteriors_major_minor,
            "per_pair_posteriors": result.per_pair_posteriors,
            "per_pair_posterior_means": result.per_pair_posterior_means,
            "per_pair_maps": result.per_pair_maps,
            "per_pair_indices": result.per_pair_indices,
            "per_pair_labels": result.per_pair_labels,
            "min_posterior_means": result.min_posterior_means,
            "argmin_posterior_means": result.argmin_posterior_means,
            "min_maps": result.min_maps,
            "argmin_maps": result.argmin_maps,
            "n_pairs_decoded": result.n_pairs_decoded,
            "site_slice": (from_pos, to_pos),
            "decode_slice": (decode_from, decode_to),
            "jobs": jobs,
            "job_index": job_index,
        },
        method_name="asmc",
        implementation_requested=implementation,
        implementation_used=implementation_used,
        effective_args=effective_args,
        input_paths=data.uns.get("source_paths"),
        seed=random_seed,
        runtime_seconds=time.perf_counter() - started,
        artifacts=artifacts,
    )

    return data


def _asmc_binary_path() -> Path | None:
    status = upstream_status("asmc")
    cache_path = Path(status["cache_path"]) / "bin/ASMC_exe"
    if cache_path.exists():
        return cache_path
    return None


def _load_gz_matrix(path: Path) -> np.ndarray:
    import gzip

    with gzip.open(path, "rt") as fh:
        return np.loadtxt(fh, dtype=np.float64)


def _asmc_upstream_pairs(
    data: SmcData,
    *,
    pairs: list[tuple[int, int]],
    mode: str,
    fold_data: bool,
    skip_csfs_distance: float,
    posterior_sums: bool,
    major_minor_posterior_sums: bool,
    store_per_pair_posterior: bool,
    store_per_pair_posterior_mean: bool,
    store_per_pair_map: bool,
    from_pos: int,
    to_pos: int | None,
    cm_burn_in: float,
    output_prefix: str | Path | None,
    implementation_requested: str,
) -> SmcData:
    """Execute the maintained upstream Python binding for explicit pairs."""
    started = time.perf_counter()
    try:
        from asmc.asmc import ASMC as UpstreamAsmc
        from asmc.asmc import DecodingParams as UpstreamDecodingParams
    except ImportError as exc:
        raise RuntimeError(
            "Explicit-pair upstream ASMC requires the optional 'asmc' extra: "
            "`pip install smckit[asmc]`."
        ) from exc

    input_root = str(data.uns["input_file_root"])
    dq_path = str(data.uns["decoding_quantities_path"])
    map_path = str(data.uns.get("map_path", ""))
    prefix = str(output_prefix) if output_prefix is not None else input_root
    params = UpstreamDecodingParams(
        in_file_root=input_root,
        dq_file=dq_path,
        out_file_root=prefix,
        jobs=1,
        job_ind=1,
        decoding_mode_string=mode,
        decoding_sequence=mode == "sequence",
        using_CSFS=not np.isinf(skip_csfs_distance),
        compress=np.isinf(skip_csfs_distance),
        use_ancestral=not fold_data,
        # ``compress`` is the upstream shorthand for an infinite CSFS skip.
        # Supplying both controls is rejected by ASMC 1.4.0.
        skip_CSFS_distance=(
            float("nan") if np.isinf(skip_csfs_distance) else float(skip_csfs_distance)
        ),
        no_batches=False,
        do_posterior_sums=False,
        do_per_pair_posterior_mean=False,
        expected_coal_times_file="",
        within_only=False,
        do_major_minor_posterior_sums=False,
        do_per_pair_MAP=False,
        map_file=map_path,
    )
    # Upstream exposes its regression-test seed through this field.
    params.useKnownSeed = True
    engine = UpstreamAsmc(params)
    need_full = store_per_pair_posterior or major_minor_posterior_sums
    engine.set_store_per_pair_posterior(need_full)
    engine.set_store_sum_of_posterior(posterior_sums)
    engine.set_store_per_pair_posterior_mean(store_per_pair_posterior_mean)
    engine.set_store_per_pair_map(store_per_pair_map)
    first = [pair[0] for pair in pairs]
    second = [pair[1] for pair in pairs]
    n_sites = np.asarray(data.uns["haplotypes"]).shape[1]
    interval_to = n_sites if to_pos in {None, 0} else int(to_pos)
    engine.decode_pairs(first, second, int(from_pos), interval_to, float(cm_burn_in))
    raw = engine.get_copy_of_results()

    pair_posteriors = (
        [np.asarray(values, dtype=np.float64).T for values in raw.per_pair_posteriors]
        if need_full
        else []
    )
    if posterior_sums:
        sum_of_posteriors = np.asarray(raw.sum_of_posteriors, dtype=np.float64).T
    elif pair_posteriors:
        sum_of_posteriors = np.sum(pair_posteriors, axis=0, dtype=np.float64)
    else:
        sum_of_posteriors = np.zeros(
            (interval_to - from_pos, len(engine.get_expected_times())),
            dtype=np.float64,
        )

    major_minor: dict[str, np.ndarray] = {}
    if major_minor_posterior_sums:
        major_minor = {
            genotype: np.zeros_like(sum_of_posteriors) for genotype in ("00", "01", "11")
        }
        original_haplotypes = np.asarray(data.uns["haplotypes"], dtype=np.uint8).copy()
        if bool(data.uns.get("fold_to_minor", True)):
            flipped = np.asarray(data.uns.get("site_was_flipped"), dtype=bool)
            original_haplotypes[:, flipped] = 1 - original_haplotypes[:, flipped]
        for posterior, (hap_a, hap_b) in zip(pair_posteriors, pairs, strict=True):
            first_allele = original_haplotypes[hap_a, from_pos:interval_to]
            second_allele = original_haplotypes[hap_b, from_pos:interval_to]
            masks = {
                "00": (first_allele == 0) & (second_allele == 0),
                "01": first_allele != second_allele,
                "11": (first_allele == 1) & (second_allele == 1),
            }
            for genotype, mask in masks.items():
                major_minor[genotype] += posterior * mask[:, None]

    means_matrix = (
        np.asarray(raw.per_pair_posterior_means, dtype=np.float64)
        if store_per_pair_posterior_mean
        else np.empty((0, interval_to - from_pos), dtype=np.float64)
    )
    maps_matrix = (
        np.asarray(raw.per_pair_MAPs, dtype=np.int32)
        if store_per_pair_map
        else np.empty((0, interval_to - from_pos), dtype=np.int32)
    )
    artifacts: list[dict[str, Any]] = []
    if output_prefix is not None:
        artifacts = _write_native_asmc_outputs(
            output_prefix,
            posterior_sums=sum_of_posteriors if posterior_sums else None,
            major_minor_sums=major_minor,
            posterior_means=list(means_matrix),
            maps=list(maps_matrix),
            pair_indices=pairs,
        )
    upstream_metadata = standard_upstream_metadata(
        "asmc",
        effective_args={
            "pairs": pairs,
            "mode": mode,
            "fold_data": fold_data,
            "skip_csfs_distance": skip_csfs_distance,
            "from_pos": from_pos,
            "to_pos": interval_to,
            "cm_burn_in": cm_burn_in,
        },
        extra={
            "interface": "asmc-asmc Python binding",
            "binding_version": "1.4.0",
            "seed": 1234,
        },
    )
    result = {
        "backend": "upstream",
        "expected_times": np.asarray(engine.get_expected_times(), dtype=np.float64),
        "discretization": np.asarray(
            data.uns["decoding_quantities"].discretization,
            dtype=np.float64,
        ),
        "sum_of_posteriors": sum_of_posteriors,
        "sum_of_posteriors_major_minor": major_minor,
        "per_pair_posteriors": pair_posteriors if store_per_pair_posterior else [],
        "per_pair_posterior_means": list(means_matrix),
        "per_pair_maps": list(maps_matrix),
        "per_pair_indices": pairs,
        "per_pair_labels": [
            (_pair_label(data, first_hap), _pair_label(data, second_hap))
            for first_hap, second_hap in pairs
        ],
        "min_posterior_means": (
            np.asarray(raw.min_posterior_means) if store_per_pair_posterior_mean else None
        ),
        "argmin_posterior_means": (
            np.asarray(raw.argmin_posterior_means) if store_per_pair_posterior_mean else None
        ),
        "min_maps": np.asarray(raw.min_MAPs) if store_per_pair_map else None,
        "argmin_maps": np.asarray(raw.argmin_MAPs) if store_per_pair_map else None,
        "n_pairs_decoded": len(pairs),
        "site_slice": (from_pos, interval_to),
        "upstream": upstream_metadata,
    }
    data.results["asmc"] = annotate_result(
        result,
        method_name="asmc",
        implementation_requested=implementation_requested,
        implementation_used="upstream",
        upstream_metadata=upstream_metadata,
        effective_args=upstream_metadata["effective_args"],
        input_paths=data.uns.get("source_paths"),
        seed=1234,
        runtime_seconds=time.perf_counter() - started,
        artifacts=artifacts,
    )
    return data


def _asmc_upstream(
    data: SmcData,
    *,
    pairs: list[tuple[int, int]] | None,
    mode: str,
    fold_data: bool,
    skip_csfs_distance: float,
    posterior_sums: bool,
    major_minor_posterior_sums: bool,
    store_per_pair_posterior: bool,
    store_per_pair_posterior_mean: bool,
    store_per_pair_map: bool,
    jobs: int,
    job_index: int,
    from_pos: int,
    to_pos: int | None,
    cm_burn_in: float,
    output_prefix: str | Path | None,
    implementation_requested: str,
    upstream_options: dict | None,
) -> SmcData:
    started = time.perf_counter()
    if pairs is not None:
        return _asmc_upstream_pairs(
            data,
            pairs=pairs,
            mode=mode,
            fold_data=fold_data,
            skip_csfs_distance=skip_csfs_distance,
            posterior_sums=posterior_sums,
            major_minor_posterior_sums=major_minor_posterior_sums,
            store_per_pair_posterior=store_per_pair_posterior,
            store_per_pair_posterior_mean=store_per_pair_posterior_mean,
            store_per_pair_map=store_per_pair_map,
            from_pos=from_pos,
            to_pos=to_pos,
            cm_burn_in=cm_burn_in,
            output_prefix=output_prefix,
            implementation_requested=implementation_requested,
        )
    if not posterior_sums and not major_minor_posterior_sums:
        raise ValueError("Upstream ASMC requires posterior_sums or major_minor_posterior_sums.")
    if jobs < 1 or job_index < 1 or job_index > jobs:
        raise ValueError("Require jobs >= 1 and 1 <= job_index <= jobs.")
    input_root = data.uns.get("input_file_root")
    dq_path = data.uns.get("decoding_quantities_path")
    if not input_root or not dq_path:
        raise ValueError("Upstream ASMC requires input file root and decoding quantities path.")
    status = upstream_status("asmc")
    if not status["cache_ready"]:
        bootstrap_upstream("asmc")
    binary = _asmc_binary_path()
    if binary is None:
        raise RuntimeError("Upstream ASMC executable is unavailable after bootstrap.")
    effective_args = {
        "inFileRoot": input_root,
        "decodingQuantFile": dq_path,
        "mode": mode,
        "skipCSFSdistance": skip_csfs_distance,
        "useAncestral": not fold_data,
        "posteriorSums": posterior_sums,
        "majorMinorPosteriorSums": major_minor_posterior_sums,
        "jobs": jobs,
        "jobInd": job_index,
        "outFileRoot": str(output_prefix) if output_prefix is not None else None,
    }
    if upstream_options:
        effective_args.update(upstream_options)
    with tempfile.TemporaryDirectory(prefix="smckit-asmc-") as tmpdir:
        out_root_path = Path(output_prefix) if output_prefix is not None else Path(tmpdir) / "asmc"
        out_root_path.parent.mkdir(parents=True, exist_ok=True)
        out_root = str(out_root_path)
        cmd = [
            str(binary),
            "--decodingQuantFile",
            str(dq_path),
            "--inFileRoot",
            str(input_root),
            "--outFileRoot",
            out_root,
            "--jobs",
            str(jobs),
            "--jobInd",
            str(job_index),
        ]
        if mode == "sequence":
            cmd.extend(["--mode", "sequence"])
        if posterior_sums:
            cmd.append("--posteriorSums")
        if major_minor_posterior_sums:
            cmd.append("--majorMinorPosteriorSums")
        if not fold_data:
            cmd.append("--useAncestral")
        if np.isinf(skip_csfs_distance):
            cmd.append("--compress")
        elif skip_csfs_distance > 0:
            cmd.extend(["--skipCSFSdistance", repr(float(skip_csfs_distance) * 100.0)])
        if upstream_options:
            for key, value in upstream_options.items():
                option = f"--{key}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(option)
                elif value is not None:
                    cmd.extend([option, str(value)])
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Upstream ASMC backend failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        sum_path = Path(f"{out_root}.sumOverPairs.gz")
        major_paths = {
            genotype: Path(f"{out_root}.{genotype}.sumOverPairs.gz")
            for genotype in ("00", "01", "11")
        }
        artifacts: list[dict[str, Any]] = []
        if output_prefix is not None:
            candidates = ([sum_path] if posterior_sums else []) + (
                list(major_paths.values()) if major_minor_posterior_sums else []
            )
            artifacts = [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "kind": f"asmc-{path.name.removeprefix(out_root_path.name + '.')}",
                }
                for path in candidates
                if path.is_file()
            ]
        result = {
            "backend": "upstream",
            "n_pairs_decoded": len(
                _select_job_pairs(
                    _all_haplotype_pairs(np.asarray(data.uns["haplotypes"]).shape[0]),
                    jobs,
                    job_index,
                )
            ),
            "jobs": jobs,
            "job_index": job_index,
            "upstream": standard_upstream_metadata(
                "asmc",
                effective_args=effective_args,
                extra={
                    "binary": str(binary),
                    "out_file_root": out_root,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "command": cmd,
                },
            ),
        }
        if posterior_sums:
            result["sum_of_posteriors"] = np.asarray(
                _load_gz_matrix(sum_path),
                dtype=np.float64,
            )
        if major_minor_posterior_sums:
            result["sum_of_posteriors_major_minor"] = {
                genotype: np.asarray(_load_gz_matrix(path), dtype=np.float64)
                for genotype, path in major_paths.items()
            }
            if not posterior_sums:
                result["sum_of_posteriors"] = sum(result["sum_of_posteriors_major_minor"].values())
        upstream_metadata = result["upstream"]
        data.results["asmc"] = annotate_result(
            result,
            method_name="asmc",
            implementation_requested=implementation_requested,
            implementation_used="upstream",
            upstream_metadata=upstream_metadata,
            effective_args=effective_args,
            input_paths=data.uns.get("source_paths"),
            runtime_seconds=time.perf_counter() - started,
            artifacts=artifacts,
        )
        return data
