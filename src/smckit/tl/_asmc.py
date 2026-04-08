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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smckit._core import SmcData
from smckit.io._asmc import DecodingQuantities
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    require_upstream_available,
    standard_upstream_metadata,
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

    xor_bits = np.logical_xor(hap_i, hap_j)   # heterozygous
    and_bits = np.logical_and(hap_i, hap_j)    # both derived

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
                csfs = dq.folded_ascertained_csfs_map if not decoding_sequence else dq.folded_csfs_map
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
                    e2m0[pos] = csfs[uc2][0] - csfs[uc0][0] if (uc0 >= 0 and csfs[uc0] is not None) else csfs[uc2][0]
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
        return dq.B_vectors[rec_dist], dq.U_vectors[rec_dist], dq.D_vectors[rec_dist], dq.row_ratio_vectors[rec_dist]

    # Nearest key fallback
    keys = np.array(list(dq.D_vectors.keys()))
    idx = np.argmin(np.abs(keys - rec_dist))
    nearest = keys[idx]
    return dq.B_vectors[nearest], dq.U_vectors[nearest], dq.D_vectors[nearest], dq.row_ratio_vectors[nearest]


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

    Returns
    -------
    alpha : (L, states) float32
        Scaled forward probabilities.
    """
    if to_pos is None:
        to_pos = len(genetic_positions)

    states = dq.states
    # Use float32 to match C++ SIMD precision exactly
    alpha = np.zeros((to_pos, states), dtype=np.float32)

    init_prob = dq.initial_state_prob  # float32

    # Initialize at from_pos
    em_vec = emission1[from_pos] + emission0minus1[from_pos] * obs_is_zero[from_pos] + emission2minus0[from_pos] * obs_is_two[from_pos]
    alpha[from_pos] = init_prob * em_vec

    # Scale initial
    s = alpha[from_pos].sum()
    if s != 0:
        alpha[from_pos] /= s

    last_gen_pos = genetic_positions[from_pos]

    for pos in range(from_pos + 1, to_pos):
        rec_dist = round_morgans(float(genetic_positions[pos] - last_gen_pos))

        B, U, D, _ = _get_transition_vectors(dq, rec_dist)
        col_ratios = dq.column_ratios

        prev = alpha[pos - 1]

        # alphaC: cumulative sum from top (state states-1 down to 0)
        alphaC = np.empty(states, dtype=np.float32)
        alphaC[states - 1] = prev[states - 1]
        for k in range(states - 2, -1, -1):
            alphaC[k] = alphaC[k + 1] + prev[k]

        # AU: upward accumulator (float32 to match C++)
        AU = np.float32(0.0)
        for k in range(states):
            if k > 0:
                AU = U[k - 1] * prev[k - 1] + col_ratios[k - 1] * AU

            # Transition: AU + D[k]*prev[k] + B[k]*alphaC[k+1]
            trans = AU + D[k] * prev[k]
            if k < states - 1:
                trans += B[k] * alphaC[k + 1]

            em = (emission1[pos, k]
                  + emission0minus1[pos, k] * obs_is_zero[pos]
                  + emission2minus0[pos, k] * obs_is_two[pos])
            alpha[pos, k] = em * trans

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

        B, U, D, RR = _get_transition_vectors(dq, rec_dist)

        # vec[k] = emission(pos+1, k) * beta(pos+1, k)
        em_next = (emission1[pos + 1]
                   + emission0minus1[pos + 1] * obs_is_zero[pos + 1]
                   + emission2minus0[pos + 1] * obs_is_two[pos + 1])
        vec = em_next * beta[pos + 1]

        # BU: upward sweep from bottom
        BU = np.zeros(states, dtype=np.float32)
        for k in range(states - 2, -1, -1):
            BU[k] = U[k] * vec[k + 1] + RR[k] * BU[k + 1]

        # BL: downward cumulative + D diagonal
        BL = np.float32(0.0)
        for k in range(states):
            if k > 0:
                BL += B[k - 1] * vec[k - 1]
            beta[pos, k] = BL + D[k] * vec[k] + BU[k]

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

    expected_times: np.ndarray          # (states,) time per state
    discretization: np.ndarray          # (states+1,) time boundaries
    sum_of_posteriors: np.ndarray       # (n_sites, states) aggregated posteriors
    per_pair_posterior_means: list[np.ndarray] = field(default_factory=list)
    per_pair_maps: list[np.ndarray] = field(default_factory=list)
    per_pair_indices: list[tuple[int, int]] = field(default_factory=list)
    n_pairs_decoded: int = 0


def asmc(
    data: SmcData,
    *,
    pairs: list[tuple[int, int]] | None = None,
    mode: str = "array",
    fold_data: bool = True,
    skip_csfs_distance: float = 0.0,
    scaling_skip: int = 1,
    store_per_pair_posterior_mean: bool = True,
    store_per_pair_map: bool = False,
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
    skip_csfs_distance : float
        Minimum distance between CSFS sites (Morgans).
    scaling_skip : int
        Apply scaling every this many positions.
    store_per_pair_posterior_mean : bool
        Store per-pair posterior mean TMRCA.
    store_per_pair_map : bool
        Store per-pair MAP state indices, matching upstream ASMC ``perPairMAP``.
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo decoder.
        ``"upstream"`` currently raises because the public upstream bridge is
        not exposed yet. ``"auto"`` resolves to the best available implementation.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["asmc"]``.
    """
    implementation = normalize_implementation(implementation)
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("asmc"),
    )
    if implementation_used == "upstream":
        return _asmc_upstream(
            data,
            mode=mode,
            skip_csfs_distance=skip_csfs_distance,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )
    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported asmc native_options keys: {unsupported}")

    haplotypes = data.uns["haplotypes"]
    genetic_positions = data.uns["genetic_positions"]
    dq: DecodingQuantities = data.uns["decoding_quantities"]
    n_haps, n_sites = haplotypes.shape
    decoding_sequence = mode == "sequence"

    logger.info("ASMC: %d haplotypes, %d sites, %d states", n_haps, n_sites, dq.states)

    # Prepare emissions (no CSFS for now unless undistinguished counts provided)
    undist_counts = data.uns.get("undistinguished_counts", None)
    use_csfs = undist_counts is not None

    emission1, emission0minus1, emission2minus0 = prepare_emissions(
        dq, genetic_positions, n_sites,
        use_csfs=use_csfs,
        skip_csfs_distance=skip_csfs_distance,
        fold_data=fold_data,
        decoding_sequence=decoding_sequence,
        undistinguished_counts=undist_counts,
    )

    # Generate pairs
    if pairs is None:
        pairs = []
        for i in range(n_haps):
            for j in range(i + 1, n_haps):
                pairs.append((i, j))

    sum_of_posteriors = np.zeros((n_sites, dq.states), dtype=np.float32)
    per_pair_means = []
    per_pair_maps = []
    per_pair_indices = []

    for pair_idx, (i, j) in enumerate(pairs):
        if (pair_idx + 1) % 100 == 0 or pair_idx == 0:
            logger.info("  Decoding pair %d/%d (haps %d, %d)", pair_idx + 1, len(pairs), i, j)

        obs = encode_pair(haplotypes, i, j)

        alpha = forward(
            dq, emission1, emission0minus1, emission2minus0,
            obs.obs_is_zero, obs.obs_is_two,
            genetic_positions,
            scaling_skip=scaling_skip,
        )

        beta = backward(
            dq, emission1, emission0minus1, emission2minus0,
            obs.obs_is_zero, obs.obs_is_two,
            genetic_positions,
            scaling_skip=scaling_skip,
        )

        posteriors = compute_posteriors(alpha, beta)
        sum_of_posteriors += posteriors
        per_pair_indices.append((i, j))

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
        per_pair_posterior_means=per_pair_means,
        per_pair_maps=per_pair_maps,
        per_pair_indices=per_pair_indices,
        n_pairs_decoded=len(pairs),
    )

    data.results["asmc"] = annotate_result({
        "expected_times": result.expected_times,
        "discretization": result.discretization,
        "sum_of_posteriors": result.sum_of_posteriors,
        "per_pair_posterior_means": result.per_pair_posterior_means,
        "per_pair_maps": result.per_pair_maps,
        "per_pair_indices": result.per_pair_indices,
        "n_pairs_decoded": result.n_pairs_decoded,
    }, implementation_requested=implementation, implementation_used=implementation_used)

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


def _asmc_upstream(
    data: SmcData,
    *,
    mode: str,
    skip_csfs_distance: float,
    implementation_requested: str,
    upstream_options: dict | None,
) -> SmcData:
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
    }
    if upstream_options:
        effective_args.update(upstream_options)
    with tempfile.TemporaryDirectory(prefix="smckit-asmc-") as tmpdir:
        out_root = str(Path(tmpdir) / "asmc")
        cmd = [
            str(binary),
            "--decodingQuantFile",
            str(dq_path),
            "--inFileRoot",
            str(input_root),
            "--outFileRoot",
            out_root,
        ]
        if mode == "sequence":
            cmd.extend(["--mode", "sequence", "--posteriorSums"])
            output_paths = [Path(out_root + ".1-1.sumOverPairs.gz")]
        else:
            cmd.append("--majorMinorPosteriorSums")
            output_paths = [
                Path(out_root + ".00.sumOverPairs.gz"),
                Path(out_root + ".01.sumOverPairs.gz"),
                Path(out_root + ".11.sumOverPairs.gz"),
            ]
        if np.isinf(skip_csfs_distance):
            cmd.append("--compress")
        elif skip_csfs_distance > 0:
            cmd.extend(["--skipCSFSdistance", repr(float(skip_csfs_distance) * 100.0)])
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream ASMC backend failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        matrices = [_load_gz_matrix(path) for path in output_paths]
        result = {
            "backend": "upstream",
            "n_pairs_decoded": None,
            "upstream": standard_upstream_metadata(
                "asmc",
                effective_args=effective_args,
                extra={
                    "binary": str(binary),
                    "out_file_root": out_root,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "output_paths": [str(p) for p in output_paths],
                },
            ),
        }
        if mode == "sequence":
            result["sum_of_posteriors"] = np.asarray(matrices[0], dtype=np.float64)
        else:
            result["sum_of_posteriors_major_minor"] = {
                "00": np.asarray(matrices[0], dtype=np.float64),
                "01": np.asarray(matrices[1], dtype=np.float64),
                "11": np.asarray(matrices[2], dtype=np.float64),
            }
            result["sum_of_posteriors"] = (
                result["sum_of_posteriors_major_minor"]["00"]
                + result["sum_of_posteriors_major_minor"]["01"]
                + result["sum_of_posteriors_major_minor"]["11"]
            )
        data.results["asmc"] = annotate_result(
            result,
            implementation_requested=implementation_requested,
            implementation_used="upstream",
        )
        return data
