"""MSMC2: Multiple Sequentially Markovian Coalescent (pairwise mode).

Reimplementation of Schiffels & Durbin (2014) / MSMC2 (Schiffels & Wang 2020).
Port of the D-language MSMC2 codebase to Python with Numba JIT compilation.

The core HMM is structurally identical to PSMC (hidden states = coalescent
time intervals, observations = hom/het/missing), but differs in:

1. Time intervals: Li & Durbin boundaries with ``factor`` parameter
2. Transition matrix: Numerical integration of coalescent process
3. Emission model: Based on mutation rate and expected coalescence time
4. Multiple pairs: E-step sums expected counts over all haplotype pairs
5. Precomputed propagators: Matrix powers T^d for distances d=1..maxDistance
"""

from __future__ import annotations

import logging
import math as pymath
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import math
import numba
import numpy as np

from smckit._core import SmcData
from smckit.io._multihetsep import read_msmc_output
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

HMM_TINY = 1e-25


# ---------------------------------------------------------------------------
# Pattern parsing (reused from PSMC, same format)
# ---------------------------------------------------------------------------

def parse_time_segment_pattern(pattern: str) -> list[int]:
    """Parse a time segment pattern string like ``"1*2+25*1+1*2+1*3"``.

    Parameters
    ----------
    pattern : str
        Pattern string. ``+`` separates groups, ``N*M`` means N repeats of
        segments of size M.

    Returns
    -------
    segment_sizes : list of int
        List of segment sizes, e.g. [2, 1, 1, ..., 1, 2, 3].
    """
    segments: list[int] = []
    tokens = pattern.replace("+", " ").split()
    for tok in tokens:
        if "*" in tok:
            parts = tok.split("*")
            repeats = int(parts[0])
            value = int(parts[1])
            segments.extend([value] * repeats)
        else:
            segments.append(int(tok))
    return segments


def _build_par_map(segment_sizes: list[int]) -> tuple[np.ndarray, int, int]:
    """Build a parameter map from segment sizes.

    Parameters
    ----------
    segment_sizes : list of int
        Output of ``parse_time_segment_pattern``.

    Returns
    -------
    par_map : (n_intervals,) int array
        Maps each time interval index to its free parameter group index.
    n_free : int
        Number of free lambda parameters (= len(segment_sizes)).
    n_intervals : int
        Total number of time intervals (= sum(segment_sizes)).
    """
    n_free = len(segment_sizes)
    n_intervals = sum(segment_sizes)
    par_map = np.empty(n_intervals, dtype=np.int32)
    idx = 0
    for group_id, group_size in enumerate(segment_sizes):
        for _ in range(group_size):
            par_map[idx] = group_id
            idx += 1
    return par_map, n_free, n_intervals


# ---------------------------------------------------------------------------
# 1. Time interval boundaries
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def li_durbin_boundaries(n_intervals, factor=0.1):
    """Li & Durbin time boundaries.

    Parameters
    ----------
    n_intervals : int
        Number of time intervals.
    factor : float
        Controls spacing near t=0.  MSMC2 default = 0.1 / n_pairs.

    Returns
    -------
    boundaries : (n_intervals + 1,) array
        boundaries[0] = 0, boundaries[n_intervals] = inf.
    """
    t_max = 15.0
    boundaries = np.empty(n_intervals + 1, dtype=np.float64)
    for i in range(n_intervals + 1):
        if i == n_intervals:
            boundaries[i] = math.inf
        else:
            boundaries[i] = factor * (
                math.exp(i / n_intervals * math.log(1.0 + t_max / factor)) - 1.0
            )
    return boundaries


@numba.njit(cache=True)
def quantile_boundaries(n_intervals, factor=1.0):
    """Quantile time boundaries (as used in original MSMC).

    Parameters
    ----------
    n_intervals : int
        Number of time intervals.
    factor : float
        Scale factor.

    Returns
    -------
    boundaries : (n_intervals + 1,) array
        boundaries[0] = 0, boundaries[n_intervals] = inf.
    """
    boundaries = np.empty(n_intervals + 1, dtype=np.float64)
    for i in range(n_intervals + 1):
        if i == n_intervals:
            boundaries[i] = math.inf
        else:
            boundaries[i] = -factor * math.log(1.0 - i / n_intervals)
    return boundaries


# ---------------------------------------------------------------------------
# 2. Mean time in interval (with lambda)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def mean_time_in_interval(left, right, lam):
    """Expected coalescence time in interval [left, right) given rate lam.

    Parameters
    ----------
    left : float
        Left boundary of interval.
    right : float
        Right boundary of interval (may be inf for last interval).
    lam : float
        Coalescence rate (inverse effective population size, scaled).

    Returns
    -------
    float
        Expected time of coalescence conditional on falling in this interval.
    """
    if lam < 0.001:
        # Low rate approximation
        if math.isinf(right):
            return left + 1.0 / lam
        else:
            return (left + right) / 2.0
    else:
        # Normal rate
        delta = right - left
        if math.isinf(delta):
            # Last interval: t = left + 1/lam
            return left + 1.0 / lam
        t = 1.0 + lam * left
        t -= math.exp(-lam * delta) * (1.0 + lam * right)
        t /= (1.0 - math.exp(-delta * lam)) * lam
        return t


# ---------------------------------------------------------------------------
# 3. Integrate lambda (survival probability across intervals)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _integrate_lambda(from_t, to_t, from_idx, to_idx, lambda_vec, boundaries,
                      lambda_fac=1.0):
    """Compute exp(-integral of lambda_fac * lambda(t) dt from from_t to to_t).

    This is the survival probability of the coalescent process across
    a range of time intervals, optionally with a multiplicative factor
    on the rates.

    Parameters
    ----------
    from_t : float
        Start time.
    to_t : float
        End time.
    from_idx : int
        Index of interval containing from_t.
    to_idx : int
        Index of interval containing to_t.
    lambda_vec : (n,) array
        Coalescence rates per interval.
    boundaries : (n+1,) array
        Time interval boundaries.
    lambda_fac : float
        Multiplicative factor on rates (default 1.0, use 2.0 for diploid).

    Returns
    -------
    float
        Survival probability.
    """
    if from_idx == to_idx:
        return math.exp(-(to_t - from_t) * lambda_fac * lambda_vec[to_idx])

    # Sum over intermediate intervals
    accum = 0.0
    for kappa in range(from_idx + 1, to_idx):
        delta_k = boundaries[kappa + 1] - boundaries[kappa]
        accum += lambda_fac * lambda_vec[kappa] * delta_k

    # Contribution from from_idx: from from_t to right boundary of from_idx
    right_from = boundaries[from_idx + 1]
    # Contribution from to_idx: from left boundary of to_idx to to_t
    left_to = boundaries[to_idx]

    result = math.exp(
        -(right_from - from_t) * lambda_fac * lambda_vec[from_idx]
        - accum
        - (to_t - left_to) * lambda_fac * lambda_vec[to_idx]
    )
    return result


# ---------------------------------------------------------------------------
# 4. Transition matrix
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def compute_transition_matrix(boundaries, rho, lambda_vec):
    """Compute n x n transition probability matrix.

    Port of ``TransitionRate`` from ``vendor/msmc2/model/transition_rate.d``.

    Parameters
    ----------
    boundaries : (n + 1,) array
        Time interval boundaries. boundaries[0] = 0, boundaries[n] = inf.
    rho : float
        Recombination rate (must be positive).
    lambda_vec : (n,) array
        Coalescence rates per interval.

    Returns
    -------
    trans : (n, n) array
        Transition probability matrix. trans[a, b] = P(next state = b | cur = a).
        Columns sum to 1.
    """
    n = lambda_vec.shape[0]
    trans = np.zeros((n, n), dtype=np.float64)

    for b in range(n):
        col_sum = 0.0
        for a in range(n):
            if a != b:
                if a < b:
                    val = _q2_integral_smaller(a, b, boundaries, rho, lambda_vec)
                else:
                    val = _q2_integral_greater(a, b, boundaries, rho, lambda_vec)
                trans[a, b] = val
                col_sum += val
        trans[b, b] = 1.0 - col_sum

    return trans


@numba.njit(cache=True)
def _q2_integral_smaller(a, b, boundaries, rho, lambda_vec):
    """Off-diagonal transition probability for a < b.

    Parameters
    ----------
    a : int
        Source state (smaller).
    b : int
        Target state (larger).
    boundaries : (n+1,) array
    rho : float
    lambda_vec : (n,) array

    Returns
    -------
    float
        Transition probability from column b to row a.
    """
    mean_t = mean_time_in_interval(
        boundaries[b], boundaries[b + 1], lambda_vec[b]
    )
    delta_a = boundaries[a + 1] - boundaries[a]

    integ = (1.0 - math.exp(-delta_a * 2.0 * lambda_vec[a])) / (
        2.0 * lambda_vec[a]
    )

    total = 0.0
    for g in range(a):
        delta_g = boundaries[g + 1] - boundaries[g]
        term = (
            2.0
            * (1.0 - math.exp(-delta_g * 2.0 * lambda_vec[g]))
            * _integrate_lambda(
                boundaries[g + 1], boundaries[a], g + 1, a,
                lambda_vec, boundaries, 2.0
            )
            / (2.0 * lambda_vec[g])
            * integ
        )
        total += term

    total += 2.0 * (delta_a - integ) / (2.0 * lambda_vec[a])

    result = (
        (1.0 - math.exp(-rho * 2.0 * mean_t))
        / (mean_t * 2.0)
        * lambda_vec[a]
        * total
    )
    return result


@numba.njit(cache=True)
def _q2_integral_greater(a, b, boundaries, rho, lambda_vec):
    """Off-diagonal transition probability for a > b.

    Parameters
    ----------
    a : int
        Source state (larger).
    b : int
        Target state (smaller).
    boundaries : (n+1,) array
    rho : float
    lambda_vec : (n,) array

    Returns
    -------
    float
        Transition probability from column b to row a.
    """
    mean_t = mean_time_in_interval(
        boundaries[b], boundaries[b + 1], lambda_vec[b]
    )
    delta_a = boundaries[a + 1] - boundaries[a]

    integ = (
        _integrate_lambda(
            mean_t, boundaries[a], b, a, lambda_vec, boundaries, 1.0
        )
        / lambda_vec[a]
        * (1.0 - math.exp(-delta_a * lambda_vec[a]))
    )

    total = 0.0
    for g in range(b):
        delta_g = boundaries[g + 1] - boundaries[g]
        term = (
            2.0
            * (1.0 - math.exp(-2.0 * lambda_vec[g] * delta_g))
            / (2.0 * lambda_vec[g])
            * _integrate_lambda(
                boundaries[g + 1], mean_t, g + 1, b,
                lambda_vec, boundaries, 2.0
            )
        )
        total += term

    total += (
        2.0
        * (1.0 - math.exp(-2.0 * lambda_vec[b] * (mean_t - boundaries[b])))
        / (2.0 * lambda_vec[b])
    )

    result = (
        integ
        * (1.0 - math.exp(-rho * 2.0 * mean_t))
        / (mean_t * 2.0)
        * lambda_vec[a]
        * total
    )
    return result


# ---------------------------------------------------------------------------
# 5. Equilibrium probability
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def equilibrium_prob(boundaries, lambda_vec):
    """Compute equilibrium (stationary) distribution over states.

    Parameters
    ----------
    boundaries : (n + 1,) array
        Time interval boundaries.
    lambda_vec : (n,) array
        Coalescence rates per interval.

    Returns
    -------
    eq : (n,) array
        Equilibrium probability for each state.  Sums to 1.
    """
    n = lambda_vec.shape[0]
    eq = np.empty(n, dtype=np.float64)
    for a in range(n):
        delta_a = boundaries[a + 1] - boundaries[a]
        eq[a] = (
            _integrate_lambda(0.0, boundaries[a], 0, a, lambda_vec, boundaries)
            * (1.0 - math.exp(-delta_a * lambda_vec[a]))
        )
    return eq


# ---------------------------------------------------------------------------
# 6. Emission probabilities
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def emission_probs(boundaries, lambda_vec, mu):
    """Compute emission probability matrix.

    Parameters
    ----------
    boundaries : (n + 1,) array
        Time interval boundaries.
    lambda_vec : (n,) array
        Coalescence rates per interval.
    mu : float
        Mutation rate (scaled, same units as boundaries).

    Returns
    -------
    e : (3, n) array
        e[0, a] = 1.0 (missing data)
        e[1, a] = P(homozygous | state a)
        e[2, a] = P(heterozygous | state a) = 1 - P(hom | state a)
    """
    n = lambda_vec.shape[0]
    e = np.empty((3, n), dtype=np.float64)

    for a in range(n):
        e[0, a] = 1.0  # missing

        lb = lambda_vec[a]
        tb = boundaries[a]
        delta = boundaries[a + 1] - boundaries[a]

        if a == n - 1:
            # Last interval (extends to infinity)
            hom_prob = math.exp(-2.0 * mu * tb) * lb / (2.0 * mu + lb)
        else:
            second_term = math.exp(-2.0 * mu * tb) * (
                1.0 - math.exp(-(2.0 * mu + lb) * delta)
            )
            if lb < 0.001:
                hom_prob = 1.0 / (delta * 2.0 * mu) * second_term
            else:
                hom_prob = (
                    1.0
                    / (1.0 - math.exp(-delta * lb))
                    * lb
                    / (lb + 2.0 * mu)
                    * second_term
                )

        if hom_prob < 0.0:
            hom_prob = 0.0
        elif hom_prob > 1.0:
            hom_prob = 1.0

        e[1, a] = hom_prob
        e[2, a] = 1.0 - hom_prob

    return e


# ---------------------------------------------------------------------------
# 7. Precomputed propagators
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def precompute_forward_propagators(trans, emission_hom, max_distance):
    """Precompute forward propagator matrices P[d] = (T * diag(e_hom))^(d+1).

    The base propagator P[0] = T * diag(e_hom) encodes one step of
    transitioning and emitting a homozygous observation.  P[d] = P[0]^(d+1).

    Parameters
    ----------
    trans : (n, n) array
        Transition matrix.
    emission_hom : (n,) array
        Homozygous emission probabilities per state.
    max_distance : int
        Maximum propagation distance.

    Returns
    -------
    props : (max_distance, n, n) array
        props[d] = P[0]^(d+1).
    """
    n = trans.shape[0]
    props = np.empty((max_distance, n, n), dtype=np.float64)

    # P[0] = trans * diag(emission_hom)
    # P[0][a, b] = trans[a, b] * emission_hom[a]
    # Note: in the D code, the propagator is set as:
    #   ret[0][a][b] = transitionMatrix[a][b] * e
    # where e = emissionProbs[1][a] (emission of the source state a)
    for a in range(n):
        e_a = emission_hom[a]
        for b in range(n):
            props[0, a, b] = trans[a, b] * e_a

    # P[d] = P[0] @ P[d-1]
    for d in range(1, max_distance):
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += props[0, i, k] * props[d - 1, k, j]
                props[d, i, j] = s

    return props


@numba.njit(cache=True)
def precompute_backward_propagators(trans, emission_hom, max_distance):
    """Precompute backward propagator matrices.

    The backward propagator differs from forward: P_bwd[d] = P_bwd[d-1] @ P_bwd[0].

    Parameters
    ----------
    trans : (n, n) array
        Transition matrix.
    emission_hom : (n,) array
        Homozygous emission probabilities per state.
    max_distance : int
        Maximum propagation distance.

    Returns
    -------
    props : (max_distance, n, n) array
    """
    n = trans.shape[0]
    props = np.empty((max_distance, n, n), dtype=np.float64)

    # P[0][a, b] = trans[a, b] * emission_hom[a]
    for a in range(n):
        e_a = emission_hom[a]
        for b in range(n):
            props[0, a, b] = trans[a, b] * e_a

    # P[d] = P[d-1] @ P[0]
    for d in range(1, max_distance):
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += props[d - 1, i, k] * props[0, k, j]
                props[d, i, j] = s

    return props


@numba.njit(cache=True)
def precompute_all_propagators(trans, emission_hom, max_distance):
    """Precompute forward and backward propagators for both hom and missing.

    Parameters
    ----------
    trans : (n, n) array
        Transition matrix.
    emission_hom : (n,) array
        Homozygous emission probabilities per state.
    max_distance : int
        Maximum propagation distance.

    Returns
    -------
    fwd_props : (max_distance, n, n) array
        Forward propagators with homozygous emission.
    bwd_props : (max_distance, n, n) array
        Backward propagators with homozygous emission.
    fwd_props_miss : (max_distance, n, n) array
        Forward propagators with missing emission (=1).
    bwd_props_miss : (max_distance, n, n) array
        Backward propagators with missing emission (=1).
    """
    n = trans.shape[0]
    emission_miss = np.ones(n, dtype=np.float64)

    fwd_props = precompute_forward_propagators(trans, emission_hom, max_distance)
    bwd_props = precompute_backward_propagators(trans, emission_hom, max_distance)
    fwd_props_miss = precompute_forward_propagators(trans, emission_miss, max_distance)
    bwd_props_miss = precompute_backward_propagators(trans, emission_miss, max_distance)

    return fwd_props, bwd_props, fwd_props_miss, bwd_props_miss


# ---------------------------------------------------------------------------
# 8. Segment chopping
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def chop_segments(seg_pos, seg_obs, max_distance):
    """Chop segments so no gap between consecutive positions exceeds maxDistance.

    Mirrors the ``chop_segsites`` function in the D codebase.

    Parameters
    ----------
    seg_pos : (L,) int64 array
        Segment positions.
    seg_obs : (L,) float64 array
        Segment observations (0=missing, 1=hom, 2=het).
    max_distance : int
        Maximum distance between consecutive segment positions.

    Returns
    -------
    new_pos : (L',) int64 array
    new_obs : (L',) float64 array
    """
    L = seg_pos.shape[0]

    # First pass: count output size
    count = 0
    last_pos = 0
    for i in range(L):
        pos = seg_pos[i]
        gap = pos - last_pos
        while gap > max_distance:
            count += 1
            last_pos += max_distance
            gap = pos - last_pos
        count += 1
        last_pos = pos

    new_pos = np.empty(count, dtype=np.int64)
    new_obs = np.empty(count, dtype=np.float64)

    # Second pass: fill
    idx = 0
    last_pos = 0
    for i in range(L):
        pos = seg_pos[i]
        obs = seg_obs[i]
        gap = pos - last_pos
        while gap > max_distance:
            fill_obs = _gap_observation(obs)
            new_pos[idx] = last_pos + max_distance
            new_obs[idx] = fill_obs
            idx += 1
            last_pos += max_distance
            gap = pos - last_pos
        new_pos[idx] = pos
        new_obs[idx] = obs
        idx += 1
        last_pos = pos

    return new_pos[:idx], new_obs[:idx]


@numba.njit(cache=True)
def _gap_observation(obs):
    """Observation value used for filled-in positions inside a gap."""
    return obs if obs <= 0.0 else 1.0


@numba.njit(cache=True)
def _emission_prob(obs, emission_matrix, state):
    """Return the emission probability for a possibly fractional observation."""
    if obs <= 0.0:
        return emission_matrix[0, state]
    if obs <= 1.0:
        if obs == 1.0:
            return emission_matrix[1, state]
        return ((1.0 - obs) * emission_matrix[0, state]
                + obs * emission_matrix[1, state])
    if obs >= 2.0:
        return emission_matrix[2, state]

    frac = obs - 1.0
    return ((1.0 - frac) * emission_matrix[1, state]
            + frac * emission_matrix[2, state])


@numba.njit(cache=True)
def _obs_class_weights(obs):
    """Map a possibly fractional observation to missing/hom/het weights."""
    if obs <= 0.0:
        return 1.0, 0.0, 0.0
    if obs <= 1.0:
        if obs == 1.0:
            return 0.0, 1.0, 0.0
        return 1.0 - obs, obs, 0.0
    if obs >= 2.0:
        return 0.0, 0.0, 1.0

    return 0.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# 9. Forward algorithm with propagators
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _compute_full_emission(obs, emission_matrix, n_states):
    """Compute emission probability for a given observation.

    For ambiguous observations (averaged float), this returns the mean of
    emission probabilities. For integer obs, returns emission_matrix[obs, :].

    Parameters
    ----------
    obs : float
        Observation: 0=missing, 1=hom, 2=het, fractional for ambiguity.
    emission_matrix : (3, n) array
    n_states : int

    Returns
    -------
    e : (n,) array
    """
    e = np.empty(n_states, dtype=np.float64)
    for a in range(n_states):
        e[a] = _emission_prob(obs, emission_matrix, a)
    return e


@numba.njit(cache=True)
def msmc_forward(seg_pos, seg_obs, trans, emission_matrix, eq_prob,
                 fwd_props, fwd_props_miss, n_states):
    """Forward pass using precomputed propagators.

    Mirrors ``PSMC_hmm.runForward()`` from the D code.

    Parameters
    ----------
    seg_pos : (L,) int64 array
        Chopped segment positions.
    seg_obs : (L,) float64 array
        Chopped segment observations (0=missing, 1=hom, 2=het, fractional
        for ambiguous phasing).
    trans : (n, n) array
        Transition matrix.
    emission_matrix : (3, n) array
        Emission probabilities.
    eq_prob : (n,) array
        Equilibrium distribution (initial state).
    fwd_props : (max_distance, n, n) array
        Forward propagators for homozygous emission.
    fwd_props_miss : (max_distance, n, n) array
        Forward propagators for missing emission.
    n_states : int
        Number of hidden states.

    Returns
    -------
    forward_states : (L, n) array
        Scaled forward probabilities.
    scaling_factors : (L,) array
        Scaling factors (sum of unscaled forward at each position).
    """
    L = seg_pos.shape[0]
    forward_states = np.empty((L, n_states), dtype=np.float64)
    scaling_factors = np.empty(L, dtype=np.float64)

    # Initialize: f[0] = equilibrium state probabilities.
    total = 0.0
    for a in range(n_states):
        forward_states[0, a] = eq_prob[a]
        total += eq_prob[a]
    scaling_factors[0] = total
    if total > 0.0:
        inv_total = 1.0 / total
        for a in range(n_states):
            forward_states[0, a] *= inv_total

    # Temporary vectors
    tmp = np.empty(n_states, dtype=np.float64)

    for idx in range(1, L):
        pos_cur = seg_pos[idx]
        pos_prev = seg_pos[idx - 1]
        obs_cur = seg_obs[idx]

        if pos_cur == pos_prev + 1:
            # Single step: propagateSingleForward
            # f_new[a] = sum_b(trans[a, b] * f_prev[b]) * emission[obs_cur, a]
            for a in range(n_states):
                dot = 0.0
                for b in range(n_states):
                    dot += trans[a, b] * forward_states[idx - 1, b]
                forward_states[idx, a] = dot * _emission_prob(
                    obs_cur, emission_matrix, a
                )
        else:
            # Multi-step propagation
            dist = pos_cur - pos_prev

            # First: propagate from prev to pos_cur - 1 using multi propagator
            # The intermediate segment has obs = min(obs_cur, 1) which is
            # the observation type for the gap
            # But we need to look at what obs_cur is for the gap.
            # In the D code, the dummy_site at pos_cur - 1 has obs = min(segsites[index].obs[0], 1)
            # So if the target obs is het (2), the gap is hom (1); if missing (0), gap is missing.
            gap_obs = _gap_observation(obs_cur)

            multi_dist = dist - 1  # number of steps in the multi propagator
            if multi_dist > 0:
                # Use propagator for multi_dist steps
                prop_idx = multi_dist - 1  # 0-indexed

                if gap_obs <= 0.0:
                    prop = fwd_props_miss[prop_idx]
                else:
                    prop = fwd_props[prop_idx]

                # tmp = prop @ f_prev
                for a in range(n_states):
                    dot = 0.0
                    for b in range(n_states):
                        dot += prop[a, b] * forward_states[idx - 1, b]
                    tmp[a] = dot
            else:
                # No multi-step needed, just copy
                for a in range(n_states):
                    tmp[a] = forward_states[idx - 1, a]

            # Then: single step from pos_cur - 1 to pos_cur
            # f_new[a] = sum_b(trans[a, b] * tmp[b]) * emission[obs_cur, a]
            for a in range(n_states):
                dot = 0.0
                for b in range(n_states):
                    dot += trans[a, b] * tmp[b]
                forward_states[idx, a] = dot * _emission_prob(
                    obs_cur, emission_matrix, a
                )

        # Scale
        total = 0.0
        for a in range(n_states):
            total += forward_states[idx, a]
        scaling_factors[idx] = total
        if total > 0.0:
            inv_total = 1.0 / total
            for a in range(n_states):
                forward_states[idx, a] *= inv_total

    return forward_states, scaling_factors


# ---------------------------------------------------------------------------
# 10. Backward algorithm with propagators
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def msmc_backward_expectations(seg_pos, seg_obs, trans, emission_matrix,
                               forward_states, scaling_factors,
                               fwd_props, fwd_props_miss,
                               bwd_props, bwd_props_miss,
                               n_states, hmm_stride_width):
    """Backward pass with on-the-fly expected sufficient statistics.

    Mirrors ``PSMC_hmm.runBackward()`` from the D code.  Instead of
    storing the full backward array, we compute expected transition and
    emission counts during the backward sweep.

    Parameters
    ----------
    seg_pos : (L,) int64 array
    seg_obs : (L,) int8 array
    trans : (n, n) array
    emission_matrix : (3, n) array
    forward_states : (L, n) array
    scaling_factors : (L,) array
    fwd_props : (max_distance, n, n) array
        Forward propagators for homozygous emission.
    fwd_props_miss : (max_distance, n, n) array
        Forward propagators for missing emission.
    bwd_props : (max_distance, n, n) array
    bwd_props_miss : (max_distance, n, n) array
    n_states : int
    hmm_stride_width : int
        Stride for collecting expectations (default 1000).

    Returns
    -------
    transitions : (n, n) array
        Expected transition counts.
    emissions : (2, n) array
        Expected emission counts. emissions[0] = hom counts, emissions[1] = het counts.
    log_likelihood : float
        Log-likelihood of the data.
    """
    L = seg_pos.shape[0]

    # Accumulate expected counts
    transitions = np.zeros((n_states, n_states), dtype=np.float64)
    emissions = np.zeros((2, n_states), dtype=np.float64)

    # Backward state vectors (current and next)
    bwd = np.empty(n_states, dtype=np.float64)
    bwd_e = np.empty(n_states, dtype=np.float64)  # bwd * emission
    bwd_next = np.empty(n_states, dtype=np.float64)
    bwd_next_e = np.empty(n_states, dtype=np.float64)
    tmp = np.empty(n_states, dtype=np.float64)
    tmp_e = np.empty(n_states, dtype=np.float64)

    # Initialize backward at L-1
    inv_s = 1.0 / scaling_factors[L - 1] if scaling_factors[L - 1] > 0.0 else 0.0
    obs_last = seg_obs[L - 1]
    for a in range(n_states):
        bwd[a] = inv_s
        bwd_e[a] = inv_s * _emission_prob(obs_last, emission_matrix, a)

    current_bwd_index = L - 1

    # Collect expectations at stride positions
    # The D code iterates from the last position backward in steps of hmmStrideWidth
    last_seg_pos = seg_pos[L - 1]
    first_seg_pos = seg_pos[0]

    pos = last_seg_pos
    while pos > first_seg_pos and pos <= last_seg_pos:
        # Get forward state at pos - 1
        # Find the segment index for pos
        fwd_at_prev = np.empty(n_states, dtype=np.float64)
        bwd_at_pos = np.empty(n_states, dtype=np.float64)

        # Find the right index for pos in seg_pos
        right_idx = _find_right_index(seg_pos, pos, L)

        # Get backward state at right_idx
        _get_backward_state_at_index(
            right_idx, current_bwd_index, bwd, bwd_e,
            seg_pos, seg_obs, trans, emission_matrix,
            scaling_factors, bwd_props, bwd_props_miss,
            n_states, L, bwd_next, bwd_next_e, tmp, tmp_e
        )
        current_bwd_index = right_idx

        # If pos matches seg_pos[right_idx], backward state is just bwd
        if seg_pos[right_idx] == pos:
            for a in range(n_states):
                bwd_at_pos[a] = bwd[a]
        else:
            # Need to interpolate - propagate backward from right_idx to pos
            _backward_to_pos(pos, right_idx, bwd, bwd_e,
                             seg_pos, seg_obs, trans, emission_matrix,
                             bwd_props, bwd_props_miss, n_states,
                             bwd_at_pos)

        # Get forward state at pos - 1
        _forward_at_pos(pos - 1, seg_pos, seg_obs, forward_states,
                        fwd_props, fwd_props_miss, n_states, L,
                        fwd_at_prev)

        # Get the observation at pos
        site_obs = _get_obs_at_pos(seg_pos, seg_obs, pos, L)

        # Transition expectation: E[a,b] += f[pos-1, b] * trans[a, b] * bwd[pos, a] * e[obs, a]
        for a in range(n_states):
            e_a = _emission_prob(site_obs, emission_matrix, a)
            for b in range(n_states):
                transitions[a, b] += fwd_at_prev[b] * trans[a, b] * bwd_at_pos[a] * e_a

        # Emission expectation at pos
        fwd_at_pos_vec = np.empty(n_states, dtype=np.float64)
        _forward_at_pos(pos, seg_pos, seg_obs, forward_states,
                        fwd_props, fwd_props_miss, n_states, L,
                        fwd_at_pos_vec)

        _, hom_w, het_w = _obs_class_weights(site_obs)
        if hom_w > 0.0 or het_w > 0.0:
            norm = 0.0
            for a in range(n_states):
                norm += fwd_at_pos_vec[a] * bwd_at_pos[a]
            if norm > 0.0:
                for a in range(n_states):
                    val = fwd_at_pos_vec[a] * bwd_at_pos[a] / norm
                    emissions[0, a] += hom_w * val
                    emissions[1, a] += het_w * val

        pos -= hmm_stride_width

    # Log-likelihood from scaling factors
    log_lik = 0.0
    for i in range(L):
        if scaling_factors[i] > 0.0:
            log_lik += math.log(scaling_factors[i])

    return transitions, emissions, log_lik


@numba.njit(cache=True)
def _find_right_index(seg_pos, pos, L):
    """Find the rightmost segment index >= pos using linear scan."""
    for i in range(L):
        if seg_pos[i] >= pos:
            return i
    return L - 1


@numba.njit(cache=True)
def _get_obs_at_pos(seg_pos, seg_obs, pos, L):
    """Get the observation at a given position."""
    idx = _find_right_index(seg_pos, pos, L)
    if seg_pos[idx] == pos:
        return seg_obs[idx]
    else:
        # Position falls within a segment; obs = min(seg_obs[idx], 1)
        o = seg_obs[idx]
        return _gap_observation(o)


@numba.njit(cache=True)
def _forward_at_pos(pos, seg_pos, seg_obs, forward_states,
                    fwd_props, fwd_props_miss, n_states, L,
                    out):
    """Get the forward state vector at a given position.

    If pos coincides with a segment boundary, copies the stored forward state.
    Otherwise, propagates from the nearest stored forward state.
    """
    idx = _find_right_index(seg_pos, pos, L)
    if seg_pos[idx] == pos:
        for a in range(n_states):
            out[a] = forward_states[idx, a]
    else:
        # Propagate from forward_states[idx - 1] to pos
        dist = pos - seg_pos[idx - 1]
        # Observation type for the gap
        gap_obs = _gap_observation(seg_obs[idx])
        if dist > 0:
            prop_idx = dist - 1
            if gap_obs <= 0.0:
                prop = fwd_props_miss[prop_idx]
            else:
                prop = fwd_props[prop_idx]
            for a in range(n_states):
                dot = 0.0
                for b in range(n_states):
                    dot += prop[a, b] * forward_states[idx - 1, b]
                out[a] = dot
        else:
            for a in range(n_states):
                out[a] = forward_states[idx - 1, a]


@numba.njit(cache=True)
def _backward_to_pos(pos, right_idx, bwd, bwd_e,
                     seg_pos, seg_obs, trans, emission_matrix,
                     bwd_props, bwd_props_miss, n_states,
                     out):
    """Propagate backward state from right_idx to pos."""
    # From D code:
    # if pos == segsites[index].pos - 1: propagateSingleBackward
    # else: propagateSingleBackward to pos_cur-1, then propagateMultiBackward to pos
    pos_right = seg_pos[right_idx]
    obs_right = seg_obs[right_idx]

    if pos == pos_right - 1:
        # Single backward step
        # from.vec[b] = sum_a trans[a, b] * bwd_e[a]
        site_obs = _get_obs_at_pos(seg_pos, seg_obs, pos, right_idx + 1)
        for b in range(n_states):
            dot = 0.0
            for a in range(n_states):
                dot += trans[a, b] * bwd_e[a]
            out[b] = dot
    else:
        # Step 1: single backward from right_idx to right_idx.pos - 1
        tmp_bwd = np.empty(n_states, dtype=np.float64)
        for b in range(n_states):
            dot = 0.0
            for a in range(n_states):
                dot += trans[a, b] * bwd_e[a]
            tmp_bwd[b] = dot

        # Step 2: multi backward from pos_right - 1 to pos
        dist = (pos_right - 1) - pos
        gap_obs = _gap_observation(obs_right)
        if dist > 0:
            prop_idx = dist - 1
            if gap_obs <= 0.0:
                prop = bwd_props_miss[prop_idx]
            else:
                prop = bwd_props[prop_idx]
            # Backward multi: from.vec[b] = sum_a prop[a, b]^T * tmp_bwd[a]
            # = sum_a prop^T[b, a] * tmp_bwd[a]
            for b in range(n_states):
                dot = 0.0
                for a in range(n_states):
                    dot += prop[a, b] * tmp_bwd[a]
                out[b] = dot
        else:
            for b in range(n_states):
                out[b] = tmp_bwd[b]


@numba.njit(cache=True)
def _get_backward_state_at_index(target_idx, current_idx, bwd, bwd_e,
                                 seg_pos, seg_obs, trans, emission_matrix,
                                 scaling_factors, bwd_props, bwd_props_miss,
                                 n_states, L,
                                 bwd_next, bwd_next_e, tmp, tmp_e):
    """Advance the backward state from current_idx down to target_idx.

    Updates bwd and bwd_e in-place and returns current_idx = target_idx.
    """
    if target_idx == L - 1:
        inv_s = 1.0 / scaling_factors[L - 1] if scaling_factors[L - 1] > 0.0 else 0.0
        obs_last = seg_obs[L - 1]
        for a in range(n_states):
            bwd[a] = inv_s
            bwd_e[a] = inv_s * _emission_prob(obs_last, emission_matrix, a)
        return

    ci = current_idx
    while target_idx < ci:
        pos_cur = seg_pos[ci]
        pos_prev = seg_pos[ci - 1]
        obs_prev = seg_obs[ci - 1]

        if pos_cur == pos_prev + 1:
            # Single backward step
            # bwd_next[b] = sum_a trans[a, b] * bwd_e[a]
            for b in range(n_states):
                dot = 0.0
                for a in range(n_states):
                    dot += trans[a, b] * bwd_e[a]
                bwd_next[b] = dot
            # bwd_next_e[b] = bwd_next[b] * emission[obs_prev, b]
            for b in range(n_states):
                bwd_next_e[b] = bwd_next[b] * _emission_prob(
                    obs_prev, emission_matrix, b
                )
        else:
            # Multi backward step
            # Step 1: single backward from ci to ci.pos - 1
            # The "dummy" site at pos_cur - 1 has obs = min(seg_obs[ci], 1)
            dummy_obs = _gap_observation(seg_obs[ci])

            # propagateSingleBackward: from current bwd state to dummy site
            for b in range(n_states):
                dot = 0.0
                for a in range(n_states):
                    dot += trans[a, b] * bwd_e[a]
                tmp[b] = dot
            # tmp_e[b] = tmp[b] * emission_matrix[dummy_obs, b]
            for b in range(n_states):
                tmp_e[b] = tmp[b] * _emission_prob(dummy_obs, emission_matrix, b)

            # Step 2: multi backward from pos_cur - 1 to pos_prev
            dist = (pos_cur - 1) - pos_prev
            if dist > 0:
                prop_idx = dist - 1
                # Gap obs is inherited from the dummy site
                if dummy_obs <= 0.0:
                    prop = bwd_props_miss[prop_idx]
                else:
                    prop = bwd_props[prop_idx]
                # propagateMultiBackward uses CblasTrans:
                # from.vec[b] = sum_a prop^T[b, a] * to.vec[a]
                # but in the D code it's: gsl_blas_dgemv(CblasTrans, 1.0, prop, to.vec, 0.0, from.vec)
                # so from[b] = sum_a prop[a, b] * to[a]
                for b in range(n_states):
                    dot = 0.0
                    for a in range(n_states):
                        dot += prop[a, b] * tmp[a]
                    bwd_next[b] = dot
            else:
                for b in range(n_states):
                    bwd_next[b] = tmp[b]

            # Apply emission at prev site
            for b in range(n_states):
                bwd_next_e[b] = bwd_next[b] * _emission_prob(
                    obs_prev, emission_matrix, b
                )

        # Copy bwd_next -> bwd, scale
        for b in range(n_states):
            bwd[b] = bwd_next[b]
            bwd_e[b] = bwd_next_e[b]

        # Scale by 1 / scaling_factors[ci - 1]
        if scaling_factors[ci - 1] > 0.0:
            inv_s = 1.0 / scaling_factors[ci - 1]
            for b in range(n_states):
                bwd[b] *= inv_s
                bwd_e[b] *= inv_s

        ci -= 1


# ---------------------------------------------------------------------------
# 11. Single-chromosome expectation
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _single_chromosome_expectation(seg_pos, seg_obs, trans, emission_matrix,
                                   eq_prob, fwd_props, bwd_props,
                                   fwd_props_miss, bwd_props_miss,
                                   n_states, hmm_stride_width):
    """Run forward + backward on a single chromosome and return expectations.

    Parameters
    ----------
    seg_pos : (L,) int64 array
        Chopped segment positions.
    seg_obs : (L,) int8 array
        Chopped segment observations.
    trans : (n, n) array
    emission_matrix : (3, n) array
    eq_prob : (n,) array
    fwd_props : (max_distance, n, n) array
    bwd_props : (max_distance, n, n) array
    fwd_props_miss : (max_distance, n, n) array
    bwd_props_miss : (max_distance, n, n) array
    n_states : int
    hmm_stride_width : int

    Returns
    -------
    transitions : (n, n) array
    emissions : (2, n) array
    log_likelihood : float
    """
    forward_states, scaling_factors = msmc_forward(
        seg_pos, seg_obs, trans, emission_matrix, eq_prob,
        fwd_props, fwd_props_miss, n_states
    )

    transitions, emissions, log_lik = msmc_backward_expectations(
        seg_pos, seg_obs, trans, emission_matrix,
        forward_states, scaling_factors,
        fwd_props, fwd_props_miss,
        bwd_props, bwd_props_miss,
        n_states, hmm_stride_width
    )

    return transitions, emissions, log_lik


# ---------------------------------------------------------------------------
# 12. Log-likelihood for parameter optimization (M-step)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _msmc_log_likelihood(transitions, emissions, boundaries, rho, lambda_vec, mu):
    """Compute the Q-function (expected log-likelihood) for given parameters.

    Parameters
    ----------
    transitions : (n, n) array
        Expected transition counts.
    emissions : (2, n) array
        Expected emission counts. [0] = hom, [1] = het.
    boundaries : (n+1,) array
    rho : float
    lambda_vec : (n,) array
    mu : float

    Returns
    -------
    float
        Log-likelihood value.
    """
    n = lambda_vec.shape[0]
    trans = compute_transition_matrix(boundaries, rho, lambda_vec)
    e = emission_probs(boundaries, lambda_vec, mu)

    ll = 0.0
    for a in range(n):
        for b in range(n):
            tp = trans[a, b]
            if tp > 0.0:
                ll += transitions[a, b] * math.log(tp)
            elif transitions[a, b] > 0.0:
                ll += transitions[a, b] * math.log(HMM_TINY)

        # emission[0] = hom counts -> e[1, a]
        ep_hom = e[1, a]
        if ep_hom > 0.0:
            ll += emissions[0, a] * math.log(ep_hom)
        elif emissions[0, a] > 0.0:
            ll += emissions[0, a] * math.log(HMM_TINY)

        # emission[1] = het counts -> e[2, a]
        ep_het = e[2, a]
        if ep_het > 0.0:
            ll += emissions[1, a] * math.log(ep_het)
        elif emissions[1, a] > 0.0:
            ll += emissions[1, a] * math.log(HMM_TINY)

    return ll


# ---------------------------------------------------------------------------
# 13. Powell's method for M-step
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _brent_minimize(ax, bx, cx, p, xi, transitions, emissions, boundaries,
                    mu, n_params, n_intervals, par_map, fixed_rho, init_rho,
                    tol=3.0e-8):
    """Brent's method for 1D minimization along a direction.

    Parameters
    ----------
    ax, bx, cx : float
        Bracket points.
    p : (n_params,) array
        Current parameter vector.
    xi : (n_params,) array
        Search direction.
    transitions : (n, n) array
    emissions : (2, n) array
    boundaries : (n+1,) array
    mu : float
    n_params : int
    n_intervals : int
    par_map : (n_intervals,) int array
    fixed_rho : bool
    init_rho : float
    tol : float

    Returns
    -------
    xmin : float
    fmin : float
    """
    ITMAX = 100
    CGOLD = 0.3819660
    ZEPS = 1.0e-25

    xt = np.empty(n_params, dtype=np.float64)

    a = min(ax, cx)
    b = max(ax, cx)

    v = bx
    w = bx
    x = bx
    e = 0.0
    d = 0.0

    # Evaluate at x
    for j in range(n_params):
        xt[j] = p[j] + x * xi[j]
    fx = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                      n_intervals, par_map, fixed_rho, init_rho)
    fv = fx
    fw = fx

    for iteration in range(ITMAX):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS
        tol2 = 2.0 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            return x, fx

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            pp = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                pp = -pp
            q = abs(q)
            etemp = e
            e = d
            if abs(pp) >= abs(0.5 * q * etemp) or pp <= q * (a - x) or pp >= q * (b - x):
                e = a - x if x >= xm else b - x
                d = CGOLD * e
            else:
                d = pp / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if xm - x >= 0 else -tol1
        else:
            e = a - x if x >= xm else b - x
            d = CGOLD * e

        u = x + d if abs(d) >= tol1 else x + (tol1 if d >= 0 else -tol1)

        for j in range(n_params):
            xt[j] = p[j] + u * xi[j]
        fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                          n_intervals, par_map, fixed_rho, init_rho)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x, fx


@numba.njit(cache=True)
def _sign(a, b):
    """Return ``abs(a)`` with the sign of ``b``."""
    if b >= 0.0:
        return a if a >= 0.0 else -a
    return -a if a >= 0.0 else a


@numba.njit(cache=True)
def _bracket(ax_in, bx_in, p, xi, transitions, emissions, boundaries,
             mu, n_params, n_intervals, par_map, fixed_rho, init_rho):
    """Bracket a minimum along direction xi."""
    GOLD = 1.618034
    GLIMIT = 100.0
    TINY = 1.0e-20

    xt = np.empty(n_params, dtype=np.float64)

    ax = ax_in
    bx = bx_in

    for j in range(n_params):
        xt[j] = p[j] + ax * xi[j]
    fa = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                      n_intervals, par_map, fixed_rho, init_rho)
    for j in range(n_params):
        xt[j] = p[j] + bx * xi[j]
    fb = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                      n_intervals, par_map, fixed_rho, init_rho)

    if fb > fa:
        ax, bx = bx, ax
        fa, fb = fb, fa

    cx = bx + GOLD * (bx - ax)
    for j in range(n_params):
        xt[j] = p[j] + cx * xi[j]
    fc = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                      n_intervals, par_map, fixed_rho, init_rho)

    while fb > fc:
        r = (bx - ax) * (fb - fc)
        q = (bx - cx) * (fb - fa)
        u = bx - ((bx - cx) * q - (bx - ax) * r) / (
            2.0 * _sign(max(abs(q - r), TINY), q - r)
        )
        ulim = bx + GLIMIT * (cx - bx)

        if (bx - u) * (u - cx) > 0.0:
            for j in range(n_params):
                xt[j] = p[j] + u * xi[j]
            fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                              n_intervals, par_map, fixed_rho, init_rho)
            if fu < fc:
                ax = bx
                bx = u
                fa = fb
                fb = fu
                break
            elif fu > fb:
                cx = u
                fc = fu
                break
            u = cx + GOLD * (cx - bx)
            for j in range(n_params):
                xt[j] = p[j] + u * xi[j]
            fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                              n_intervals, par_map, fixed_rho, init_rho)
        elif (cx - u) * (u - ulim) > 0.0:
            for j in range(n_params):
                xt[j] = p[j] + u * xi[j]
            fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                              n_intervals, par_map, fixed_rho, init_rho)
            if fu < fc:
                old_bx = bx
                old_cx = cx
                old_u = u
                fb = fc
                fc = fu
                bx = old_cx
                cx = old_u
                u = old_u + GOLD * (old_u - old_cx)
                for j in range(n_params):
                    xt[j] = p[j] + u * xi[j]
                fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                                  n_intervals, par_map, fixed_rho, init_rho)
        elif (u - ulim) * (ulim - cx) >= 0.0:
            u = ulim
            for j in range(n_params):
                xt[j] = p[j] + u * xi[j]
            fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                              n_intervals, par_map, fixed_rho, init_rho)
        else:
            u = cx + GOLD * (cx - bx)
            for j in range(n_params):
                xt[j] = p[j] + u * xi[j]
            fu = _eval_neg_ll(xt, transitions, emissions, boundaries, mu,
                              n_intervals, par_map, fixed_rho, init_rho)

        ax = bx
        bx = cx
        cx = u
        fa = fb
        fb = fc
        fc = fu

    return ax, bx, cx


@numba.njit(cache=True)
def _eval_neg_ll(x, transitions, emissions, boundaries, mu,
                 n_intervals, par_map, fixed_rho, init_rho):
    """Evaluate negative log-likelihood from parameter vector x.

    x contains log(lambda) values for each segment group, and optionally
    log(rho) as the last element.
    """
    n_seg = par_map.max() + 1  # number of segment groups
    lambda_vec = np.empty(n_intervals, dtype=np.float64)
    for i in range(n_intervals):
        lambda_vec[i] = math.exp(x[par_map[i]])

    if fixed_rho:
        rho = init_rho
    else:
        rho = math.exp(x[n_seg])

    if rho <= 0.0:
        rho = 1e-10

    ll = _msmc_log_likelihood(transitions, emissions, boundaries, rho,
                              lambda_vec, mu)
    return -ll


@numba.njit(cache=True)
def _powell_minimize(x0, transitions, emissions, boundaries, mu,
                     n_intervals, par_map, fixed_rho, init_rho,
                     ftol=3.0e-8):
    """Powell's method for multidimensional minimization of -Q.

    Parameters
    ----------
    x0 : (n_params,) array
        Initial parameter vector (log-space).
    transitions : (n, n) array
    emissions : (2, n) array
    boundaries : (n+1,) array
    mu : float
    n_intervals : int
    par_map : (n_intervals,) int array
    fixed_rho : bool
    init_rho : float
    ftol : float

    Returns
    -------
    p : (n_params,) optimized parameter vector
    fret : float, function value at optimum
    """
    ITMAX = 200
    TINY = 1.0e-25

    n = x0.shape[0]
    p = x0.copy()

    # Initialize direction matrix to identity
    ximat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ximat[i, i] = 1.0

    xi = np.empty(n, dtype=np.float64)
    pt = np.empty(n, dtype=np.float64)
    ptt = np.empty(n, dtype=np.float64)

    fret = _eval_neg_ll(p, transitions, emissions, boundaries, mu,
                        n_intervals, par_map, fixed_rho, init_rho)

    for j in range(n):
        pt[j] = p[j]

    for iteration in range(ITMAX):
        fp = fret
        ibig = 0
        delta = 0.0

        for i in range(n):
            for j in range(n):
                xi[j] = ximat[j, i]
            fptt = fret

            # Line minimization along xi
            ax, bx, cx = _bracket(0.0, 1.0, p, xi, transitions, emissions,
                                  boundaries, mu, n, n_intervals, par_map,
                                  fixed_rho, init_rho)
            xmin, fret = _brent_minimize(ax, bx, cx, p, xi, transitions,
                                         emissions, boundaries, mu, n,
                                         n_intervals, par_map, fixed_rho,
                                         init_rho)

            for j in range(n):
                xi[j] *= xmin
                p[j] += xi[j]

            if fptt - fret > delta:
                delta = fptt - fret
                ibig = i + 1

        if 2.0 * (fp - fret) <= ftol * (abs(fp) + abs(fret)) + TINY:
            return p, fret

        for j in range(n):
            ptt[j] = 2.0 * p[j] - pt[j]
            xi[j] = p[j] - pt[j]
            pt[j] = p[j]

        fptt = _eval_neg_ll(ptt, transitions, emissions, boundaries, mu,
                            n_intervals, par_map, fixed_rho, init_rho)
        if fptt < fp:
            t = 2.0 * (fp - 2.0 * fret + fptt) * (fp - fret - delta) ** 2 - delta * (fp - fptt) ** 2
            if t < 0.0:
                ax, bx, cx = _bracket(0.0, 1.0, p, xi, transitions, emissions,
                                      boundaries, mu, n, n_intervals, par_map,
                                      fixed_rho, init_rho)
                xmin, fret = _brent_minimize(ax, bx, cx, p, xi, transitions,
                                             emissions, boundaries, mu, n,
                                             n_intervals, par_map, fixed_rho,
                                             init_rho)
                for j in range(n):
                    xi[j] *= xmin
                    p[j] += xi[j]

                for j in range(n):
                    ximat[j, ibig - 1] = ximat[j, n - 1]
                    ximat[j, n - 1] = xi[j]

    return p, fret


# ---------------------------------------------------------------------------
# 14. EM step
# ---------------------------------------------------------------------------

def _msmc_em_step(
    segments_list: list[dict],
    pairs: list[tuple[int, int]],
    boundaries: np.ndarray,
    lambda_vec: np.ndarray,
    mu: float,
    rho: float,
    par_map: np.ndarray,
    segment_sizes: list[int],
    fixed_rho: bool,
    hmm_stride_width: int,
    max_distance: int,
) -> tuple[np.ndarray, float, float, float]:
    """Run one EM iteration.

    Parameters
    ----------
    segments_list : list of dicts
        Segment data per chromosome from ``read_multihetsep``.
    pairs : list of (int, int)
        Haplotype pairs to analyze.
    boundaries : (n+1,) array
        Time interval boundaries.
    lambda_vec : (n,) array
        Current coalescence rates.
    mu : float
        Mutation rate (scaled).
    rho : float
        Recombination rate (scaled).
    par_map : (n_intervals,) int array
    segment_sizes : list of int
    fixed_rho : bool
    hmm_stride_width : int
    max_distance : int

    Returns
    -------
    lambda_vec_new : (n,) array
        Updated coalescence rates.
    rho_new : float
        Updated recombination rate.
    log_likelihood : float
    q_after : float
        Q-function value after maximization.
    """
    n_states = lambda_vec.shape[0]
    n_intervals = n_states

    # Build HMM parameters
    trans = compute_transition_matrix(boundaries, rho, lambda_vec)
    e = emission_probs(boundaries, lambda_vec, mu)
    eq = equilibrium_prob(boundaries, lambda_vec)

    # Precompute propagators
    emission_hom = e[1, :].copy()
    fwd_props, bwd_props, fwd_props_miss, bwd_props_miss = (
        precompute_all_propagators(trans, emission_hom, max_distance)
    )

    # E-step: accumulate expectations over all pairs and chromosomes
    total_transitions = np.zeros((n_states, n_states), dtype=np.float64)
    total_emissions = np.zeros((2, n_states), dtype=np.float64)
    total_ll = 0.0

    for seg_dict in segments_list:
        obs_dict = seg_dict["obs"]
        positions = seg_dict["positions"]
        n_called = seg_dict["n_called"]

        for pair in pairs:
            pair_key = tuple(pair)
            if pair_key not in obs_dict:
                continue

            pair_obs = obs_dict[pair_key]

            # Build segment representation for this pair on this chromosome
            seg_pos, seg_obs = _build_segments_for_pair(
                positions, n_called, pair_obs
            )

            if seg_pos.shape[0] < 2:
                continue

            # Chop segments
            seg_pos_c, seg_obs_c = chop_segments(seg_pos, seg_obs, max_distance)

            if seg_pos_c.shape[0] < 2:
                continue

            # Run forward-backward
            tr, em, ll = _single_chromosome_expectation(
                seg_pos_c, seg_obs_c, trans, e, eq,
                fwd_props, bwd_props, fwd_props_miss, bwd_props_miss,
                n_states, hmm_stride_width,
            )

            total_transitions += tr
            total_emissions += em
            total_ll += ll

    # Q before maximization
    q_before = _msmc_log_likelihood(
        total_transitions, total_emissions, boundaries, rho, lambda_vec, mu
    )

    n_free = par_map.max() + 1
    n_opt_params = n_free + (0 if fixed_rho else 1)

    x0 = np.empty(n_opt_params, dtype=np.float64)
    for i in range(n_free):
        idx = 0
        for j in range(lambda_vec.shape[0]):
            if par_map[j] == i:
                idx = j
                break
        x0[i] = math.log(lambda_vec[idx])

    if not fixed_rho:
        x0[n_free] = math.log(rho)

    x_opt, fret = _powell_minimize(
        x0,
        total_transitions,
        total_emissions,
        boundaries,
        mu,
        n_intervals,
        par_map,
        fixed_rho,
        rho,
    )

    # Extract optimized parameters
    lambda_vec_new = np.empty(n_states, dtype=np.float64)
    for i in range(n_states):
        lambda_vec_new[i] = math.exp(x_opt[par_map[i]])

    if fixed_rho:
        rho_new = rho
    else:
        rho_new = math.exp(x_opt[n_free])

    q_after = fret

    return lambda_vec_new, rho_new, total_ll, q_after


def _build_segments_for_pair(positions, n_called, pair_obs):
    """Build the segment position and observation arrays for one pair.

    Mirrors the D readSegSites logic: each multihetsep line produces
    segment(s) with position, and observation per pair.

    Parameters
    ----------
    positions : (L,) int64 array
        Positions from multihetsep.
    n_called : (L,) int64 array
        Number of called sites per segment.
    pair_obs : (L,) int8 or float64 array
        Observations for this pair (0=missing, 1=hom, 2=het).

    Returns
    -------
    seg_pos : int64 array
    seg_obs : float64 array
    """
    L = positions.shape[0]
    # Worst case: each line can produce up to 3 segments
    max_segs = L * 3
    seg_pos_buf = np.empty(max_segs, dtype=np.int64)
    seg_obs_buf = np.empty(max_segs, dtype=np.float64)
    count = 0
    last_pos = -1

    for i in range(L):
        pos = int(positions[i])
        nc = int(n_called[i])
        obs = float(pair_obs[i])

        if last_pos == -1:
            last_pos = pos - nc

        # Missing data gap before called sites
        if nc < pos - last_pos:
            seg_pos_buf[count] = pos - nc
            seg_obs_buf[count] = 0.0  # missing
            count += 1

        if obs == 0.0:
            # Missing at this site
            if nc > 1:
                seg_pos_buf[count] = pos - 1
                seg_obs_buf[count] = 1.0  # hom for the called sites before
                count += 1
            seg_pos_buf[count] = pos
            seg_obs_buf[count] = 0.0  # missing
            count += 1
        elif obs < 0.0:
            # Ambiguous sites skipped by upstream -s become missing at pos
            # without adding an extra homozygous site at pos - 1.
            seg_pos_buf[count] = pos
            seg_obs_buf[count] = 0.0
            count += 1
        else:
            # hom or het
            seg_pos_buf[count] = pos
            seg_obs_buf[count] = obs
            count += 1

        last_pos = pos

    return seg_pos_buf[:count].copy(), seg_obs_buf[:count].copy()


# ---------------------------------------------------------------------------
# 15. Theta estimation from data
# ---------------------------------------------------------------------------

def _estimate_theta(segments_list, pairs):
    """Estimate theta from expanded segments, matching D code's getTheta.

    The D code operates on expanded SegSite_t arrays (with missing-gap
    markers). We replicate this by first expanding the raw data via
    ``_build_segments_for_pair``, then computing theta over the expanded
    segments. This matters because missing-gap markers update ``lastPos``,
    changing inter-site distance calculations.
    """
    total_hets = 0.0
    total_called = 0

    for seg_dict in segments_list:
        obs_dict = seg_dict["obs"]
        positions = seg_dict["positions"]
        n_called = seg_dict["n_called"]

        for pair in pairs:
            pair_key = tuple(pair)
            if pair_key not in obs_dict:
                continue

            pair_obs = obs_dict[pair_key]

            # Expand segments like the D code's readSegSites
            seg_pos, seg_obs = _build_segments_for_pair(
                positions, n_called, pair_obs
            )

            # Now compute theta on expanded segments (matches D getTheta)
            last_pos = 0
            for i in range(len(seg_pos)):
                pos = int(seg_pos[i])
                obs = float(seg_obs[i])

                if obs > 0:
                    if last_pos > 0:
                        total_called += pos - last_pos
                    if obs > 1.0:
                        total_hets += 1.0

                # lastPos updates unconditionally (matches D code)
                last_pos = pos

    if total_called == 0:
        return 0.001  # fallback
    return total_hets / total_called


# ---------------------------------------------------------------------------
# 16. Public API
# ---------------------------------------------------------------------------

@dataclass
class Msmc2Result:
    """Results from an MSMC2 run."""

    time_boundaries: np.ndarray   # (n+1,) time boundaries
    lambda_vec: np.ndarray        # (n,) coalescence rates per interval
    ne: np.ndarray                # (n,) N_e(t) = 1 / (2 * mu * lambda)
    time_years: np.ndarray        # (n,) left boundary in years
    mu: float = 0.0
    rho: float = 0.0
    log_likelihood: float = 0.0
    n_iterations: int = 0
    time_pattern: str = ""
    rounds: list[dict] = field(default_factory=list)


def msmc2(
    data: SmcData,
    n_iterations: int = 20,
    time_pattern: str = "1*2+25*1+1*2+1*3",
    mu: float = 1.25e-8,
    rho_over_mu: float = 0.25,
    fixed_rho: bool = False,
    generation_time: float = 25.0,
    stride_width: int = 1000,
    max_distance: int = 1000,
    quantile_bounds: bool = False,
    time_factor: float = 1.0,
    implementation: str = "auto",
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run MSMC2 demographic inference.

    Parameters
    ----------
    data : SmcData
        Input data from ``smckit.io.read_multihetsep()``.
    n_iterations : int
        Number of EM iterations (default 20).
    time_pattern : str
        Time segment pattern (default ``"1*2+25*1+1*2+1*3"``).
    mu : float
        Per-base per-generation mutation rate.
    rho_over_mu : float
        Initial ratio of recombination over mutation rate (default 0.25).
    fixed_rho : bool
        If True, keep recombination rate fixed during optimization.
    generation_time : float
        Generation time in years (for scaling output).
    stride_width : int
        HMM stride width for expectation collection (default 1000).
    max_distance : int
        Maximum distance for propagator precomputation (default 1000).
    quantile_bounds : bool
        Use quantile boundaries instead of Li & Durbin (default False).
    time_factor : float
        Multiplicative factor for time boundary computation (default 1.0).
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo MSMC2
        port. ``"upstream"`` currently raises because no public upstream bridge
        is exposed yet. ``"auto"`` resolves to the best available implementation.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["msmc2"]``.
    """
    implementation = normalize_implementation(implementation)
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("msmc2"),
    )
    if implementation_used == "upstream":
        return _msmc2_upstream(
            data,
            n_iterations=n_iterations,
            time_pattern=time_pattern,
            rho_over_mu=rho_over_mu,
            fixed_rho=fixed_rho,
            mu=mu,
            quantile_bounds=quantile_bounds,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )
    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported msmc2 native_options keys: {unsupported}")

    segments_list = data.uns["segments"]
    pairs = data.uns["pairs"]
    n_pairs = len(pairs)

    # Parse time segment pattern
    segment_sizes = parse_time_segment_pattern(time_pattern)
    n_intervals = sum(segment_sizes)
    par_map, n_free, _ = _build_par_map(segment_sizes)

    logger.info(
        "MSMC2: n_intervals=%d, n_free=%d, n_pairs=%d, pattern=%s",
        n_intervals, n_free, n_pairs, time_pattern,
    )

    # Compute time boundaries
    if quantile_bounds:
        time_constant = time_factor / n_pairs
        boundaries = quantile_boundaries(n_intervals, time_constant)
    else:
        time_constant = time_factor * 0.1 / n_pairs
        boundaries = li_durbin_boundaries(n_intervals, time_constant)

    # Estimate mutation rate (theta) from data
    theta = _estimate_theta(segments_list, pairs) / 2.0
    if theta <= 0.0:
        theta = mu  # fallback

    # Use theta as the internal mutation rate (matching D code convention)
    mu_internal = theta
    rho_internal = mu_internal * rho_over_mu

    logger.info(
        "MSMC2: estimated mu_internal=%.6e, rho_internal=%.6e",
        mu_internal, rho_internal,
    )

    # Initialize lambda_vec to all 1s
    lambda_vec = np.ones(n_intervals, dtype=np.float64)

    # EM loop
    rounds: list[dict] = []

    # Record initial state
    rounds.append({
        "round": 0,
        "mu": mu_internal,
        "rho": rho_internal,
        "lambda": lambda_vec.copy(),
        "boundaries": boundaries.copy(),
    })

    for iteration in range(n_iterations):
        logger.info("MSMC2: EM iteration %d/%d", iteration + 1, n_iterations)

        lambda_vec_new, rho_new, ll, q_after = _msmc_em_step(
            segments_list,
            pairs,
            boundaries,
            lambda_vec,
            mu_internal,
            rho_internal,
            par_map,
            segment_sizes,
            fixed_rho,
            stride_width,
            max_distance,
        )

        lambda_vec = lambda_vec_new
        rho_internal = rho_new

        rd = {
            "round": iteration + 1,
            "log_likelihood": ll,
            "q_after": q_after,
            "mu": mu_internal,
            "rho": rho_internal,
            "lambda": lambda_vec.copy(),
            "boundaries": boundaries.copy(),
        }
        rounds.append(rd)
        logger.info(
            "  LL=%.2f Q=%.4f rho=%.6e",
            ll, q_after, rho_internal,
        )

    # Final results
    # Convert to physical units
    # In the D code, the final output scales boundaries by mu and lambda by 1/mu
    # boundaries are in units of 2*N_e generations (coalescent units)
    # lambda is the inverse population size (also in coalescent units)
    #
    # MSMC2 output convention:
    #   left_boundary_scaled = boundary * mu_internal
    #   right_boundary_scaled = boundary * mu_internal
    #   lambda_scaled = lambda / mu_internal
    #
    # To get Ne(t):
    #   N_e(t) = 1 / (2 * mu * lambda_scaled)
    #         = 1 / (2 * mu * lambda / mu_internal)
    #         = mu_internal / (2 * mu * lambda)
    #
    # Time in years:
    #   t_years = boundary * mu_internal / mu * generation_time

    lambda_scaled = lambda_vec / mu_internal
    ne = 1.0 / (2.0 * mu * lambda_scaled)

    # Left boundaries for each interval in years
    left_bounds = boundaries[:n_intervals].copy()
    time_years = left_bounds * mu_internal / mu * generation_time

    result = Msmc2Result(
        time_boundaries=boundaries.copy(),
        lambda_vec=lambda_vec.copy(),
        ne=ne,
        time_years=time_years,
        mu=mu_internal,
        rho=rho_internal,
        log_likelihood=rounds[-1].get("log_likelihood", 0.0) if len(rounds) > 1 else 0.0,
        n_iterations=n_iterations,
        time_pattern=time_pattern,
        rounds=rounds,
    )

    data.results["msmc2"] = annotate_result({
        "time_boundaries": result.time_boundaries,
        "left_boundary": boundaries[:n_intervals] * mu_internal,
        "right_boundary": boundaries[1:] * mu_internal,
        "lambda": lambda_vec / mu_internal,
        "lambda_raw": lambda_vec.copy(),
        "ne": result.ne,
        "time_years": result.time_years,
        "mu": mu_internal,
        "rho": rho_internal,
        "log_likelihood": result.log_likelihood,
        "time_pattern": result.time_pattern,
        "rounds": result.rounds,
    }, implementation_requested=implementation, implementation_used=implementation_used)
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time

    return data


def _msmc2_binary_path() -> Path | None:
    status = upstream_status("msmc2")
    cache_path = Path(status["cache_path"]) / "bin/msmc2"
    if cache_path.exists():
        return cache_path
    for candidate in [
        Path(status["vendor_path"]) / "build/release/msmc2",
        Path(status["vendor_path"]) / "build/msmc2",
    ]:
        if candidate.exists():
            return candidate
    return None


def _pair_indices_arg(pairs: list[tuple[int, int]] | None) -> str | None:
    if not pairs:
        return None
    return ",".join(f"{i}-{j}" for i, j in pairs)


def _msmc2_upstream(
    data: SmcData,
    *,
    n_iterations: int,
    time_pattern: str,
    rho_over_mu: float,
    fixed_rho: bool,
    mu: float,
    quantile_bounds: bool,
    implementation_requested: str,
    upstream_options: dict | None,
) -> SmcData:
    status = upstream_status("msmc2")
    if not status["cache_ready"]:
        bootstrap_upstream("msmc2")
    binary = _msmc2_binary_path()
    if binary is None:
        raise RuntimeError("Upstream MSMC2 executable is unavailable after bootstrap.")
    input_paths = data.uns.get("source_paths")
    if not input_paths:
        raise ValueError("Upstream MSMC2 requires source multihetsep file path(s).")
    effective_args = {
        "maxIterations": int(n_iterations),
        "timeSegmentPattern": time_pattern,
        "rhoOverMu": float(rho_over_mu),
        "pairIndices": _pair_indices_arg(data.uns.get("pairs")),
        "fixedRecombination": bool(fixed_rho),
        "quantileBoundaries": bool(quantile_bounds),
    }
    if upstream_options:
        effective_args.update(upstream_options)
    with tempfile.TemporaryDirectory(prefix="smckit-msmc2-") as tmpdir:
        out_prefix = Path(tmpdir) / "msmc2"
        cmd = [
            str(binary),
            "-i",
            str(int(n_iterations)),
            "-o",
            str(out_prefix),
            "-r",
            repr(float(rho_over_mu)),
            "-p",
            time_pattern,
        ]
        pair_arg = _pair_indices_arg(data.uns.get("pairs"))
        if pair_arg:
            cmd.extend(["-I", pair_arg])
        if fixed_rho:
            cmd.append("-R")
        if quantile_bounds:
            cmd.append("--quantileBoundaries")
        cmd.extend([str(Path(p)) for p in input_paths])
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream MSMC2 backend failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        final_path = out_prefix.with_suffix(".final.txt")
        parsed = read_msmc_output(final_path)
        left = np.asarray(parsed["left_boundary"], dtype=np.float64)
        right = np.asarray(parsed["right_boundary"], dtype=np.float64)
        lam = np.asarray(parsed["lambda"], dtype=np.float64)
        ne = 1.0 / np.maximum(lam, 1e-300) / max(2.0 * mu, 1e-300)
        time_years = left / max(mu, 1e-300)
        data.results["msmc2"] = annotate_result(
            {
                "time_index": np.asarray(parsed["time_index"], dtype=np.int64),
                "left_boundary": left,
                "right_boundary": right,
                "lambda": lam,
                "ne": ne,
                "time": left,
                "time_years": time_years,
                "backend": "upstream",
                "upstream": standard_upstream_metadata(
                    "msmc2",
                    effective_args=effective_args,
                    extra={
                        "binary": str(binary),
                        "out_prefix": str(out_prefix),
                        "final_path": str(final_path),
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    },
                ),
            },
            implementation_requested=implementation_requested,
            implementation_used="upstream",
        )
    data.params["mu"] = mu
    return data
