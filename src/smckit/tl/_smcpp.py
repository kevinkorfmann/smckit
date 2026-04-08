"""SMC++: SMC with many unphased individuals.

Reimplementation of Terhorst, Kamm & Song (2017).
See docs/smcpp_internals.md for the full mathematical reference.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar
from scipy.special import comb

from smckit._core import SmcData
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    standard_upstream_metadata,
    warn_if_native_not_trusted,
)

logger = logging.getLogger(__name__)

SMCPP_T_INF = 1000.0
SMCPP_INTERNAL_PIECES = 100
SMCPP_MSTEP_XATOL = 1e-4


# ---------------------------------------------------------------------------
# Time discretization
# ---------------------------------------------------------------------------

def compute_time_intervals(
    n_intervals: int,
    max_t: float = 15.0,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute exponentially spaced time boundaries.

    Same formula as PSMC: t_k = alpha * (exp(beta * k) - 1).

    Parameters
    ----------
    n_intervals : int
        Number of coalescence intervals (= number of HMM states).
    max_t : float
        Maximum coalescent time (units of 2N_0).
    alpha : float
        Controls resolution near t=0.

    Returns
    -------
    t : (n_intervals + 1,) time boundaries.
    """
    K = n_intervals
    t = np.empty(K + 1, dtype=np.float64)
    beta = np.log(1.0 + max_t / alpha) / K
    for k in range(K):
        t[k] = alpha * (np.exp(beta * k) - 1.0)
    t[K] = max_t
    return t


def _balanced_hidden_states_constant(total_breaks: int, eta0: float = 2.0) -> np.ndarray:
    """Approximate upstream one-pop hidden-state initialization on a constant model."""
    if total_breaks < 2:
        return np.array([0.0, np.inf], dtype=np.float64)
    m = total_breaks - 1
    hs = [0.0]
    for i in range(1, m):
        hs.append(-eta0 * np.log((m - i) / m))
    hs.append(np.inf)
    return np.asarray(hs, dtype=np.float64)


def _balanced_hidden_states_piecewise(t: np.ndarray, eta: np.ndarray, total_breaks: int) -> np.ndarray:
    """Port of upstream ``balance_hidden_states`` for a piecewise-constant model."""
    if total_breaks < 2:
        return np.array([0.0, np.inf], dtype=np.float64)

    cumulative = np.zeros(len(eta) + 1, dtype=np.float64)
    for k in range(len(eta)):
        cumulative[k + 1] = cumulative[k] + (t[k + 1] - t[k]) / max(eta[k], 1e-300)

    m = total_breaks - 1
    hs = [0.0]
    for i in range(1, m):
        target_r = -np.log((m - i) / m)
        idx = int(np.searchsorted(cumulative, target_r, side="right") - 1)
        idx = min(max(idx, 0), len(eta) - 1)
        dt = (target_r - cumulative[idx]) * eta[idx]
        hs.append(float(t[idx] + dt))
    hs.append(np.inf)
    return np.asarray(hs, dtype=np.float64)


def _native_time_grid(
    n_intervals: int,
    max_t: float,
    alpha: float,
    n_distinguished: int,
    init_eta0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return native time boundaries and optional upstream-style hidden states."""
    if n_distinguished != 2:
        return compute_time_intervals(n_intervals, max_t, alpha), None

    hs = _balanced_hidden_states_constant(2 * n_intervals, eta0=init_eta0)
    finite_hs = hs[np.isfinite(hs)]
    if len(finite_hs) < n_intervals + 1:
        return compute_time_intervals(n_intervals, max_t, alpha), hs
    t = np.r_[0.0, finite_hs[1::2]]
    if len(t) == n_intervals:
        t = np.r_[t, finite_hs[-1]]
    return np.asarray(t[: n_intervals + 1], dtype=np.float64), hs


def _onepop_knots_from_hidden_states(hidden_states: np.ndarray) -> np.ndarray:
    """Mirror upstream ``_init_knots`` for the one-pop piecewise model."""
    hs = np.asarray(hidden_states, dtype=np.float64)
    if hs.size < 3:
        return np.array([0.0], dtype=np.float64)
    return np.r_[0.0, hs[1:-1:2]]


def _piecewise_log_eta_at_knots(
    old_knots: np.ndarray,
    old_log_eta: np.ndarray,
    new_knots: np.ndarray,
) -> np.ndarray:
    """Evaluate an upstream-style piecewise spline at new knot locations."""
    old_knots = np.asarray(old_knots, dtype=np.float64)
    old_log_eta = np.asarray(old_log_eta, dtype=np.float64)
    new_knots = np.asarray(new_knots, dtype=np.float64)
    if old_log_eta.size == 0 or new_knots.size == 0:
        return np.asarray(old_log_eta, dtype=np.float64).copy()
    idx = np.searchsorted(old_knots, new_knots, side="right") - 1
    idx = np.clip(idx, 0, old_log_eta.size - 1)
    return old_log_eta[idx]


def _empirical_hidden_states_from_records(
    records: list[dict],
    theta: float,
    total_breaks: int,
    window_size: int,
) -> np.ndarray | None:
    """Approximate upstream empirical hidden-state initialization from mutation counts."""
    if total_breaks < 3 or theta <= 0:
        return None

    counts: list[float] = []
    for rec in records:
        seen = 0
        nmiss = 0
        mut = 0
        for span, a_obs, _b_obs in rec["observations"]:
            if span <= 0:
                continue
            span_left = int(span)
            while span_left > 0:
                take = min(span_left, window_size - seen)
                if a_obs >= 0:
                    mut += take * (int(a_obs) % 2)
                    nmiss += take
                seen += take
                span_left -= take
                if seen == window_size:
                    if nmiss > 0.5 * window_size:
                        counts.append(mut * window_size / max(nmiss, 1))
                    seen = 0
                    nmiss = 0
                    mut = 0
        if seen > 0 and nmiss > 0.5 * window_size:
            counts.append(mut * window_size / max(nmiss, 1))

    x = np.asarray([c for c in counts if c > 0], dtype=np.float64)
    if x.size < max(total_breaks, 2):
        return None
    p = np.logspace(np.log10(0.01), np.log10(0.99), total_breaks - 2)
    q = np.quantile(x, p) / max(2.0 * theta * window_size, 1e-300)
    q = np.maximum.accumulate(q)
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + 1e-6
    return np.r_[0.0, q, np.inf]


# ---------------------------------------------------------------------------
# Lineage counting death process
# ---------------------------------------------------------------------------

def _lineage_rate_matrix(n: int) -> np.ndarray:
    """Build rate matrix for the pure death process of n lineages.

    States indexed 0..n-1, representing 1..n lineages.
    Rate from j to j-1 lineages: j*(j-1)/2 (standard coalescent).
    """
    Q = np.zeros((n, n), dtype=np.float64)
    for j in range(2, n + 1):
        rate = j * (j - 1) / 2.0
        idx = j - 1
        Q[idx, idx] = -rate
        Q[idx, idx - 1] = rate
    return Q


def _propagate_lineages(
    Q: np.ndarray,
    n_undist: int,
    t: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Compute lineage probability distribution at each time boundary.

    Returns p[k, j-1] = P(j undist lineages at time t[k]).
    """
    K = len(eta)
    p = np.zeros((K + 1, n_undist), dtype=np.float64)
    p[0, n_undist - 1] = 1.0

    for k in range(K):
        tau_k = t[k + 1] - t[k]
        if tau_k > 0 and eta[k] > 0:
            P_k = expm(Q * tau_k / eta[k])
            p[k + 1] = p[k] @ P_k
        else:
            p[k + 1] = p[k].copy()

    return p


def _expected_lineages(p_bound: np.ndarray, n_undist: int) -> np.ndarray:
    """Compute E[A(t_k)] at each boundary from lineage distributions."""
    j_vec = np.arange(1, n_undist + 1, dtype=np.float64)
    return np.array([float(j_vec @ p_bound[k]) for k in range(p_bound.shape[0])])


# ---------------------------------------------------------------------------
# SFS weights (Polanski-Kimmel formula)
# ---------------------------------------------------------------------------

def _sfs_weights(n: int) -> np.ndarray:
    """Compute SFS weights w[j-1, b-1] = P(b derived | j lineages, sample n).

    Formula: w(b, j, n) = C(n-b-1, j-2) / C(n-1, j-1)
    """
    w = np.zeros((n, n), dtype=np.float64)
    w[0, n - 1] = 1.0  # j=1: single lineage subtends all n

    for j in range(2, n + 1):
        denom = comb(n - 1, j - 1, exact=True)
        if denom == 0:
            continue
        for b in range(1, n - j + 2):
            numer = comb(n - b - 1, j - 2, exact=True)
            w[j - 1, b - 1] = numer / denom

    return w


# ---------------------------------------------------------------------------
# PSMC-style state-dependent transition matrix for SMC++
# ---------------------------------------------------------------------------

def _compute_psmc_style_transition(
    n_undist: int,
    n_states: int,
    t: np.ndarray,
    eta: np.ndarray,
    rho_base: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute state-dependent transition matrix using PSMC formulas
    adapted for n-lineage SMC++.

    The key adaptation: replace PSMC's coalescence rate 1/lambda_k with
    E[A_k]/eta_k, where E[A_k] is the expected number of undistinguished
    lineages in interval k. This is equivalent to setting lambda_eff_k =
    eta_k / E[A_k] and running the standard PSMC formulas.

    Parameters
    ----------
    n_undist : int
        Number of undistinguished haplotypes.
    n_states : int
        Number of HMM states (coalescence intervals).
    t : (n_states + 1,) time boundaries.
    eta : (n_states,) population sizes per interval.
    rho_base : float
        Per-base recombination rate (rho/2).

    Returns
    -------
    a_mat : (n_states, n_states) transition matrix.
    sigma : (n_states,) initial/stationary distribution.
    avg_t : (n_states,) average coalescence time per state.
    E_A : (n_states,) expected lineage count per interval (midpoint).
    """
    # Compute expected undistinguished lineages at boundaries
    Q = _lineage_rate_matrix(n_undist)
    p_bound = _propagate_lineages(Q, n_undist, t, eta)
    E_A_bound = _expected_lineages(p_bound, n_undist)
    E_A_mid = (E_A_bound[:-1] + E_A_bound[1:]) / 2.0

    # Effective lambda: lambda_eff_k = eta_k / E[A_k]
    # This makes the coalescence rate 1/lambda_eff = E[A]/eta, matching SMC++.
    lam_eff = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        lam_eff[k] = eta[k] / max(E_A_mid[k], 1.0)

    # --- PSMC formulas (adapted from psmc_internals.md / _numba.py) ---
    # Interval widths
    tau = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        tau[k] = t[k + 1] - t[k]

    # Alpha: survival probabilities
    # alpha_k = P(distinguished lineage not coalesced before t_k)
    alpha = np.empty(n_states + 1, dtype=np.float64)
    alpha[0] = 1.0
    cumsum_log = 0.0
    for k in range(n_states - 1):
        cumsum_log += -tau[k] / lam_eff[k]
        alpha[k + 1] = np.exp(cumsum_log)
    alpha[n_states] = 0.0

    alpha_safe = np.maximum(alpha, 1e-300)

    # Beta (weighted accumulated time)
    beta = np.empty(n_states, dtype=np.float64)
    beta[0] = 0.0
    for k in range(n_states - 1):
        beta[k + 1] = (beta[k]
                        + lam_eff[k] * (1.0 / alpha_safe[k + 1] - 1.0 / alpha_safe[k]))

    # q_aux
    q_aux = np.empty(n_states, dtype=np.float64)
    for ii in range(n_states):
        ak1_ii = alpha[ii] - alpha[ii + 1]
        q_aux[ii] = ak1_ii * (beta[ii] - lam_eff[ii] / alpha_safe[ii]) + tau[ii]

    # ak1, cpik, pik, sigma
    ak1 = alpha[:-1] - alpha[1:]

    # Cumulative time
    sum_t = np.empty(n_states, dtype=np.float64)
    sum_t[0] = 0.0
    for k in range(n_states - 1):
        sum_t[k + 1] = sum_t[k] + tau[k]

    # C_pi normalization
    C_pi = float(np.sum(lam_eff * ak1))
    if C_pi < 1e-300:
        C_pi = 1e-300
    C_sigma = 1.0 / (C_pi * max(rho_base, 1e-300)) + 0.5

    cpik = np.empty(n_states, dtype=np.float64)
    pik = np.empty(n_states, dtype=np.float64)
    sigma = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        cpik[k] = ak1[k] * (sum_t[k] + lam_eff[k]) - alpha[k + 1] * tau[k]
        pik[k] = cpik[k] / C_pi
        sigma[k] = (ak1[k] / (C_pi * max(rho_base, 1e-300)) + pik[k] / 2.0) / C_sigma

    # Average coalescence time per state (for CSFS computation)
    avg_t = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        denom = C_sigma * sigma[k]
        ratio = pik[k] / denom if denom > 0 else 2.0

        if 0.0 < ratio < 1.0:
            val = -np.log(max(1.0 - ratio, 1e-300)) / max(rho_base, 1e-300)
        else:
            val = np.nan

        if np.isnan(val) or val < sum_t[k] or val > sum_t[k] + tau[k]:
            if ak1[k] > 0:
                val = sum_t[k] + lam_eff[k] - tau[k] * alpha[k + 1] / ak1[k]
            else:
                val = sum_t[k]
        avg_t[k] = val

    # --- Build transition matrix q[k,l] ---
    cpik_safe = np.maximum(cpik, 1e-300)
    a_mat = np.zeros((n_states, n_states), dtype=np.float64)

    for k in range(n_states):
        if cpik[k] > 0:
            rk = ak1[k] / cpik_safe[k]
            # Lower triangle: l < k
            for ll in range(k):
                a_mat[k, ll] = rk * q_aux[ll]
            # Diagonal
            a_mat[k, k] = ((ak1[k] ** 2 * (beta[k] - lam_eff[k] / alpha_safe[k])
                             + 2.0 * lam_eff[k] * ak1[k]
                             - 2.0 * alpha[k + 1] * tau[k]) / cpik_safe[k])
            # Upper triangle: l > k
            if k < n_states - 1:
                rk2 = q_aux[k] / cpik_safe[k]
                for ll in range(k + 1, n_states):
                    a_mat[k, ll] = ak1[ll] * rk2
        else:
            a_mat[k, k] = 1.0

    # Convert q → transition probabilities
    tmp_p = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        denom = C_sigma * sigma[k]
        tmp_p[k] = pik[k] / denom if denom > 0 else 0.0

    for k in range(n_states):
        a_mat[k, :] *= tmp_p[k]
        a_mat[k, k] += 1.0 - tmp_p[k]

    return a_mat, sigma, avg_t, E_A_mid


# ---------------------------------------------------------------------------
# Conditioned SFS (CSFS) — emission matrix
# ---------------------------------------------------------------------------

def compute_csfs(
    n_undist: int,
    n_states: int,
    t: np.ndarray,
    eta: np.ndarray,
    theta: float,
    avg_t: np.ndarray,
    E_A_mid: np.ndarray,
    n_distinguished: int = 1,
    hidden_states: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Conditioned SFS emission matrix.

    Uses avg_t (from the PSMC-style transition computation) as the
    representative coalescence time for each state, providing a
    coalescence-density-weighted average rather than a midpoint.

    Includes both before-T and after-T contributions.

    Parameters
    ----------
    n_undist : int
        Number of undistinguished haplotypes.
    n_states : int
        Number of HMM states.
    t : (n_states+1,) time boundaries.
    eta : (n_states,) population sizes per interval.
    theta : float
        Scaled mutation rate per base (4N_0 * mu).
    avg_t : (n_states,) average coalescence time per state.
    E_A_mid : (n_states,) expected undist lineages per interval midpoint.

    Returns
    -------
    e : (n_obs, n_states) emission matrix.
    """
    if n_distinguished not in (1, 2):
        raise ValueError("Only 1 or 2 distinguished lineages are currently supported")

    if n_distinguished == 2:
        xi_two = _compute_onepop_raw_csfs(n_undist, t, eta, hidden_states=hidden_states)
        return _incorporate_theta_into_csfs(xi_two, theta)

    n_obs = observation_space_size(n_undist, 1)
    n_total = n_undist + 1
    xi = np.zeros((n_obs, n_states), dtype=np.float64)

    w_undist = _sfs_weights(n_undist)
    Q_undist = _lineage_rate_matrix(n_undist)
    aug_states, aug_index = _augmented_after_t_states(n_undist)
    Q_aug = _augmented_after_t_rate_matrix(n_undist, aug_states, aug_index)
    lam_eff = eta / np.maximum(E_A_mid, 1.0)

    # Precompute lineage distributions at each boundary
    p_bound = _propagate_lineages(Q_undist, n_undist, t, eta)

    # Precompute exact cumulative lineage occupation times at boundaries
    # R[k] = integral_0^{t_k} p(s) ds, shape (K+1, n_undist)
    R = _precompute_cumulative_occupation(Q_undist, n_undist, t, eta, p_bound)

    for k in range(n_states):
        xi_trunk = 0.0
        xi_undist = np.zeros(n_undist, dtype=np.float64)
        xi_after = np.zeros(n_obs, dtype=np.float64)
        quad_t, quad_w = _state_time_quadrature(
            t[k], t[k + 1], lam_eff[k], avg_t[k]
        )

        for T_k, weight_k in zip(quad_t, quad_w):
            R_at_T = _interpolate_occupation(
                R, t, T_k, Q_undist, eta, p_bound, n_states
            )

            xi_trunk_k = 0.0
            xi_undist_k = np.zeros(n_undist, dtype=np.float64)
            for j_idx in range(n_undist):
                j = j_idx + 1
                occ = R_at_T[j_idx]
                if occ < 1e-300:
                    continue
                xi_trunk_k += occ / (j + 1)
                xi_undist_k += occ * j / (j + 1) * w_undist[j_idx]

            p_at_T = _lineage_dist_at_time(T_k, t, eta, Q_undist, p_bound, n_states)
            p_aug = _initial_after_t_distribution(
                p_at_T, n_undist, aug_states, aug_index, w_undist
            )
            xi_after_k = np.zeros(n_obs, dtype=np.float64)
            _compute_after_t_csfs_exact(
                xi_after_k,
                p_aug,
                T_k,
                t,
                eta,
                Q_aug,
                aug_states,
                n_undist,
                n_total,
                n_states,
            )

            xi_trunk += weight_k * xi_trunk_k
            xi_undist += weight_k * xi_undist_k
            xi_after += weight_k * xi_after_k

        xi[n_undist + 1, k] = xi_trunk
        for b in range(1, n_undist + 1):
            xi[b, k] = xi_undist[b - 1]
        for obs in range(1, n_obs - 1):
            xi[obs, k] += xi_after[obs]

    return _incorporate_theta_into_csfs(xi, theta / 2.0)


def _incorporate_theta_into_csfs(csfs: np.ndarray, theta_rate: float) -> np.ndarray:
    """Convert branch-length CSFS into an emission probability matrix.

    This mirrors upstream SMC++'s exact ``1 - exp(-theta * tau)`` transform
    rather than the earlier first-order small-theta approximation.
    """
    e = np.zeros_like(csfs, dtype=np.float64)
    missing_idx = csfs.shape[0] - 1

    for k in range(csfs.shape[1]):
        tauh = float(csfs[:-1, k].sum())
        if tauh <= 0 or theta_rate <= 0:
            e[0, k] = 1.0
            e[missing_idx, k] = 1.0
            continue

        scale = -np.expm1(-theta_rate * tauh) / tauh
        e[1:missing_idx, k] = np.maximum(csfs[1:missing_idx, k] * scale, 0.0)
        variant_mass = float(e[1:missing_idx, k].sum())
        e[0, k] = max(1.0 - variant_mass, 1e-20)
        e[missing_idx, k] = 1.0

        col_sum = float(e[:-1, k].sum())
        if col_sum > 0:
            e[:-1, k] /= col_sum
        else:
            e[0, k] = 1.0

    return e


def _compute_onepop_raw_csfs(
    n_undist: int,
    t: np.ndarray,
    eta: np.ndarray,
    hidden_states: np.ndarray | None = None,
) -> np.ndarray:
    """Port of upstream one-pop raw SFS on the current hidden-state grid.

    Returns flattened branch-length CSFS in the upstream ``3 x (n+1)`` layout,
    plus the trailing missing row required by the HMM.
    """
    rate = _build_onepop_piecewise_rate(t, eta, hidden_states=hidden_states)
    K = len(rate["hidden_states"]) - 1
    n_obs = observation_space_size(n_undist, 2)
    out = np.zeros((n_obs, K), dtype=np.float64)

    raw = _compute_onepop_raw_csfs_tensor(n_undist, t, eta, hidden_states=hidden_states)
    for k in range(K):
        for a in range(3):
            for b in range(n_undist + 1):
                out[encode_obs(a, b, n_undist, 2), k] = raw[k, a, b]
        out[-1, k] = 1.0
    return out


def _compute_onepop_raw_csfs_tensor(
    n_undist: int,
    t: np.ndarray,
    eta: np.ndarray,
    hidden_states: np.ndarray | None = None,
) -> np.ndarray:
    """Compute upstream-style one-pop raw CSFS as ``(K, 3, n+1)``."""
    rate = _build_onepop_piecewise_rate(t, eta, hidden_states=hidden_states)
    K = len(rate["hidden_states"]) - 1
    n = n_undist
    cache = _onepop_matrix_cache(n)
    tjj_below = _tjj_double_integral_below_grid(n, rate, K)
    m0_below = tjj_below @ cache["M0"]
    m1_below = tjj_below @ cache["M1"]

    c_above = _tjj_double_integral_above_grid(n, rate, K)

    csfs = np.zeros((K, 3, n + 1), dtype=np.float64)
    for h in range(K):
        csfs[h, 0, 1:] = m0_below[h]
        csfs[h, 1, :] = m1_below[h]

        c0 = c_above[h].T
        c2 = np.flipud(c_above[h]).T
        tmp0 = np.sum(cache["X0"] * c0, axis=0)
        tmp2 = np.sum(cache["X2"] * c2, axis=0)
        csfs[h, 0, 1:] += tmp0 @ cache["Uinv_mp0"]
        csfs[h, 2, :n] = tmp2 @ cache["Uinv_mp2"]

    csfs = np.maximum(csfs, 0.0)
    csfs[:, 0, 0] = 0.0
    csfs[:, 2, n] = 0.0
    return csfs


def _coalescent_antiderivative(t: np.ndarray, eta: np.ndarray) -> np.ndarray:
    rrng = np.zeros(len(eta) + 1, dtype=np.float64)
    for k in range(len(eta)):
        rrng[k + 1] = rrng[k] + (t[k + 1] - t[k]) / max(eta[k], 1e-300)
    rrng[-1] = np.inf
    return rrng


def _build_onepop_piecewise_rate(
    t: np.ndarray,
    eta: np.ndarray,
    hidden_states: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build upstream-style piecewise-rate data with hidden states inserted."""
    if hidden_states is None:
        hidden_states = np.asarray(t, dtype=np.float64)
    else:
        hidden_states = np.asarray(hidden_states, dtype=np.float64)

    ts = list(np.asarray(t, dtype=np.float64))
    if not np.isinf(ts[-1]):
        ts.append(np.inf)
    ada = [1.0 / max(x, 1e-300) for x in np.r_[eta, eta[-1]]]

    for h in hidden_states:
        if np.isinf(h):
            continue
        inserted = False
        for x in ts:
            if abs(x - h) < 1e-8:
                inserted = True
                break
        if inserted:
            continue
        ip = int(np.searchsorted(ts, h, side="right") - 1)
        ip = min(max(ip, 0), len(ts) - 2)
        ts.insert(ip + 1, float(h))
        ada.insert(ip + 1, ada[ip])

    ts_arr = np.asarray(ts, dtype=np.float64)
    ada_arr = np.asarray(ada, dtype=np.float64)
    rrng = np.zeros(len(ts_arr), dtype=np.float64)
    for k in range(len(ada_arr)):
        rrng[k + 1] = rrng[k] + ada_arr[k] * (ts_arr[k + 1] - ts_arr[k])
    hs_indices = []
    for h in hidden_states:
        if np.isinf(h):
            hs_indices.append(len(ts_arr) - 1)
        else:
            hs_indices.append(int(np.argmin(np.abs(ts_arr - h))))
    return {
        "ts": ts_arr,
        "ada": ada_arr,
        "rrng": rrng,
        "hidden_states": hidden_states,
        "hs_indices": np.asarray(hs_indices, dtype=np.int64),
    }


def _expand_onepop_model_pieces(
    t: np.ndarray,
    eta: np.ndarray,
    n_pieces: int = SMCPP_INTERNAL_PIECES,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate upstream SMCModel.stepwise_values()/s for piecewise spline."""
    t = np.asarray(t, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if len(eta) == 0 or len(t) < 2:
        return t, eta
    knots = np.asarray(t[1:], dtype=np.float64)
    if len(knots) < 2 or knots[0] <= 0:
        return t, eta

    piece_points = np.logspace(np.log10(knots[0]), np.log10(knots[-1]), n_pieces)
    expanded_t = np.r_[0.0, piece_points]
    expanded_eta = np.empty(n_pieces, dtype=np.float64)
    for i, x in enumerate(piece_points):
        idx = np.searchsorted(knots, x, side="right") - 1
        idx = min(max(idx, 0), len(eta) - 1)
        expanded_eta[i] = eta[idx]
    return expanded_t, expanded_eta


def _single_integral_smcpp(rate: int, t0: float, t1: float, inv_eta: float, rrng0: float, log_coef: float) -> float:
    if rate == 0:
        return float(np.exp(log_coef) * (t1 - t0))
    ret = np.exp(-rate * rrng0 + log_coef)
    if np.isfinite(t1):
        ret *= -np.expm1(-rate * inv_eta * (t1 - t0))
    ret /= inv_eta * rate
    return float(max(ret, 0.0))


def _double_integral_below_helper_smcpp(rate: int, t0: float, t1: float, inv_eta: float, rrng0: float, log_denom: float) -> float:
    if inv_eta <= 0:
        return 0.0
    l1r = 1 + rate
    l1rinv = 1.0 / l1r
    diff = t1 - t0
    adadiff = inv_eta * diff
    if rate == 0:
        if np.isinf(t1):
            return float(np.exp(-rrng0 - log_denom) / inv_eta)
        return float(np.exp(-rrng0 - log_denom) * (1.0 - np.exp(-adadiff) * (1.0 + adadiff)) / inv_eta)
    if np.isinf(t1):
        return float(np.exp(-l1r * rrng0 - log_denom) * (1.0 - l1rinv) / (rate * inv_eta))
    return float(
        np.exp(-l1r * rrng0 - log_denom)
        * (np.expm1(-l1r * adadiff) * l1rinv - np.expm1(-adadiff))
        / (rate * inv_eta)
    )


def _double_integral_above_helper_smcpp(
    rate: int,
    lam: int,
    t0: float,
    t1: float,
    inv_eta: float,
    rrng0: float,
    log_coef: float,
) -> float:
    if inv_eta <= 0:
        return 0.0
    diff = t1 - t0
    adadiff = inv_eta * diff
    l1 = lam + 1
    if rate == 0:
        return float(
            np.exp(-l1 * rrng0 + log_coef)
            * (np.expm1(-l1 * adadiff) + l1 * adadiff)
            / (l1 * l1 * inv_eta)
        )
    if l1 == rate:
        if np.isinf(t1):
            return float(np.exp(-rate * rrng0 + log_coef) / (rate * rate * inv_eta))
        return float(
            np.exp(-rate * rrng0 + log_coef)
            * (1.0 - np.exp(-rate * adadiff) * (1.0 + rate * adadiff))
            / (rate * rate * inv_eta)
        )
    if np.isinf(t1):
        return float(np.exp(-l1 * rrng0 + log_coef) / (l1 * rate * inv_eta))
    if rate < l1:
        inner = np.expm1(-l1 * adadiff) / l1 + np.exp(-rate * adadiff) * (-np.expm1(-(l1 - rate) * adadiff) / (l1 - rate))
    else:
        inner = np.expm1(-l1 * adadiff) / l1 + np.exp(-l1 * adadiff) * (np.expm1(-(rate - l1) * adadiff) / (l1 - rate))
    return float(-np.exp(-l1 * rrng0 + log_coef) * inner / (rate * inv_eta))


def _tjj_double_integral_below_grid(
    n: int,
    rate: dict[str, np.ndarray],
    n_hidden_states: int,
) -> np.ndarray:
    ts = rate["ts"]
    ada = rate["ada"]
    rrng = rate["rrng"]
    hs_indices = rate["hs_indices"]
    K = len(ada)
    tgt = np.zeros((n_hidden_states, n + 1), dtype=np.float64)

    for h in range(n_hidden_states):
        h0 = hs_indices[h]
        h1 = hs_indices[h + 1]
        rh = rrng[h0]
        rh1 = rrng[h1]
        if abs(rh1 - rh) < 1e-15:
            continue
        log_denom = -rh
        if np.isfinite(rh1):
            log_denom += np.log(-np.expm1(-(rh1 - rh)))
        for m in range(h0, h1):
            rm = rrng[m]
            rm1 = rrng[m + 1]
            log_coef = -rm
            fac = 1.0
            if m < K - 1:
                fac = -np.expm1(-(rm1 - rm))
            for j in range(2, n + 3):
                coal_rate = j * (j - 1) // 2 - 1
                val = _double_integral_below_helper_smcpp(coal_rate, ts[m], ts[m + 1], ada[m], rrng[m], log_denom)
                for k in range(m):
                    val += fac * _single_integral_smcpp(coal_rate, ts[k], ts[k + 1], ada[k], rrng[k], log_coef - log_denom)
                tgt[h, j - 2] += max(val, 0.0)
    return tgt


def _tjj_double_integral_above_grid(
    n: int,
    rate: dict[str, np.ndarray],
    n_hidden_states: int,
) -> np.ndarray:
    ts = rate["ts"]
    ada = rate["ada"]
    rrng = rate["rrng"]
    hs_indices = rate["hs_indices"]
    K = len(ada)
    c = np.zeros((n_hidden_states, n + 1, n), dtype=np.float64)

    for jj in range(2, n + 3):
        lam = jj * (jj - 1) // 2 - 1
        rate_row = jj - 2
        for h in range(n_hidden_states):
            h0 = hs_indices[h]
            h1 = hs_indices[h + 1]
            rh = rrng[h0]
            rh1 = rrng[h1]
            if abs(rh1 - rh) < 1e-15:
                continue
            log_denom = -rh
            if np.isfinite(rh1):
                log_denom += np.log(-np.expm1(-(rh1 - rh)))
            for m in range(h0, h1):
                for j in range(2, n + 2):
                    coal_rate = j * (j - 1) // 2
                    val = _double_integral_above_helper_smcpp(coal_rate, lam, ts[m], ts[m + 1], ada[m], rrng[m], -log_denom)
                    rp = lam + 1 - coal_rate
                    rm = rrng[m]
                    rm1 = rrng[m + 1]
                    log_coef = -log_denom
                    if rp == 0:
                        fac = rm1 - rm
                    elif rp < 0:
                        if -rp * (rm1 - rm) > 20:
                            log_coef += -rp * rm1
                            fac = -1.0 / rp
                        else:
                            log_coef += -rp * rm
                            fac = -np.expm1(-rp * (rm1 - rm)) / rp
                    else:
                        if -rp * (rm - rm1) > 20:
                            log_coef += -rp * rm
                            fac = 1.0 / rp
                        else:
                            log_coef += -rp * rm1
                            fac = np.expm1(-rp * (rm - rm1)) / rp
                    for k in range(m + 1, K):
                        val += _single_integral_smcpp(coal_rate, ts[k], ts[k + 1], ada[k], rrng[k], log_coef) * fac
                    c[h, rate_row, j - 2] += max(val, 0.0)
    return c


@lru_cache(maxsize=None)
def _onepop_matrix_cache(n: int) -> dict[str, np.ndarray]:
    U, Uinv = _compute_moran_eigensystem_numeric(n)

    d_subtend_above = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    d_subtend_below = 2.0 / np.arange(2, n + 3, dtype=np.float64)
    lsp = np.arange(2, n + 3, dtype=np.float64)

    wnbj = np.zeros((n, n), dtype=np.float64)
    for b in range(1, n + 1):
        for j in range(2, n + 2):
            wnbj[b - 1, j - 2] = _calculate_wnbj(n + 1, b, j)

    p_dist = np.zeros((n + 1, n + 1), dtype=np.float64)
    for k in range(0, n + 1):
        for b in range(1, n - k + 2):
            p_dist[k, b - 1] = _pnkb_dist(n, k, b)

    p_undist = np.zeros((n + 1, n), dtype=np.float64)
    for k in range(1, n + 1):
        for b in range(1, n - k + 2):
            p_undist[k, b - 1] = _pnkb_undist(n, k, b)

    bc = _compute_below_coeffs(n)
    x0 = wnbj.T @ np.diag(1.0 - d_subtend_above) @ U[1:, :]
    x2 = wnbj.T @ np.diag(d_subtend_above) @ np.flipud(np.fliplr(U))[:n, :]
    m0 = bc @ np.diag(lsp) @ np.diag(1.0 - d_subtend_below) @ p_undist
    m1 = bc @ np.diag(lsp) @ np.diag(d_subtend_below) @ p_dist

    return {
        "X0": np.asarray(x0, dtype=np.float64),
        "X2": np.asarray(x2, dtype=np.float64),
        "M0": np.asarray(m0, dtype=np.float64),
        "M1": np.asarray(m1, dtype=np.float64),
        "Uinv_mp0": np.asarray(Uinv[:, 1:], dtype=np.float64),
        "Uinv_mp2": np.asarray(np.flipud(np.fliplr(Uinv))[:, :n], dtype=np.float64),
    }


def _modified_moran_rate_matrix_numeric(n: int, a: int = 0, na: int = 2) -> np.ndarray:
    m = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        sm = 0.0
        if i > 0:
            b = (na - a) * i + i * (n - i) / 2.0
            m[i, i - 1] = b
            sm += b
        if i < n:
            b = a * (n - i) + i * (n - i) / 2.0
            m[i, i + 1] = b
            sm += b
        m[i, i] = -sm
    return m


@lru_cache(maxsize=None)
def _compute_moran_eigensystem_numeric(n: int) -> tuple[np.ndarray, np.ndarray]:
    m = _modified_moran_rate_matrix_numeric(n)
    evals, evecs = np.linalg.eig(m)
    expected = np.array([-(k * (k - 1) / 2.0 - 1.0) for k in range(2, n + 3)], dtype=np.float64)
    order: list[int] = []
    remaining = set(range(len(evals)))
    for target in expected:
        idx = min(remaining, key=lambda i: abs(evals[i] - target))
        order.append(idx)
        remaining.remove(idx)
    U = np.real_if_close(evecs[:, order]).astype(np.float64)
    Uinv = np.linalg.inv(U)
    return U, Uinv


@lru_cache(maxsize=None)
def _compute_below_coeffs(n: int) -> np.ndarray:
    mlast = None
    for nn in range(2, n + 3):
        mnew = np.zeros((n + 1, nn - 1), dtype=np.float64)
        mnew[nn - 2, nn - 2] = 1.0
        if mlast is not None:
            for k in range(nn - 1, 1, -1):
                denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2)
                c1 = (nn + 1) * (nn - 2) / denom
                mnew[:, k - 2] = mlast[:, k - 2] * c1
            for k in range(nn - 1, 1, -1):
                denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2)
                c2 = (k + 2) * (k - 1) / denom
                mnew[:, k - 2] -= mnew[:, k - 1] * c2
        mlast = mnew
    return np.asarray(mlast, dtype=np.float64)


@lru_cache(maxsize=None)
def _calculate_wnbj(n: int, b: int, j: int) -> float:
    if j == 2:
        return 6.0 / (n + 1.0)
    if j == 3:
        if n == 2 * b:
            return 0.0
        return 30.0 * (n - 2 * b) / ((n + 1.0) * (n + 2.0))
    jj = j - 2
    c1 = -((1 + jj) * (3 + 2 * jj) * (n - jj)) / (jj * (2 * jj - 1) * (n + jj + 1))
    c2 = ((3 + 2 * jj) * (n - 2 * b)) / (jj * (n + jj + 1))
    return _calculate_wnbj(n, b, jj) * c1 + _calculate_wnbj(n, b, jj + 1) * c2


@lru_cache(maxsize=None)
def _pnkb_dist(n: int, m: int, l1: int) -> float:
    return float(l1 * comb(n + 2 - l1, m + 1, exact=False) / comb(n + 3, m + 3, exact=False))


@lru_cache(maxsize=None)
def _pnkb_undist(n: int, m: int, l3: int) -> float:
    return float(comb(n + 3 - l3, m + 2, exact=False) / comb(n + 3, m + 3, exact=False))


def _lift_csfs_to_two_distinguished(e_one: np.ndarray, n_undist: int) -> np.ndarray:
    """Lift the legacy one-distinguished emission grid into upstream's 3x(n+1) layout.

    This preserves the existing surrogate emission model while exposing the
    upstream one-pop observation space. Current surrogate ``a=0`` variant mass
    is split evenly across the folded ``a=0`` / ``a=2`` mirror pair.
    """
    K = e_one.shape[1]
    e_two = np.zeros((observation_space_size(n_undist, 2), K), dtype=np.float64)
    missing_idx_one = observation_space_size(n_undist, 1) - 1
    missing_idx_two = observation_space_size(n_undist, 2) - 1

    e_two[encode_obs(0, 0, n_undist, 2), :] += e_one[0, :]
    for b in range(1, n_undist + 1):
        mass = e_one[b, :]
        e_two[encode_obs(0, b, n_undist, 2), :] += 0.5 * mass
        e_two[encode_obs(2, n_undist - b, n_undist, 2), :] += 0.5 * mass
    for b in range(0, n_undist + 1):
        e_two[encode_obs(1, b, n_undist, 2), :] += e_one[n_undist + 1 + b, :]
    e_two[missing_idx_two, :] = e_one[missing_idx_one, :]
    return e_two


def _state_time_quadrature(
    t_left: float,
    t_right: float,
    lam_eff: float,
    fallback_mean: float,
    n_quad: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights for coalescence time within one interval.

    Uses a truncated exponential density with rate ``1 / lam_eff`` on the
    interval ``[t_left, t_right]``. Falls back to a point mass at
    ``fallback_mean`` when the interval is degenerate.
    """
    delta = t_right - t_left
    if delta <= 0:
        return np.array([fallback_mean], dtype=np.float64), np.array([1.0], dtype=np.float64)

    x, w = np.polynomial.legendre.leggauss(n_quad)
    nodes = t_left + 0.5 * (x + 1.0) * delta
    weights = 0.5 * delta * w

    rate = 0.0 if lam_eff <= 0 else 1.0 / lam_eff
    if rate > 1e-10:
        unnorm = rate * np.exp(-rate * (nodes - t_left))
        z = 1.0 - np.exp(-rate * delta)
        weights = weights * (unnorm / max(z, 1e-300))

    weights_sum = weights.sum()
    if weights_sum <= 0:
        return np.array([fallback_mean], dtype=np.float64), np.array([1.0], dtype=np.float64)
    weights /= weights_sum
    return nodes.astype(np.float64), weights.astype(np.float64)


def _augmented_after_t_states(
    n_undist: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """Enumerate valid post-coalescence states (j, d).

    ``j`` is the number of full-sample lineages after the distinguished
    lineage has coalesced. ``d`` is the number of sampled haplotypes subtended
    by the tagged lineage carrying the distinguished haplotype.
    """
    n_total = n_undist + 1
    states: list[tuple[int, int]] = []
    for j in range(1, n_undist + 1):
        min_d = n_total if j == 1 else 2
        max_d = n_total - j + 1
        for d in range(min_d, max_d + 1):
            states.append((j, d))
    return states, {state: i for i, state in enumerate(states)}


def _uniform_lineage_size_distribution(n_leaves: int, n_lineages: int) -> np.ndarray:
    """Distribution of descendant counts for a uniformly chosen lineage."""
    dist = np.zeros(n_leaves + 1, dtype=np.float64)
    if n_lineages <= 0 or n_leaves <= 0:
        return dist
    if n_lineages == 1:
        dist[n_leaves] = 1.0
        return dist
    w = _sfs_weights(n_leaves)
    dist[1:] = w[n_lineages - 1]
    return dist


def _augmented_after_t_rate_matrix(
    n_undist: int,
    states: list[tuple[int, int]],
    state_index: dict[tuple[int, int], int],
) -> np.ndarray:
    """Build the exact post-coalescence rate matrix on augmented states."""
    n_total = n_undist + 1
    Q = np.zeros((len(states), len(states)), dtype=np.float64)

    for idx, (j, d) in enumerate(states):
        if j == 1:
            continue

        rate_tagged = j - 1
        rate_untagged = (j - 1) * (j - 2) / 2.0
        total_rate = j * (j - 1) / 2.0
        Q[idx, idx] = -total_rate

        if rate_untagged > 0:
            Q[idx, state_index[(j - 1, d)]] += rate_untagged

        remaining = n_total - d
        size_dist = _uniform_lineage_size_distribution(remaining, j - 1)
        for b in range(1, remaining + 1):
            prob_b = size_dist[b]
            if prob_b <= 0:
                continue
            Q[idx, state_index[(j - 1, d + b)]] += rate_tagged * prob_b

    return Q


def _initial_after_t_distribution(
    p_at_T: np.ndarray,
    n_undist: int,
    states: list[tuple[int, int]],
    state_index: dict[tuple[int, int], int],
    w_undist: np.ndarray,
) -> np.ndarray:
    """Initial augmented-state distribution immediately after coalescence.

    Conditional on the distinguished lineage coalescing at time T, the
    undistinguished-lineage count distribution is tilted by the hazard j.
    """
    p_init = np.zeros(len(states), dtype=np.float64)
    weights = np.zeros_like(p_at_T, dtype=np.float64)

    for j_idx, p_j in enumerate(p_at_T):
        weights[j_idx] = (j_idx + 1) * p_j

    total_weight = weights.sum()
    if total_weight <= 0:
        return p_init
    weights /= total_weight

    for j_idx, p_j in enumerate(weights):
        j = j_idx + 1
        if p_j <= 0:
            continue
        if j == 1:
            p_init[state_index[(1, n_undist + 1)]] += p_j
            continue

        partner_sizes = w_undist[j_idx]
        for x in range(1, n_undist + 1):
            prob_x = partner_sizes[x - 1]
            if prob_x <= 0:
                continue
            p_init[state_index[(j, x + 1)]] += p_j * prob_x

    return p_init


def _accumulate_after_t_exact(
    xi_after: np.ndarray,
    occ: np.ndarray,
    states: list[tuple[int, int]],
    n_undist: int,
    n_total: int,
) -> None:
    """Accumulate exact post-coalescence branch-length contributions."""
    for idx, (j, d) in enumerate(states):
        weight = occ[idx]
        if weight <= 1e-300:
            continue

        tagged_obs = n_undist + 1 + (d - 1)
        if d < n_total and 0 <= tagged_obs < len(xi_after):
            xi_after[tagged_obs] += weight / j

        if j <= 1:
            continue

        remaining = n_total - d
        if remaining <= 0:
            continue

        size_dist = _uniform_lineage_size_distribution(remaining, j - 1)
        for b in range(1, remaining + 1):
            prob_b = size_dist[b]
            if prob_b <= 0:
                continue
            xi_after[b] += weight * (j - 1) / j * prob_b


def _compute_after_t_csfs_exact(
    xi_after: np.ndarray,
    p_aug_init: np.ndarray,
    T: float,
    t: np.ndarray,
    eta: np.ndarray,
    Q_aug: np.ndarray,
    states: list[tuple[int, int]],
    n_undist: int,
    n_total: int,
    n_states: int,
) -> None:
    """Compute exact after-T CSFS contributions via augmented state tracking."""
    k_start = 0
    for k in range(n_states):
        if t[k] <= T <= t[k + 1]:
            k_start = k
            break

    p_cur = p_aug_init.copy()
    dim = Q_aug.shape[0]

    intervals: list[tuple[float, float]] = []
    remaining = t[k_start + 1] - T
    if remaining > 0:
        intervals.append((remaining, eta[k_start]))
    for kk in range(k_start + 1, n_states):
        tau = t[kk + 1] - t[kk]
        if tau > 0:
            intervals.append((tau, eta[kk]))

    for dt, eta_k in intervals:
        if dt <= 0 or eta_k <= 0:
            continue

        scaled = Q_aug / eta_k
        block = np.zeros((2 * dim, 2 * dim), dtype=np.float64)
        block[:dim, :dim] = scaled
        block[:dim, dim:] = np.eye(dim)
        block_exp = expm(block * dt)
        P = block_exp[:dim, :dim]
        integral_matrix = block_exp[:dim, dim:]

        occ = p_cur @ integral_matrix
        _accumulate_after_t_exact(xi_after, occ, states, n_undist, n_total)
        p_cur = p_cur @ P


def _precompute_cumulative_occupation(
    Q: np.ndarray,
    n: int,
    t: np.ndarray,
    eta: np.ndarray,
    p_bound: np.ndarray,
) -> np.ndarray:
    """Precompute exact cumulative lineage occupation at each boundary.

    R[k, j-1] = integral_0^{t_k} P(j lineages at s) ds

    Uses the block-matrix exponential trick for exact integration.
    """
    K = len(eta)
    R = np.zeros((K + 1, n), dtype=np.float64)

    for k in range(K):
        tau_k = t[k + 1] - t[k]
        if tau_k <= 0 or eta[k] <= 0:
            R[k + 1] = R[k].copy()
            continue

        # Block matrix trick: expm([[Q/eta, I], [0, 0]] * tau)
        # Upper-right block gives integral_0^tau expm(Q*s/eta) ds
        Q_scaled = Q * tau_k / eta[k]
        block = np.zeros((2 * n, 2 * n), dtype=np.float64)
        block[:n, :n] = Q_scaled / tau_k  # Q/eta
        block[:n, n:] = np.eye(n)
        block_exp = expm(block * tau_k)
        integral_matrix = block_exp[:n, n:]  # integral_0^tau expm(Qs/eta) ds

        # Occupation in this interval: p_bound[k] @ integral_matrix
        occ_k = p_bound[k] @ integral_matrix
        R[k + 1] = R[k] + occ_k

    return R


def _interpolate_occupation(
    R: np.ndarray,
    t: np.ndarray,
    T: float,
    Q: np.ndarray,
    eta: np.ndarray,
    p_bound: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Interpolate cumulative occupation to arbitrary time T."""
    # Find interval containing T
    for k in range(n_states):
        if t[k] <= T <= t[k + 1]:
            if T == t[k]:
                return R[k].copy()
            ds = T - t[k]
            if ds <= 0 or eta[k] <= 0:
                return R[k].copy()

            n = Q.shape[0]
            scaled = Q / eta[k]
            block = np.zeros((2 * n, 2 * n), dtype=np.float64)
            block[:n, :n] = scaled
            block[:n, n:] = np.eye(n)
            block_exp = expm(block * ds)
            integral_matrix = block_exp[:n, n:]
            occ = p_bound[k] @ integral_matrix
            return R[k] + occ
    return R[n_states].copy()


def _lineage_dist_at_time(
    T: float,
    t: np.ndarray,
    eta: np.ndarray,
    Q: np.ndarray,
    p_bound: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Compute lineage distribution at arbitrary time T."""
    for k in range(n_states):
        if t[k] <= T <= t[k + 1]:
            ds = T - t[k]
            if ds <= 0:
                return p_bound[k].copy()
            if eta[k] > 0:
                P_partial = expm(Q * ds / eta[k])
                return p_bound[k] @ P_partial
            return p_bound[k].copy()
    return p_bound[n_states].copy()


def _compute_after_T_csfs(
    xi_after: np.ndarray,
    p_at_T: np.ndarray,
    T: float,
    t: np.ndarray,
    eta: np.ndarray,
    Q: np.ndarray,
    w_full: np.ndarray,
    n_undist: int,
    n_total: int,
    n_states: int,
) -> None:
    """Compute after-T CSFS contributions."""
    # Find interval containing T
    k_start = 0
    for k in range(n_states):
        if t[k] <= T <= t[k + 1]:
            k_start = k
            break

    p_cur = p_at_T.copy()

    # Partial interval (T to t_{k_start+1})
    remaining = t[k_start + 1] - T
    if remaining > 0 and eta[k_start] > 0:
        P_part = expm(Q * remaining / eta[k_start])
        p_next = p_cur @ P_part
        _accumulate_after_T(xi_after, p_cur, p_next, remaining,
                            w_full, n_undist, n_total)
        p_cur = p_next

    # Full remaining intervals
    for kk in range(k_start + 1, n_states):
        tau = t[kk + 1] - t[kk]
        if tau > 0 and eta[kk] > 0:
            P_kk = expm(Q * tau / eta[kk])
            p_next = p_cur @ P_kk
        else:
            p_next = p_cur.copy()
        _accumulate_after_T(xi_after, p_cur, p_next, tau,
                            w_full, n_undist, n_total)
        p_cur = p_next


def _accumulate_after_T(
    xi_after: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
    dt: float,
    w_full: np.ndarray,
    n_undist: int,
    n_total: int,
) -> None:
    """Accumulate after-T CSFS contributions from one sub-interval."""
    if dt <= 0:
        return

    p_avg = (p_start + p_end) / 2.0

    for j_idx in range(n_undist):
        j = j_idx + 1
        pj = p_avg[j_idx]
        if pj < 1e-300 or j < 2:
            continue

        j_full = min(j, n_total)
        for d in range(1, n_total):
            if d > n_total - j_full + 1:
                break
            wd = w_full[j_full - 1, d - 1]
            if wd < 1e-300:
                continue

            branch_weight = pj * dt * wd
            p_in = d / n_total

            b_a1 = d - 1
            if 0 <= b_a1 < n_undist:
                xi_after[n_undist + 1 + b_a1] += branch_weight * p_in

            b_a0 = d
            if 1 <= b_a0 <= n_undist:
                xi_after[b_a0] += branch_weight * (1.0 - p_in)


# ---------------------------------------------------------------------------
# HMM parameters container
# ---------------------------------------------------------------------------

@dataclass
class SmcppHmmParams:
    """Container for SMC++ HMM parameters."""

    a: np.ndarray        # (K, K) transition matrix per base
    e: np.ndarray        # (n_obs, K) emission matrix
    a0: np.ndarray       # (K,) initial distribution
    n_obs: int           # number of observation symbols
    n_undist: int        # undistinguished sample size
    n_distinguished: int = 1
    avg_t: np.ndarray | None = None
    theta: float = 0.0
    polarization_error: float = 0.0
    observation_scale: float = 1.0


@dataclass
class SmcppExpectationStats:
    gamma0: np.ndarray
    gamma_sums: dict[int, np.ndarray]
    xisum: np.ndarray
    log_likelihood: float


def compute_hmm_params(
    eta: np.ndarray,
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    n_distinguished: int = 1,
    hidden_states: np.ndarray | None = None,
    polarization_error: float = 0.0,
    observation_scale: float = 1.0,
) -> SmcppHmmParams:
    """Compute full HMM parameters from population sizes."""
    if n_distinguished == 2:
        if hidden_states is None:
            hidden_states = np.r_[t, np.inf]
        t_internal, eta_internal = _expand_onepop_model_pieces(t, eta)
        a_mat, sigma = _compute_onepop_transition_and_pi(t_internal, eta_internal, hidden_states, rho_base)
        rate = _build_onepop_piecewise_rate(t_internal, eta_internal, hidden_states=hidden_states)
        avg_t = _average_coal_times_onepop(rate)
        e = compute_csfs(
            n_undist,
            a_mat.shape[0],
            t_internal,
            eta_internal,
            theta,
            np.empty(a_mat.shape[0], dtype=np.float64),
            np.empty(a_mat.shape[0], dtype=np.float64),
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
        )
        return SmcppHmmParams(
            a=a_mat,
            e=e,
            a0=sigma.copy(),
            n_obs=e.shape[0],
            n_undist=n_undist,
            n_distinguished=n_distinguished,
            avg_t=avg_t,
            theta=theta,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )

    K = len(eta)
    a_mat, sigma, avg_t, E_A_mid = _compute_psmc_style_transition(
        n_undist, K, t, eta, rho_base,
    )
    e = compute_csfs(
        n_undist, K, t, eta, theta, avg_t, E_A_mid,
        n_distinguished=n_distinguished,
        hidden_states=hidden_states,
    )
    return SmcppHmmParams(
        a=a_mat, e=e, a0=sigma.copy(),
        n_obs=e.shape[0], n_undist=n_undist, n_distinguished=n_distinguished,
        avg_t=avg_t,
        theta=theta,
        polarization_error=polarization_error,
        observation_scale=observation_scale,
    )


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def observation_space_size(n_undist: int, n_distinguished: int = 1) -> int:
    """Return the number of observation symbols including the missing symbol."""
    if n_distinguished == 1:
        return 2 * (n_undist + 1) + 1
    if n_distinguished == 2:
        return 3 * (n_undist + 1) + 1
    raise ValueError("Only 1 or 2 distinguished lineages are currently supported")


def encode_obs(a: int, b: int, n_undist: int, n_distinguished: int = 1) -> int:
    """Encode ``(a, b)`` as an observation index.

    ``n_distinguished=1`` preserves the existing surrogate-model layout.
    ``n_distinguished=2`` matches upstream one-pop SMC++'s ``3 x (n+1)``
    observation grid, with rows ordered by ``a in {0,1,2}``.
    """
    if a < 0 or b < 0:
        return observation_space_size(n_undist, n_distinguished) - 1
    if n_distinguished == 1:
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return b
        return n_undist + 1 + b
    if n_distinguished == 2:
        if not 0 <= a <= 2:
            raise ValueError("For n_distinguished=2, distinguished count a must be 0, 1, or 2")
        if not 0 <= b <= n_undist:
            raise ValueError("Undistinguished derived count b out of range")
        return a * (n_undist + 1) + b
    raise ValueError("Only 1 or 2 distinguished lineages are currently supported")


def _unpack_observation(
    obs: tuple[int, ...],
    default_n_undist: int,
) -> tuple[int, int, int, int]:
    if len(obs) == 3:
        span, a_obs, b_obs = obs
        return int(span), int(a_obs), int(b_obs), int(default_n_undist if b_obs >= 0 else 0)
    if len(obs) == 4:
        span, a_obs, b_obs, n_obs = obs
        return int(span), int(a_obs), int(b_obs), int(n_obs)
    raise ValueError("SMC++ observations must be (span, a, b) or (span, a, b, n)")


def _hypergeom_pmf(k: int, successes: int, failures: int, draws: int) -> float:
    if k < 0 or k > successes or draws - k < 0 or draws - k > failures:
        return 0.0
    denom = comb(successes + failures, draws)
    if denom <= 0:
        return 0.0
    return float(comb(successes, k) * comb(failures, draws - k) / denom)


def _fold_onepop_key(a_obs: int, b_obs: int, n_obs: int) -> tuple[int, int, int]:
    return 2 - a_obs, n_obs - b_obs, n_obs


def _emission_vector_onepop(
    hp: SmcppHmmParams,
    a_obs: int,
    b_obs: int,
    n_obs: int,
) -> np.ndarray:
    """Compute upstream-style one-pop emission probabilities for an observed key."""
    if a_obs < 0:
        return np.ones(hp.a.shape[0], dtype=np.float64)
    if n_obs <= 0:
        if hp.avg_t is None:
            raise ValueError("avg_t is required for reduced one-pop observations")
        log_e = -2.0 * hp.observation_scale * hp.theta * hp.avg_t
        if a_obs % 2 == 0:
            return np.exp(log_e)
        return -np.expm1(log_e)

    weights: dict[tuple[int, int], float] = {}
    full_n = hp.n_undist
    polarization_error = hp.polarization_error
    for b_full in range(b_obs, full_n + b_obs - n_obs + 1):
        prob = _hypergeom_pmf(b_obs, b_full, full_n - b_full, n_obs)
        if prob <= 0:
            continue
        a_full = a_obs
        if a_full == 2 and b_full == full_n:
            a_full, b_full = 0, 0
        # Upstream recodes fully derived monomorphic rows to (0, 0) and then
        # treats that monomorphic key directly rather than splitting it across
        # the folded (2, n) mirror under polarization error.
        if a_full == 0 and b_full == 0:
            weights[(0, 0)] = weights.get((0, 0), 0.0) + prob
            continue
        weights[(a_full, b_full)] = weights.get((a_full, b_full), 0.0) + (1.0 - polarization_error) * prob
        a_fold, b_fold, _ = _fold_onepop_key(a_full, b_full, full_n)
        weights[(a_fold, b_fold)] = weights.get((a_fold, b_fold), 0.0) + polarization_error * prob

    weights = {k: v for k, v in weights.items() if v > 0}
    total = sum(weights.values())
    if total <= 0:
        return np.full(hp.a.shape[0], 1e-20, dtype=np.float64)

    out = np.zeros(hp.a.shape[0], dtype=np.float64)
    for (a_full, b_full), weight in weights.items():
        out += (weight / total) * hp.e[encode_obs(a_full, b_full, hp.n_undist, 2)]
    return np.maximum(out, 1e-20)


def _thin_onepop_observations(
    observations: list[tuple[int, ...]],
    thinning: int,
    default_n_undist: int,
) -> list[tuple[int, int, int, int]]:
    if thinning <= 0:
        return [(*_unpack_observation(obs, default_n_undist),)[:4] for obs in observations]
    i = 0
    out: list[tuple[int, int, int, int]] = []
    for obs in observations:
        span, a_obs, b_obs, n_obs = _unpack_observation(obs, default_n_undist)
        sa = max(a_obs, 0)
        thin_a = 0 if sa == 2 else a_obs
        while span > 0:
            if i < thinning and i + span >= thinning:
                left = thinning - i
                if left > 1:
                    out.append((left - 1, thin_a, 0, 0))
                if sa == 2 and b_obs == n_obs:
                    out.append((1, 0, n_obs, n_obs))
                else:
                    out.append((1, a_obs, b_obs, n_obs))
                span -= left
                i = 0
            else:
                out.append((span, thin_a, 0, 0))
                i += span
                break
    return out


def _process_onepop_bin(rows: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    max_sample_size = -2
    best = rows[0]
    for row in rows:
        span, a_obs, _b_obs, n_obs = row
        if span <= 0:
            continue
        sample_size = n_obs + (2 if a_obs >= 0 else 0)
        seg = max(0, a_obs)
        if sample_size > max_sample_size:
            best = row
            max_sample_size = sample_size
        if max_sample_size == 2 and seg == 1:
            best = row
    return 1, best[1], best[2], best[3]


def _bin_onepop_observations(
    observations: list[tuple[int, ...]],
    width: int,
    default_n_undist: int,
) -> list[tuple[int, int, int, int]]:
    rows = [_unpack_observation(obs, default_n_undist) for obs in observations]
    ret: list[tuple[int, int, int, int]] = []
    i = 0
    j = 0
    seen = 0
    while j < len(rows):
        span = rows[j][0]
        if seen + span > width:
            left = width - seen
            curr = rows[j]
            rows[j] = (left, curr[1], curr[2], curr[3])
            ret.append(_process_onepop_bin(rows[i : j + 1]))
            rows[j] = (span - left, curr[1], curr[2], curr[3])
            seen = 0
            i = j
        else:
            j += 1
            seen += span
    if j > 0:
        ret.append(_process_onepop_bin(rows[i:j]))
    return ret


def _recode_monomorphic_onepop(
    observations: list[tuple[int, ...]],
    default_n_undist: int,
) -> list[tuple[int, int, int, int]]:
    out = []
    for obs in observations:
        span, a_obs, b_obs, n_obs = _unpack_observation(obs, default_n_undist)
        if a_obs == 2 and b_obs == n_obs:
            out.append((span, 0, 0, n_obs))
        else:
            out.append((span, a_obs, b_obs, n_obs))
    return out


def _compress_onepop_observations(
    observations: list[tuple[int, ...]],
    default_n_undist: int,
) -> list[tuple[int, int, int, int]]:
    rows = [_unpack_observation(obs, default_n_undist) for obs in observations]
    if not rows:
        return []
    out = [list(rows[0])]
    for row in rows[1:]:
        if tuple(row[1:]) == tuple(out[-1][1:]):
            out[-1][0] += row[0]
        else:
            out.append(list(row))
    return [tuple(row) for row in out]


def _preprocess_onepop_records(records: list[dict], n_undist: int) -> list[dict]:
    thinning = int(500 * np.log(2 + n_undist))
    processed = []
    for rec in records:
        rec_n_undist = int(rec.get("n_undist", n_undist))
        # Upstream's BaseAnalysis always passes one-pop contigs through
        # BreakLongSpans first, which prepends a single missing row even when
        # no long missing spans are present. That one-base offset changes the
        # thinning/binning phase on small fixtures, so keep it here too.
        raw_obs = [(1, -1, 0, 0)] + [
            _unpack_observation(obs, rec_n_undist) for obs in rec["observations"]
        ]
        obs = _thin_onepop_observations(raw_obs, thinning, rec_n_undist)
        obs = _bin_onepop_observations(obs, 100, rec_n_undist)
        obs = _recode_monomorphic_onepop(obs, rec_n_undist)
        obs = _compress_onepop_observations(obs, rec_n_undist)
        variable = any((a_obs > 0) or (b_obs > 0) for _, a_obs, b_obs, _ in obs)
        if variable:
            processed.append({**rec, "observations": obs})
    if not processed:
        raise RuntimeError("No contigs have mutation data after one-pop preprocessing.")
    return processed


def _onepop_matrix_exp(delta: float, inv_eta: float, rho: float) -> np.ndarray:
    c_eta = inv_eta * delta
    c_rho = rho * delta
    sq = np.sqrt(max(4.0 * c_eta * c_eta + c_rho * c_rho, 0.0))
    if sq <= 1e-300:
        return np.eye(3, dtype=np.float64)
    s = np.sinh(0.5 * sq) / sq
    c = np.cosh(0.5 * sq)
    e = np.exp(-c_eta - c_rho / 2.0)
    q = np.zeros((3, 3), dtype=np.float64)
    q[0, 0] = e * (c + (2.0 * c_eta - c_rho) * s)
    q[0, 1] = 2.0 * e * c_rho * s
    q[0, 2] = 1.0 - q[0, 0] - q[0, 1]
    q[1, 0] = 2.0 * e * c_eta * s
    q[1, 1] = e * (c - (2.0 * c_eta - c_rho) * s)
    q[1, 2] = 1.0 - q[1, 0] - q[1, 1]
    q[2, 2] = 1.0
    return q


def _rate_R(rate: dict[str, np.ndarray], x: float) -> float:
    ts = rate["ts"]
    ada = rate["ada"]
    rrng = rate["rrng"]
    ip = int(np.searchsorted(ts, x, side="right") - 1)
    ip = min(max(ip, 0), len(ada) - 1)
    return float(rrng[ip] + ada[ip] * (x - ts[ip]))


def _rate_R_integral(rate: dict[str, np.ndarray], a: float, b: float, log_denom: float) -> float:
    ts = rate["ts"]
    ada = rate["ada"]
    ret = 0.0
    ip_a = int(np.searchsorted(ts, a, side="right") - 1)
    ip_b = len(ts) - 2 if np.isinf(b) else int(np.searchsorted(ts, b, side="right") - 1)
    for i in range(ip_a, ip_b + 1):
        left = max(a, ts[i])
        right = min(b, ts[i + 1])
        diff = right - left
        rleft = _rate_R(rate, left)
        val = np.exp(-(rleft + log_denom))
        if ada[i] > 0.0:
            if np.isfinite(diff):
                val *= -np.expm1(-diff * ada[i])
            val /= ada[i]
        else:
            val *= diff
        ret += val
    return float(ret)


def _average_coal_times_onepop(rate: dict[str, np.ndarray]) -> np.ndarray:
    hs = rate["hidden_states"]
    hs_indices = rate["hs_indices"]
    rrng = rate["rrng"]
    out = []
    for i in range(1, len(hs)):
        if rrng[hs_indices[i - 1]] == rrng[hs_indices[i]]:
            out.append(np.nan)
            continue
        log_denom = -rrng[hs_indices[i - 1]]
        inf = np.isinf(hs[i])
        if not inf:
            log_denom += np.log(-np.expm1(-(rrng[hs_indices[i]] - rrng[hs_indices[i - 1]])))
        x = hs[i - 1] * np.exp(-(rrng[hs_indices[i - 1]] + log_denom))
        x += _rate_R_integral(rate, hs[i - 1], hs[i], log_denom)
        if not inf:
            x -= hs[i] * np.exp(-(rrng[hs_indices[i]] + log_denom))
        out.append(float(x))
    return np.asarray(out, dtype=np.float64)


def _compute_onepop_transition_and_pi(
    t: np.ndarray,
    eta: np.ndarray,
    hidden_states: np.ndarray,
    rho_base: float,
) -> tuple[np.ndarray, np.ndarray]:
    rate = _build_onepop_piecewise_rate(t, eta, hidden_states=hidden_states)
    hs = rate["hidden_states"]
    hs_indices = rate["hs_indices"]
    ts = rate["ts"]
    ada = rate["ada"]

    m = len(hs) - 1
    pi = np.zeros(m, dtype=np.float64)
    for i in range(m - 1):
        pi[i] = np.exp(-_rate_R(rate, hs[i])) - np.exp(-_rate_R(rate, hs[i + 1]))
    pi[m - 1] = np.exp(-_rate_R(rate, hs[m - 1]))
    pi = np.maximum(pi, 1e-20)
    pi /= pi.sum()

    expms = [np.eye(3, dtype=np.float64) for _ in range(len(ts))]
    prefix = [np.eye(3, dtype=np.float64)]
    for i in range(len(ts) - 1):
        if np.isinf(ts[i + 1]):
            q = np.zeros((3, 3), dtype=np.float64)
            q[:, 2] = 1.0
        else:
            q = _onepop_matrix_exp(ts[i + 1] - ts[i], ada[i], rho_base)
        expms[i + 1 if i + 1 < len(ts) else i] = q
        prefix.append(prefix[-1] @ q)

    avg = _average_coal_times_onepop(rate)
    avc_ip = [min(max(int(np.searchsorted(ts, x, side="right") - 1), 0), len(ada) - 1) for x in avg]
    phi = np.zeros((m, m), dtype=np.float64)
    expm_diff = np.array([prefix[hs_indices[k]][0, 2] - prefix[hs_indices[k - 1]][0, 2] for k in range(1, m - 1 + 1)], dtype=np.float64)

    for j in range(1, m + 1):
        if j - 1 > 0:
            phi[j - 1, : j - 1] = expm_diff[: j - 1]
        rct = avg[j - 1]
        rct_ip = avc_ip[j - 1]
        a_mat = np.eye(3, dtype=np.float64)
        for ell in range(hs_indices[j - 1], rct_ip):
            a_mat = a_mat @ expms[ell]
        delta = rct - ts[rct_ip]
        a_mat = a_mat @ _onepop_matrix_exp(delta, ada[rct_ip], rho_base)
        b_mat = prefix[hs_indices[j - 1]] @ a_mat
        # Match upstream's shifted interval indexing in HJTransition:
        # the current interval contributes its full hazard here.
        rj = ada[rct_ip] * (ts[rct_ip + 1] - ts[rct_ip])
        for jj in range(rct_ip + 2, hs_indices[j]):
            rj += ada[jj] * (ts[jj + 1] - ts[jj])
        p_float = b_mat[0, 1] * np.exp(-rj)
        rjk1 = 0.0
        for k in range(j + 1, m + 1):
            inc = 0.0
            for jj in range(hs_indices[k - 1], hs_indices[k]):
                inc += ada[jj] * (ts[jj + 1] - ts[jj])
            p_coal = np.exp(-rjk1)
            rjk1 += inc
            if np.isfinite(inc):
                p_coal *= -np.expm1(-inc)
            phi[j - 1, k - 1] += p_float * p_coal
        phi[j - 1, j - 1] = 1.0 - phi[j - 1].sum()

    phi = np.maximum(phi, 1e-20)
    beta = 1e-5
    phi = phi * (1.0 - beta) + beta / (m + 1)
    return phi, pi


# ---------------------------------------------------------------------------
# Span-based forward/backward (full matrix, O(K^2) per step)
# ---------------------------------------------------------------------------

def _forward_spans(
    hp: SmcppHmmParams,
    observations: list[tuple[int, int, int]],
) -> tuple[list[np.ndarray], list[float]]:
    """Scaled forward algorithm for span-encoded observations.

    Each input row is interpreted as one observation repeated ``span`` times,
    matching upstream SMC++ run-length semantics.
    """
    K = hp.a.shape[0]
    a = hp.a
    e = hp.e
    n_undist = hp.n_undist

    f = hp.a0.copy()
    f_list = []
    log_c_list = []
    power_cache: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    for obs in observations:
        span, a_obs, b_obs, n_obs = _unpack_observation(obs, n_undist)
        if span <= 0:
            continue
        if hp.n_distinguished == 2:
            em_direct = _emission_vector_onepop(hp, a_obs, b_obs, n_obs)
        else:
            obs_idx_direct = encode_obs(a_obs, b_obs, n_undist, hp.n_distinguished)
            em_direct = e[obs_idx_direct]
        if span == 1:
            f_new = em_direct * (a.T @ f)
            f_new = np.real_if_close(f_new).astype(np.float64)
            f_new = np.maximum(f_new, 0.0)
            s_raw = f_new.sum()
            if s_raw > 0:
                f = f_new / s_raw
                log_c = np.log(max(s_raw, 1e-300))
            else:
                f = np.ones(K, dtype=np.float64) / K
                log_c = np.log(1e-300)
            f = np.maximum(f, 1e-10)
            f_list.append(f.copy())
            log_c_list.append(float(log_c))
            continue
        if hp.n_distinguished == 2:
            obs_key = (a_obs, b_obs, n_obs)
            if obs_key not in power_cache:
                M_obs = a * em_direct[np.newaxis, :]
                eigvals, V = np.linalg.eig(M_obs.T)
                scale = max(float(np.max(np.abs(eigvals))), 1e-300)
                power_cache[obs_key] = (np.real(eigvals), np.real(V), np.real(np.linalg.inv(V)), scale)
            eigvals, V, V_inv, scale = power_cache[obs_key]
        else:
            if obs_idx_direct not in power_cache:
                M_obs = a * em_direct[np.newaxis, :]
                eigvals, V = np.linalg.eig(M_obs.T)
                scale = max(float(np.max(np.abs(eigvals))), 1e-300)
                power_cache[obs_idx_direct] = (np.real(eigvals), np.real(V), np.real(np.linalg.inv(V)), scale)
            eigvals, V, V_inv, scale = power_cache[obs_idx_direct]
        eigpow = (eigvals / scale) ** span
        f_new = np.real_if_close(V @ (eigpow * (V_inv @ f))).astype(np.float64)
        f_new = np.maximum(f_new, 0.0)
        s_raw = f_new.sum()
        if s_raw > 0:
            f = f_new / s_raw
            log_c = np.log(max(s_raw, 1e-300)) + span * np.log(scale)
        else:
            f = np.ones(K, dtype=np.float64) / K
            log_c = np.log(1e-300)
        f = np.maximum(f, 1e-10)
        f_list.append(f.copy())
        log_c_list.append(float(log_c))

    return f_list, log_c_list


def _backward_spans(
    hp: SmcppHmmParams,
    observations: list[tuple[int, int, int]],
    log_c_list: list[float],
) -> list[np.ndarray]:
    """Scaled backward algorithm for span-encoded observations."""
    K = hp.a.shape[0]
    a = hp.a
    e = hp.e
    n_undist = hp.n_undist

    n_obs_total = len(observations)
    b_list = [np.zeros(K, dtype=np.float64) for _ in range(n_obs_total)]
    power_cache: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    s_idx = len(log_c_list) - 1
    b = np.ones(K, dtype=np.float64)
    b_list[n_obs_total - 1] = b.copy()

    for i in range(n_obs_total - 2, -1, -1):
        span_next, a_next, b_next_obs, n_next = _unpack_observation(observations[i + 1], n_undist)
        if hp.n_distinguished == 2:
            obs_key_next = (a_next, b_next_obs, n_next)
            if obs_key_next not in power_cache:
                em = _emission_vector_onepop(hp, a_next, b_next_obs, n_next)
                M_obs = a * em[np.newaxis, :]
                eigvals, V = np.linalg.eig(M_obs)
                scale = max(float(np.max(np.abs(eigvals))), 1e-300)
                power_cache[obs_key_next] = (np.real(eigvals), np.real(V), np.real(np.linalg.inv(V)), scale)
            eigvals, V, V_inv, scale = power_cache[obs_key_next]
        else:
            obs_idx_next = encode_obs(a_next, b_next_obs, n_undist, hp.n_distinguished)
            if obs_idx_next not in power_cache:
                M_obs = a * e[obs_idx_next][np.newaxis, :]
                eigvals, V = np.linalg.eig(M_obs)
                scale = max(float(np.max(np.abs(eigvals))), 1e-300)
                power_cache[obs_idx_next] = (np.real(eigvals), np.real(V), np.real(np.linalg.inv(V)), scale)
            eigvals, V, V_inv, scale = power_cache[obs_idx_next]
        eigpow = (eigvals / scale) ** span_next
        b = np.real_if_close(V @ (eigpow * (V_inv @ b))).astype(np.float64)
        b = np.maximum(b, 0.0)
        s_idx -= 1
        if s_idx >= 0:
            b /= np.exp(log_c_list[s_idx])

        b_list[i] = b.copy()

    return b_list


def _log_likelihood_spans(log_c_list: list[float]) -> float:
    """Compute log-likelihood from scaling factors."""
    return float(np.sum(log_c_list))


def _obs_eigensystem(
    hp: SmcppHmmParams,
    obs_idx: int | tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(obs_idx, tuple):
        em = _emission_vector_onepop(hp, obs_idx[0], obs_idx[1], obs_idx[2])
    else:
        em = hp.e[obs_idx]
    m_obs = hp.a * em[np.newaxis, :]
    eigvals, p = np.linalg.eig(m_obs.T)
    pinv = np.linalg.inv(p)
    return eigvals, p, pinv


def _span_geometric_sum(eigvals: np.ndarray, span: int) -> np.ndarray:
    q = np.zeros((len(eigvals), len(eigvals)), dtype=np.float64)
    for i, d1 in enumerate(eigvals):
        q[i, i] = span * (d1 ** (span - 1))
        for j in range(i + 1, len(eigvals)):
            d2 = eigvals[j]
            if abs(d1 - d2) < 1e-14:
                val = span * (d1 ** (span - 1))
            else:
                val = (d1 ** span - d2 ** span) / (d1 - d2)
            q[i, j] = val
            q[j, i] = val
    return q


def _expectation_stats_unit_observations(
    hp: SmcppHmmParams,
    observations: list[tuple[int, int, int, int]],
) -> SmcppExpectationStats:
    """Exact scaled forward-backward statistics for unit-span observations."""
    k = hp.a.shape[0]
    e_vectors: list[np.ndarray] = []
    obs_keys: list[int | tuple[int, int, int]] = []
    alpha_prev_list: list[np.ndarray] = []
    alpha_hat_list: list[np.ndarray] = []
    c_list: list[float] = []

    prev = np.asarray(hp.a0, dtype=np.float64)
    for span, a_obs, b_obs, n_obs in observations:
        if span != 1:
            raise ValueError("Unit-observation expectation path requires span=1 rows")
        if hp.n_distinguished == 2:
            obs_key = (a_obs, b_obs, n_obs)
            e_obs = _emission_vector_onepop(hp, a_obs, b_obs, n_obs)
        else:
            obs_key = encode_obs(a_obs, b_obs, hp.n_undist, hp.n_distinguished)
            e_obs = hp.e[obs_key]
        alpha_prev_list.append(prev.copy())
        alpha = e_obs * (hp.a.T @ prev)
        c = max(float(alpha.sum()), 1e-300)
        prev = alpha / c
        obs_keys.append(obs_key)
        e_vectors.append(e_obs)
        alpha_hat_list.append(prev.copy())
        c_list.append(c)

    gamma_sums: dict[object, np.ndarray] = {}
    xisum = np.zeros((k, k), dtype=np.float64)
    beta_hat = np.ones(k, dtype=np.float64)
    gamma0 = np.zeros(k, dtype=np.float64)

    for i in range(len(obs_keys) - 1, -1, -1):
        obs_key = obs_keys[i]
        gamma = np.maximum(alpha_hat_list[i] * beta_hat, 1e-300)
        gamma /= max(gamma.sum(), 1e-300)
        gamma_sums.setdefault(obs_key, np.zeros(k, dtype=np.float64))
        gamma_sums[obs_key] += gamma

        xi = alpha_prev_list[i][:, np.newaxis] * hp.a * (e_vectors[i] * beta_hat)[np.newaxis, :]
        xi /= max(c_list[i], 1e-300)
        xi /= max(xi.sum(), 1e-300)
        xisum += np.maximum(xi, 1e-300)
        if i == 0:
            gamma0 = xi.sum(axis=1)
        beta_hat = hp.a @ (e_vectors[i] * beta_hat)
        beta_hat /= max(c_list[i], 1e-300)

    log_likelihood = float(np.sum(np.log(np.asarray(c_list, dtype=np.float64))))
    return SmcppExpectationStats(
        gamma0=np.maximum(gamma0, 1e-20),
        gamma_sums=gamma_sums,
        xisum=np.maximum(xisum, 1e-20),
        log_likelihood=log_likelihood,
    )


def _expectation_stats_spans(
    hp: SmcppHmmParams,
    observations: list[tuple[int, int, int]],
) -> SmcppExpectationStats:
    """Port of the upstream HMM E-step sufficient for one-pop native Q()."""
    k = hp.a.shape[0]
    if k == 1:
        gamma_sums: dict[object, np.ndarray] = {}
        total_span = 0
        for obs in observations:
            span, a_obs, b_obs, n_obs = _unpack_observation(obs, hp.n_undist)
            if span <= 0:
                continue
            if hp.n_distinguished == 2:
                obs_idx = (a_obs, b_obs, n_obs)
            else:
                obs_idx = encode_obs(a_obs, b_obs, hp.n_undist, hp.n_distinguished)
            gamma_sums.setdefault(obs_idx, np.zeros(1, dtype=np.float64))
            gamma_sums[obs_idx][0] += float(span)
            total_span += int(span)
        _, log_c_list = _forward_spans(hp, observations)
        return SmcppExpectationStats(
            gamma0=np.ones(1, dtype=np.float64),
            gamma_sums=gamma_sums,
            xisum=np.full((1, 1), float(total_span), dtype=np.float64),
            log_likelihood=_log_likelihood_spans(log_c_list),
        )

    f_list, log_c_list = _forward_spans(hp, observations)
    log_likelihood = _log_likelihood_spans(log_c_list)

    gamma_sums: dict[object, np.ndarray] = {}
    xisum = np.zeros((k, k), dtype=np.float64)
    beta = np.ones(k, dtype=np.float64)
    eig_cache: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    for ell in range(len(observations) - 1, -1, -1):
        span, a_obs, b_obs, n_obs = _unpack_observation(observations[ell], hp.n_undist)
        if hp.n_distinguished == 2:
            obs_idx = (a_obs, b_obs, n_obs)
            e_obs = _emission_vector_onepop(hp, a_obs, b_obs, n_obs)
        else:
            obs_idx = encode_obs(a_obs, b_obs, hp.n_undist, hp.n_distinguished)
            e_obs = hp.e[obs_idx]
        alpha_prev = hp.a0 if ell == 0 else f_list[ell - 1]
        gamma_sums.setdefault(obs_idx, np.zeros(k, dtype=np.float64))

        if span > 1:
            eigvals, p, pinv, scale = eig_cache.get(obs_idx, (None, None, None, None))
            if eigvals is None:
                eigvals_c, p_c, pinv_c = _obs_eigensystem(hp, obs_idx)
                scale = max(float(np.max(np.abs(eigvals_c))), 1e-300)
                eigvals = np.real(eigvals_c)
                p = np.real(p_c)
                pinv = np.real(pinv_c)
                eig_cache[obs_idx] = (eigvals, p, pinv, scale)
            d_scaled = eigvals / scale
            log_p = np.log(scale) * (span - 1)
            q = _span_geometric_sum(d_scaled, span)
            q_r = (pinv @ (alpha_prev[:, np.newaxis] * beta[np.newaxis, :]) @ p).astype(np.float64)
            q_r = q_r * q

            diag_vals = np.abs(
                np.real_if_close(np.diag(p @ (np.diag(eigvals) @ q_r) @ pinv)).astype(np.float64)
            )
            diag_vals = np.maximum(diag_vals, 1e-300)
            v_log = np.log(diag_vals) - log_c_list[ell] + log_p
            v_m = float(np.max(v_log))
            log_c = np.log(span) - v_m - np.log(np.sum(np.exp(v_log - v_m)))
            v = np.exp(v_log + log_c)

            xis_vals = np.abs(np.real_if_close(p @ q_r @ pinv @ np.diag(e_obs)).astype(np.float64))
            xis_vals = np.maximum(xis_vals, 1e-300)
            xis = np.exp(np.log(xis_vals) - log_c_list[ell] + log_p + log_c)

            beta_vals = np.real_if_close(
                pinv.T @ ((d_scaled ** span) * (p.T @ beta))
            ).astype(np.float64)
            beta_vals = np.maximum(np.abs(beta_vals), 1e-300)
            log_beta = np.log(beta_vals) + log_p + log_c + np.log(scale)
            beta_m = float(np.max(log_beta))
            beta = np.exp(log_beta - beta_m)
        else:
            alpha_curr = f_list[ell]
            v = alpha_curr * beta
            p_norm = max(v.sum(), 1e-300)
            v /= p_norm
            xis = alpha_prev[:, np.newaxis] * (beta * e_obs)[np.newaxis, :]
            xis /= max(np.exp(log_c_list[ell]) * p_norm, 1e-300)
            beta = hp.a @ (e_obs * beta)

        xisum += np.maximum(xis, 1e-20)
        gamma_sums[obs_idx] += np.maximum(v, 0.0)
        beta_sum = beta.sum()
        if beta_sum > 0:
            beta /= beta_sum

    gamma0 = np.maximum(hp.a0 * beta, 1e-20)
    return SmcppExpectationStats(
        gamma0=gamma0,
        gamma_sums=gamma_sums,
        xisum=np.maximum(xisum * hp.a, 1e-20),
        log_likelihood=log_likelihood,
    )


def _maybe_expand_onepop_observation_spans(
    observations: list[tuple[int, ...]],
    hp: SmcppHmmParams,
    max_expanded_length: int = 50000,
) -> list[tuple[int, int, int, int]]:
    """Preserve upstream one-pop run-length semantics."""
    del max_expanded_length
    return [_unpack_observation(obs, hp.n_undist) for obs in observations]


def _collect_expectation_stats(
    hp: SmcppHmmParams,
    records: list[dict],
) -> SmcppExpectationStats:
    k = hp.a.shape[0]
    gamma0 = np.zeros(k, dtype=np.float64)
    xisum = np.zeros((k, k), dtype=np.float64)
    gamma_sums: dict[int, np.ndarray] = {}
    total_ll = 0.0

    for rec in records:
        obs = _maybe_expand_onepop_observation_spans(rec["observations"], hp)
        if len(obs) == 0:
            continue
        stats = _expectation_stats_spans(hp, obs)
        gamma0 += stats.gamma0
        xisum += stats.xisum
        total_ll += stats.log_likelihood
        for obs_idx, val in stats.gamma_sums.items():
            gamma_sums.setdefault(obs_idx, np.zeros(k, dtype=np.float64))
            gamma_sums[obs_idx] += val

    return SmcppExpectationStats(
        gamma0=np.maximum(gamma0, 1e-20),
        gamma_sums=gamma_sums,
        xisum=np.maximum(xisum, 1e-20),
        log_likelihood=total_ll,
    )


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _objective(
    log_eta: np.ndarray,
    records: list[dict],
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    n_distinguished: int = 1,
    hidden_states: np.ndarray | None = None,
    polarization_error: float = 0.0,
    observation_scale: float = 1.0,
) -> float:
    """Negative log-likelihood + roughness penalty."""
    eta = np.exp(log_eta)
    hp = compute_hmm_params(
        eta,
        n_undist,
        t,
        theta,
        rho_base,
        n_distinguished=n_distinguished,
        hidden_states=hidden_states,
        polarization_error=polarization_error,
        observation_scale=observation_scale,
    )

    total_ll = 0.0
    for rec in records:
        obs = _maybe_expand_onepop_observation_spans(rec["observations"], hp)
        if len(obs) == 0:
            continue
        _, s_list = _forward_spans(hp, obs)
        total_ll += _log_likelihood_spans(s_list)

    penalty = 0.0
    if len(log_eta) >= 3:
        d2 = np.diff(log_eta, n=2)
        penalty = float(np.dot(d2, d2))

    return -total_ll + regularization * penalty


def _m_step_objective(
    log_eta: np.ndarray,
    stats: SmcppExpectationStats,
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    n_distinguished: int = 1,
    hidden_states: np.ndarray | None = None,
    polarization_error: float = 0.0,
    observation_scale: float = 1.0,
) -> float:
    """Negative upstream-style Q() objective for one M-step."""
    eta = np.exp(log_eta)
    hp = compute_hmm_params(
        eta,
        n_undist,
        t,
        theta,
        rho_base,
        n_distinguished=n_distinguished,
        hidden_states=hidden_states,
        polarization_error=polarization_error,
        observation_scale=observation_scale,
    )

    q = float(np.dot(stats.gamma0, np.log(np.maximum(hp.a0, 1e-300))))
    for obs_idx, gamma_sum in stats.gamma_sums.items():
        if isinstance(obs_idx, tuple):
            em = _emission_vector_onepop(hp, obs_idx[0], obs_idx[1], obs_idx[2])
        else:
            em = hp.e[obs_idx]
        q += float(np.dot(gamma_sum, np.log(np.maximum(em, 1e-300))))
    q += float(np.sum(stats.xisum * np.log(np.maximum(hp.a, 1e-300))))

    penalty = 0.0
    if len(log_eta) >= 3:
        d2 = np.diff(log_eta, n=2)
        penalty = float(np.dot(d2, d2))

    return -q + regularization * penalty


def _set_log_eta_coords(
    log_eta: np.ndarray,
    coords: list[int],
    values: np.ndarray,
) -> np.ndarray:
    updated = np.array(log_eta, copy=True, dtype=np.float64)
    updated[np.asarray(coords, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    return updated


def _coordinate_optimize_m_step(
    log_eta: np.ndarray,
    coords: list[int],
    stats: SmcppExpectationStats,
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    n_distinguished: int,
    hidden_states: np.ndarray | None,
    polarization_error: float,
    observation_scale: float,
) -> tuple[np.ndarray, dict]:
    x0 = np.asarray(log_eta[coords], dtype=np.float64)
    lower = np.maximum(x0 - 3.0, np.log(1e-4))
    upper = np.minimum(x0 + 3.0, np.log(1e4))

    if len(coords) == 1:
        coord = coords[0]

        def objective_scalar(x: float) -> float:
            trial = np.array(log_eta, copy=True, dtype=np.float64)
            trial[coord] = x
            return _m_step_objective(
                trial,
                stats,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished=n_distinguished,
                hidden_states=hidden_states,
                polarization_error=polarization_error,
                observation_scale=observation_scale,
            )

        res = minimize_scalar(
            objective_scalar,
            bounds=(float(lower[0]), float(upper[0])),
            method="bounded",
            options={"xatol": SMCPP_MSTEP_XATOL},
        )
        updated = _set_log_eta_coords(log_eta, coords, np.asarray([res.x], dtype=np.float64))
        return updated, {
            "coords": coords,
            "success": bool(res.success),
            "fun": float(res.fun),
            "nfev": int(res.nfev),
        }

    def objective_vector(x: np.ndarray) -> float:
        trial = _set_log_eta_coords(log_eta, coords, np.asarray(x, dtype=np.float64))
        return _m_step_objective(
            trial,
            stats,
            n_undist,
            t,
            theta,
            rho_base,
            regularization,
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )

    res = minimize(
        objective_vector,
        x0,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=False)),
    )
    updated = _set_log_eta_coords(log_eta, coords, np.asarray(res.x, dtype=np.float64))
    return updated, {
        "coords": coords,
        "success": bool(res.success),
        "fun": float(res.fun),
        "nfev": int(res.nfev),
    }


def _scale_optimize_m_step(
    log_eta: np.ndarray,
    stats: SmcppExpectationStats,
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    n_distinguished: int,
    hidden_states: np.ndarray | None,
    polarization_error: float,
    observation_scale: float,
) -> tuple[np.ndarray, dict]:
    x0 = np.asarray(log_eta, dtype=np.float64)
    x0_eval = np.array(x0, copy=True, dtype=np.float64)
    current = np.array(x0, copy=True, dtype=np.float64)

    def objective_scalar(shift: float) -> float:
        nonlocal current
        trial = x0_eval + shift
        current = np.array(trial, copy=True, dtype=np.float64)
        return _m_step_objective(
            trial,
            stats,
            n_undist,
            t,
            theta,
            rho_base,
            regularization,
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )

    res = minimize_scalar(
        objective_scalar,
        bounds=(-1.0, 1.0),
        method="bounded",
    )
    # Match upstream ScaleOptimizer's aliased final assignment. The bounded
    # search evaluates against a fixed starting vector, but the final
    # ``analysis.model[:] = x0 + res.x`` uses a view that already contains the
    # last trial point, effectively applying ``res.x`` on top of that state.
    updated = current + float(res.x)
    return updated, {
        "success": bool(res.success),
        "fun": float(res.fun),
        "nfev": int(res.nfev),
        "shift": float(res.x),
        "last_trial_shift": float(current[0] - x0_eval[0]),
    }


def _optimize_log_eta_em(
    log_eta_init: np.ndarray,
    records: list[dict],
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    max_iterations: int,
    n_distinguished: int = 1,
    hidden_states: np.ndarray | None = None,
    polarization_error: float = 0.0,
    observation_scale: float = 1.0,
    single_coordinate: bool = True,
) -> tuple[np.ndarray, dict]:
    log_eta = np.asarray(log_eta_init, dtype=np.float64).copy()
    history: list[dict[str, float | int | list[dict]]] = []
    success = True
    message = "max iterations reached"
    prev_log_likelihood: float | None = None

    for iteration in range(max_iterations):
        hp = compute_hmm_params(
            np.exp(log_eta),
            n_undist,
            t,
            theta,
            rho_base,
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )
        stats = _collect_expectation_stats(hp, records)
        improvement = None if prev_log_likelihood is None else (
            (prev_log_likelihood - float(stats.log_likelihood)) / prev_log_likelihood
            if prev_log_likelihood != 0.0
            else None
        )
        q_before = -_m_step_objective(
            log_eta,
            stats,
            n_undist,
            t,
            theta,
            rho_base,
            regularization,
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )

        coord_groups = [[k] for k in range(len(log_eta) - 1, -1, -1)] if single_coordinate else [list(range(len(log_eta)))]
        m_steps = []
        prev_log_eta = log_eta.copy()
        scale_step = None
        if n_distinguished == 2:
            log_eta, scale_step = _scale_optimize_m_step(
                log_eta,
                stats,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished,
                hidden_states,
                polarization_error,
                observation_scale,
            )
        for coords in coord_groups:
            log_eta, step_info = _coordinate_optimize_m_step(
                log_eta,
                coords,
                stats,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished,
                hidden_states,
                polarization_error,
                observation_scale,
            )
            m_steps.append(step_info)

        q_after = -_m_step_objective(
            log_eta,
            stats,
            n_undist,
            t,
            theta,
            rho_base,
            regularization,
            n_distinguished=n_distinguished,
            hidden_states=hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )
        history.append({
            "iteration": iteration,
            "log_eta_before": prev_log_eta.tolist(),
            "eta_before": np.exp(prev_log_eta).tolist(),
            "log_likelihood": float(stats.log_likelihood),
            "log_likelihood_improvement": None if improvement is None else float(improvement),
            "q_before": float(q_before),
            "q_after": float(q_after),
            "max_abs_delta": float(np.max(np.abs(log_eta - prev_log_eta))),
            "log_eta_after": log_eta.tolist(),
            "eta_after": np.exp(log_eta).tolist(),
            "scale_step": scale_step,
            "m_steps": m_steps,
        })
        if improvement is not None:
            if improvement < 0:
                message = "log likelihood decreased"
            elif improvement < 1e-4:
                message = "log likelihood improvement below tolerance"
                prev_log_likelihood = float(stats.log_likelihood)
                break
        prev_log_likelihood = float(stats.log_likelihood)

    return log_eta, {
        "success": success,
        "message": message,
        "n_iterations": int(len(history)),
        "history": history,
    }


def _watterson_theta(records: list[dict], default_n_undist: int | None = None) -> float:
    """Approximate upstream SMC++ Watterson estimator.

    Upstream computes the denominator row-wise using
    ``log(sample_size) + 0.5 / sample_size + 0.57721``, where the sample size
    is the observed undistinguished total plus one distinguished block.
    """
    segregating = 0.0
    denom = 0.0
    for rec in records:
        n_undist = rec.get("n_undist")
        if n_undist is None:
            n_undist = default_n_undist
        if n_undist is None:
            n_undist = 0
            for _, a_obs, b_obs in rec["observations"]:
                if b_obs >= 0:
                    n_undist = max(n_undist, b_obs)
        for obs in rec["observations"]:
            span, a_obs, b_obs, row_n_undist = _unpack_observation(obs, n_undist)
            if span <= 0:
                continue
            sample_size = 0
            if a_obs >= 0:
                sample_size += 1
            if b_obs >= 0:
                sample_size += row_n_undist
            if a_obs >= 1 or b_obs > 0:
                segregating += span
            if sample_size > 0:
                denom += span * (
                    np.log(sample_size) + 0.5 / sample_size + 0.57721
                )
    if denom <= 0:
        return 1e-8
    return max(segregating / denom, 1e-8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SmcppResult:
    """Results from an SMC++ run."""

    time: np.ndarray
    time_boundaries: np.ndarray
    eta: np.ndarray
    ne: np.ndarray
    time_years: np.ndarray
    theta: float = 0.0
    rho: float = 0.0
    n0: float = 0.0
    log_likelihood: float = 0.0
    n_undist: int = 0
    n_intervals: int = 0
    regularization: float = 0.0
    optimization_result: dict = field(default_factory=dict)


def _resolve_upstream_smcpp_python() -> str | None:
    """Locate the controlled Python environment that can run upstream SMC++."""
    env_python = os.environ.get("SMCKIT_SMCPP_PYTHON")
    if env_python:
        return env_python
    repo_python = Path(__file__).resolve().parents[3] / "vendor/smcpp/.venv/bin/python"
    if repo_python.exists():
        return str(repo_python)
    default_python = "/tmp/smcpp39/bin/python"
    if os.path.exists(default_python):
        return default_python
    return None


def _serialize_smcpp_record(
    record: dict,
    path: Path,
    n_undist: int,
    n_distinguished: int,
    pids: list[str] | None,
) -> None:
    """Write one record back to upstream SMC++ .smc format."""
    if pids:
        header_pids = list(pids)
    else:
        header_pids = ["pop1"]

    distinguished_samples = record.get("distinguished_samples")
    if not distinguished_samples:
        distinguished_samples = [[["distinguished", i] for i in range(max(n_distinguished, 2))]]
    undistinguished_samples = record.get("undistinguished_samples")
    if not undistinguished_samples:
        undistinguished_samples = [[["undistinguished", i] for i in range(n_undist)]]

    header = {
        "version": "smckit",
        "pids": header_pids,
        "dist": distinguished_samples,
        "undist": undistinguished_samples,
    }

    with path.open("wt", encoding="utf-8") as fh:
        fh.write("# SMC++ ")
        json.dump(header, fh, separators=(",", ":"))
        fh.write("\n")
        for obs in record["observations"]:
            span, a, b, n_obs = _unpack_observation(obs, n_undist)
            if a < 0 or b < 0:
                fh.write(f"{int(span)} -1 0 0\n")
            else:
                fh.write(f"{int(span)} {int(a)} {int(b)} {int(n_obs)}\n")


def _run_upstream_smcpp(
    data: SmcData,
    *,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    generation_time: float,
    regularization: float,
    max_iterations: int,
    seed: int | None,
) -> dict:
    """Run the real upstream SMC++ model in a controlled side environment."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_result.json"),
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "generation_time": float(generation_time),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": int(max_iterations),
            "seed": None if seed is None else int(seed),
            "trace": True,
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ backend failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        return json.loads((tmpdir_path / "smcpp_result.json").read_text(encoding="utf-8"))


def _run_upstream_smcpp_fixed_model_stats(
    data: SmcData,
    *,
    model_vector: np.ndarray | None = None,
    model_dict: dict | None = None,
    alpha: float,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict:
    """Evaluate upstream one-pop E-step statistics at a fixed model."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-fixed-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "fixed_model_stats",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_fixed_stats.json"),
            "alpha": float(alpha),
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
        }
        if model_dict is not None:
            payload["model_dict"] = model_dict
        elif model_vector is not None:
            payload["model_vector"] = np.asarray(model_vector, dtype=float).tolist()
        else:
            raise ValueError("Provide either model_vector or model_dict")
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ fixed-model stats failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return json.loads((tmpdir_path / "smcpp_fixed_stats.json").read_text(encoding="utf-8"))


def _run_upstream_smcpp_fixed_stats_q_compare(
    data: SmcData,
    *,
    start_model_vector: np.ndarray,
    candidate_models: dict[str, np.ndarray],
    alpha: float,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict[str, float]:
    """Evaluate upstream ``analysis.Q()`` on frozen stats for candidate models."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-q-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "fixed_stats_q_compare",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_fixed_q.json"),
            "alpha": float(alpha),
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
            "start_model_vector": np.asarray(start_model_vector, dtype=float).tolist(),
            "candidate_models": {
                str(name): np.asarray(vec, dtype=float).tolist()
                for name, vec in candidate_models.items()
            },
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ fixed-stats Q compare failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        result = json.loads((tmpdir_path / "smcpp_fixed_q.json").read_text(encoding="utf-8"))
        return {str(name): float(value) for name, value in result.items()}


def _run_upstream_smcpp_fixed_stats_one_mstep(
    data: SmcData,
    *,
    start_model_vector: np.ndarray,
    alpha: float,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict:
    """Run exactly one upstream frozen-stats M-step from a fixed start model."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-onemstep-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "fixed_stats_one_mstep",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_one_mstep.json"),
            "alpha": float(alpha),
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
            "start_model_vector": np.asarray(start_model_vector, dtype=float).tolist(),
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ fixed-stats one-M-step failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
        )
        return json.loads((tmpdir_path / "smcpp_one_mstep.json").read_text(encoding="utf-8"))


def _run_upstream_smcpp_onepop_initialization_summary(
    data: SmcData,
    *,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict:
    """Return upstream one-pop prefit and main-loop initialization models."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-init-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "onepop_initialization_summary",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_init.json"),
            "alpha": 1.0,
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ initialization summary failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return json.loads((tmpdir_path / "smcpp_init.json").read_text(encoding="utf-8"))


def _run_upstream_smcpp_onepop_prefit_fixed_stats_q_compare(
    data: SmcData,
    *,
    start_model_vector: np.ndarray,
    candidate_models: dict[str, np.ndarray],
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict[str, float]:
    """Evaluate upstream one-pop prefit fixed-stats Q on candidate models."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-prefitq-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "onepop_prefit_fixed_stats_q_compare",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_prefit_q.json"),
            "alpha": 1.0,
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
            "start_model_vector": np.asarray(start_model_vector, dtype=float).tolist(),
            "candidate_models": {
                str(name): np.asarray(vec, dtype=float).tolist()
                for name, vec in candidate_models.items()
            },
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ prefit fixed-stats Q compare failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        result = json.loads((tmpdir_path / "smcpp_prefit_q.json").read_text(encoding="utf-8"))
        return {str(name): float(value) for name, value in result.items()}


def _run_upstream_smcpp_onepop_prefit_fixed_model_stats(
    data: SmcData,
    *,
    model_vector: np.ndarray,
    n_intervals: int,
    mu: float,
    recombination_rate: float,
    regularization: float,
    seed: int | None,
) -> dict:
    """Return upstream one-pop prefit-stage fixed-model stats."""
    python_exe = _resolve_upstream_smcpp_python()
    if python_exe is None:
        raise RuntimeError(
            "Upstream SMC++ backend is unavailable. Set SMCKIT_SMCPP_PYTHON "
            "or create /tmp/smcpp39/bin/python."
        )

    runner = Path(__file__).with_name("_smcpp_upstream_runner.py")
    records = data.uns["records"]
    n_undist = int(data.uns["n_undist"])
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    pids = data.uns.get("pids")

    with tempfile.TemporaryDirectory(prefix="smckit-smcpp-prefitstats-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_paths: list[str] = []
        for i, record in enumerate(records):
            record_path = tmpdir_path / f"record_{i}.smc"
            _serialize_smcpp_record(record, record_path, n_undist, n_distinguished, pids)
            input_paths.append(str(record_path))

        payload = {
            "mode": "onepop_prefit_fixed_model_stats",
            "input_paths": input_paths,
            "output_json": str(tmpdir_path / "smcpp_prefit_stats.json"),
            "alpha": 1.0,
            "mu": float(mu),
            "recombination_rate": float(recombination_rate),
            "n_intervals": int(n_intervals),
            "regularization": float(regularization),
            "max_iterations": 2,
            "seed": None if seed is None else int(seed),
            "model_vector": np.asarray(model_vector, dtype=float).tolist(),
        }
        payload_path = tmpdir_path / "runner_payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))

        proc = subprocess.run(
            [python_exe, str(runner), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream SMC++ prefit fixed-model stats failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return json.loads((tmpdir_path / "smcpp_prefit_stats.json").read_text(encoding="utf-8"))


def _native_fixed_stats_q_compare(
    stats: SmcppExpectationStats,
    *,
    candidate_models: dict[str, np.ndarray],
    n_undist: int,
    t: np.ndarray,
    theta: float,
    rho_base: float,
    regularization: float,
    hidden_states: np.ndarray,
    observation_scale: float,
    polarization_error: float = 0.5,
) -> dict[str, float]:
    """Evaluate native fixed-stats Q for candidate models."""
    out = {}
    for name, log_eta in candidate_models.items():
        out[str(name)] = float(
            -_m_step_objective(
                np.asarray(log_eta, dtype=np.float64),
                stats,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished=2,
                hidden_states=hidden_states,
                polarization_error=polarization_error,
                observation_scale=observation_scale,
            )
        )
    return out


def _smcpp_native(
    data: SmcData,
    n_intervals: int = 32,
    max_t: float = 15.0,
    alpha: float = 0.1,
    mu: float = 1.25e-8,
    recombination_rate: float = 1e-8,
    generation_time: float = 25.0,
    regularization: float = 1.0,
    max_iterations: int = 100,
    seed: int | None = None,
    implementation_requested: str = "native",
) -> SmcData:
    """Run SMC++ demographic inference.

    Parameters
    ----------
    data : SmcData
        Input data from ``smckit.io.read_smcpp_input()``.
        Must have ``data.uns["records"]`` with observation lists and
        ``data.uns["n_undist"]`` with the undistinguished sample size.
    n_intervals : int
        Number of time intervals (HMM states).
    max_t : float
        Maximum coalescent time (units of 2N_0).
    alpha : float
        Time interval spacing parameter.
    mu : float
        Per-base per-generation mutation rate.
    recombination_rate : float
        Per-base per-generation recombination rate.
    generation_time : float
        Generation time in years.
    regularization : float
        Roughness penalty weight (higher = smoother).
    max_iterations : int
        Maximum L-BFGS iterations.
    seed : int, optional
        Random seed for initialization.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["smcpp"]``.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    records = data.uns["records"]
    n_undist = data.uns["n_undist"]
    n_distinguished = int(data.uns.get("n_distinguished", 2))
    polarization_error = 0.5 if n_distinguished == 2 else 0.0
    K_requested = n_intervals
    K = K_requested
    preprocessing: dict[str, int | bool] = {
        "applied": False,
        "thinning": 0,
        "bin_width": 1,
        "n_input_records": len(records),
        "n_fit_records": len(records),
    }
    observation_scale = 1.0

    if n_distinguished == 2:
        n0_fixed = 0.5e-4 / mu
        theta = 2.0 * n0_fixed * mu
        rho = 2.0 * n0_fixed * recombination_rate
        rho_base = rho
        theta_hat_init = _watterson_theta(records, n_undist)
        init_eta0 = max(theta_hat_init / max(2.0 * mu * n0_fixed, 1e-300), 1e-3)
    else:
        theta = _watterson_theta(records, n_undist)
        rho = recombination_rate / max(mu, 1e-20) * theta
        rho_base = rho / 2.0
        init_eta0 = 1.0

    t, native_hidden_states = _native_time_grid(K, max_t, alpha, n_distinguished, init_eta0=init_eta0)
    initial_hidden_states = None if native_hidden_states is None else native_hidden_states.copy()
    prefit_t = t.copy()
    if n_distinguished == 2:
        initial_hidden_states = _balanced_hidden_states_constant(2 * K_requested, eta0=init_eta0)
        native_hidden_states = initial_hidden_states.copy()
        t = _onepop_knots_from_hidden_states(native_hidden_states)
        prefit_hidden_states = _balanced_hidden_states_constant(2 + K_requested, eta0=init_eta0)
        prefit_t = _onepop_knots_from_hidden_states(prefit_hidden_states)
        K = len(t) - 1

    logger.info(
        "SMC++ n_dist=%d n_undist=%d K=%d theta=%.6f rho=%.6f",
        n_distinguished, n_undist, K, theta, rho,
    )

    log_eta_init = np.full(K, np.log(init_eta0), dtype=np.float64)
    init_jitter = 0.0001 if n_distinguished == 2 else 0.01

    if n_distinguished == 2 and K_requested >= 2:
        log_eta_prefit_init = np.full(len(prefit_t) - 1, np.log(init_eta0), dtype=np.float64)
        log_eta_prefit_init += rng.normal(0.0, init_jitter, len(log_eta_prefit_init))
        prefit_hidden_states = np.array([0.0, np.inf], dtype=np.float64)
        prefit_log_eta, _prefit_meta = _optimize_log_eta_em(
            log_eta_prefit_init,
            records,
            n_undist,
            prefit_t,
            theta,
            rho_base,
            regularization,
            max_iterations=min(max_iterations, 1),
            n_distinguished=n_distinguished,
            hidden_states=prefit_hidden_states,
            polarization_error=polarization_error,
            observation_scale=1.0,
            single_coordinate=False,
        )
        mutation_window = max(int(2e-3 * n0_fixed / max(rho, 1e-300)), 1)
        native_hidden_states = _empirical_hidden_states_from_records(
            records,
            theta=theta,
            total_breaks=2 * K_requested,
            window_size=mutation_window,
        )
        if native_hidden_states is None:
            native_hidden_states = initial_hidden_states
        t = _onepop_knots_from_hidden_states(native_hidden_states)
        K = len(t) - 1
        log_eta_init = _piecewise_log_eta_at_knots(
            prefit_t[1:],
            prefit_log_eta,
            t[1:],
        )
    else:
        log_eta_init += rng.normal(0, init_jitter, K)

    fit_records = records
    if n_distinguished == 2:
        try:
            fit_records = _preprocess_onepop_records(records, n_undist)
            preprocessing = {
                "applied": True,
                "thinning": int(500 * np.log(2 + n_undist)),
                "bin_width": 100,
                "n_input_records": len(records),
                "n_fit_records": len(fit_records),
            }
            observation_scale = 100.0
        except RuntimeError:
            fit_records = records
            preprocessing = {
                "applied": False,
                "thinning": int(500 * np.log(2 + n_undist)),
                "bin_width": 100,
                "n_input_records": len(records),
                "n_fit_records": len(fit_records),
            }
            observation_scale = 1.0

    logger.info("Starting optimization (max_iter=%d)", max_iterations)

    initial_objective = _objective(
        log_eta_init,
        fit_records,
        n_undist,
        t,
        theta,
        rho_base,
        regularization,
        n_distinguished,
        native_hidden_states,
        polarization_error,
        observation_scale,
    )

    if n_distinguished == 2:
        log_eta_opt, optimizer_meta = _optimize_log_eta_em(
            log_eta_init,
            fit_records,
            n_undist,
            t,
            theta,
            rho_base,
            regularization,
            max_iterations=max_iterations,
            n_distinguished=n_distinguished,
            hidden_states=native_hidden_states,
            polarization_error=polarization_error,
            observation_scale=observation_scale,
        )
        result_success = bool(optimizer_meta["success"])
        result_message = str(optimizer_meta["message"])
        result_nit = int(optimizer_meta["n_iterations"])
        result_nfev = int(sum(
            step.get("nfev", 0)
            for history in optimizer_meta["history"]
            for step in history["m_steps"]
        ))
        final_objective = float(
            _objective(
                log_eta_opt,
                fit_records,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished,
                native_hidden_states,
                polarization_error,
                observation_scale,
            )
        )
    else:
        result = minimize(
            _objective,
            log_eta_init,
            args=(
                fit_records,
                n_undist,
                t,
                theta,
                rho_base,
                regularization,
                n_distinguished,
                native_hidden_states,
                polarization_error,
                observation_scale,
            ),
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )
        log_eta_opt = result.x
        optimizer_meta = None
        result_success = bool(result.success)
        result_message = str(result.message)
        result_nit = int(result.nit)
        result_nfev = int(result.nfev)
        final_objective = float(result.fun)

    eta_opt = np.exp(log_eta_opt)

    logger.info(
        "Optimization %s after %d iterations, f=%.4f",
        "converged" if result_success else "stopped",
        result_nit,
        final_objective,
    )

    if n_distinguished == 2:
        n0 = 0.5e-4 / mu
        ne = eta_opt * 2.0 * n0
    else:
        n0 = theta / (4.0 * mu)
        ne = eta_opt * n0
    if n_distinguished == 2:
        time_mid = t[1:]
    else:
        time_mid = (t[:-1] + t[1:]) / 2.0
    time_years = time_mid * 2.0 * n0 * generation_time

    hp = compute_hmm_params(
        eta_opt, n_undist, t, theta, rho_base,
        n_distinguished=n_distinguished,
        hidden_states=native_hidden_states,
        polarization_error=polarization_error,
        observation_scale=observation_scale,
    )
    final_ll = 0.0
    for rec in fit_records:
        obs = _maybe_expand_onepop_observation_spans(rec["observations"], hp)
        if len(obs) == 0:
            continue
        _, s_list = _forward_spans(hp, obs)
        final_ll += _log_likelihood_spans(s_list)

    smcpp_result = SmcppResult(
        time=time_mid,
        time_boundaries=t,
        eta=eta_opt,
        ne=ne,
        time_years=time_years,
        theta=theta,
        rho=rho,
        n0=n0,
        log_likelihood=final_ll,
        n_undist=n_undist,
        n_intervals=K,
        regularization=regularization,
        optimization_result={
            "success": result_success,
            "message": result_message,
            "n_iterations": result_nit,
            "n_function_evals": result_nfev,
            "initial_objective": float(initial_objective),
            "final_objective": final_objective,
            "preprocessing": preprocessing,
            "observation_scale": float(observation_scale),
            "history": [] if optimizer_meta is None else optimizer_meta["history"],
        },
    )

    data.results["smcpp"] = annotate_result({
        "time": smcpp_result.time,
        "time_boundaries": smcpp_result.time_boundaries,
        "eta": smcpp_result.eta,
        "ne": smcpp_result.ne,
        "time_years": smcpp_result.time_years,
        "theta": smcpp_result.theta,
        "rho": smcpp_result.rho,
        "n0": smcpp_result.n0,
        "log_likelihood": smcpp_result.log_likelihood,
        "n_undist": smcpp_result.n_undist,
        "n_distinguished": n_distinguished,
        "n_intervals": smcpp_result.n_intervals,
        "regularization": smcpp_result.regularization,
        "optimization": smcpp_result.optimization_result,
        "native_hidden_states": native_hidden_states,
        "preprocessing": preprocessing,
        "observation_scale": float(observation_scale),
    }, implementation_requested=implementation_requested, implementation_used="native")
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time
    data.params["recombination_rate"] = recombination_rate

    return data


def _smcpp_upstream(
    data: SmcData,
    n_intervals: int = 32,
    max_t: float = 15.0,
    alpha: float = 0.1,
    mu: float = 1.25e-8,
    recombination_rate: float = 1e-8,
    generation_time: float = 25.0,
    regularization: float = 1.0,
    max_iterations: int = 100,
    seed: int | None = None,
    upstream_options: dict | None = None,
    implementation_requested: str = "upstream",
) -> SmcData:
    """Run the real upstream SMC++ backend and map results into SmcData."""
    del max_t, alpha
    effective_args = {
        "n_intervals": int(n_intervals),
        "mu": float(mu),
        "recombination_rate": float(recombination_rate),
        "generation_time": float(generation_time),
        "regularization": float(regularization),
        "max_iterations": int(max_iterations),
        "seed": None if seed is None else int(seed),
    }
    if upstream_options:
        effective_args.update(upstream_options)
    payload = _run_upstream_smcpp(
        data,
        n_intervals=n_intervals,
        mu=mu,
        recombination_rate=recombination_rate,
        generation_time=generation_time,
        regularization=regularization,
        max_iterations=max_iterations,
        seed=seed,
    )

    time = np.asarray(payload["time"], dtype=np.float64)
    time_boundaries = np.asarray(payload["time_boundaries"], dtype=np.float64)
    ne = np.asarray(payload["ne"], dtype=np.float64)
    n0 = float(payload["n0"])
    eta = ne / max(2.0 * n0, 1e-30)

    data.results["smcpp"] = annotate_result({
        "time": time,
        "time_boundaries": time_boundaries,
        "eta": eta,
        "ne": ne,
        "time_years": np.asarray(payload["time_years"], dtype=np.float64),
        "theta": float(payload["theta"]),
        "rho": float(payload["rho"]),
        "n0": n0,
        "log_likelihood": float(payload["log_likelihood"]),
        "n_undist": int(payload["n_undist"]),
        "n_distinguished": int(payload["n_distinguished"]),
        "n_intervals": int(payload["n_intervals"]),
        "regularization": float(payload["regularization"]),
        "optimization": dict(payload["optimization"]),
        "backend": "upstream",
        "upstream": standard_upstream_metadata(
            "smcpp",
            effective_args=effective_args,
            extra={
                "alpha": float(payload["alpha"]),
                "model": payload["model"],
                "stepwise_ne": payload["stepwise_ne"],
                "hidden_states": payload["hidden_states"],
            },
        ),
    }, implementation_requested=implementation_requested, implementation_used="upstream")
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time
    data.params["recombination_rate"] = recombination_rate
    return data


def smcpp(
    data: SmcData,
    n_intervals: int = 32,
    max_t: float = 15.0,
    alpha: float = 0.1,
    mu: float = 1.25e-8,
    recombination_rate: float = 1e-8,
    generation_time: float = 25.0,
    regularization: float = 1.0,
    max_iterations: int = 100,
    seed: int | None = None,
    implementation: str = "auto",
    backend: str | None = None,
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run SMC++ demographic inference.

    Parameters are the same as the native implementation. ``implementation``
    may be ``"native"``, ``"upstream"``, or ``"auto"``. ``"auto"`` currently
    prefers upstream when the controlled side environment is available.
    ``backend`` is a deprecated compatibility alias.
    """
    implementation = normalize_implementation(
        implementation,
        backend=backend,
    )
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("smcpp"),
    )
    warn_if_native_not_trusted("smcpp", implementation_used)

    if implementation_used == "upstream":
        return _smcpp_upstream(
            data,
            n_intervals=n_intervals,
            max_t=max_t,
            alpha=alpha,
            mu=mu,
            recombination_rate=recombination_rate,
            generation_time=generation_time,
            regularization=regularization,
            max_iterations=max_iterations,
            seed=seed,
            upstream_options=upstream_options,
            implementation_requested=implementation,
        )

    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported smcpp native_options keys: {unsupported}")

    return _smcpp_native(
        data,
        n_intervals=n_intervals,
        max_t=max_t,
        alpha=alpha,
        mu=mu,
        recombination_rate=recombination_rate,
        generation_time=generation_time,
        regularization=regularization,
        max_iterations=max_iterations,
        seed=seed,
        implementation_requested=implementation,
    )
