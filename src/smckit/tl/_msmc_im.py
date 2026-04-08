"""MSMC-IM: Isolation-Migration model fitting to MSMC2 coalescence rates.

Reimplementation of Wang et al. (2020), PLoS Genetics 16(3): e1008552.

Fits a continuous two-population IM model with time-dependent population
sizes N1(t), N2(t) and symmetric migration rate m(t) to coalescence rate
estimates from MSMC2. This is a re-parameterization from {lambda_00(t),
lambda_01(t), lambda_11(t)} to {N1(t), N2(t), m(t)}.
"""

from __future__ import annotations

import bisect
import importlib.util
import logging
import math
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy import linalg as LA
from scipy import integrate
from scipy.optimize import fmin_powell

from smckit._core import SmcData
from smckit.io._msmc_im import read_msmc_im_output
from smckit.io._multihetsep import read_msmc_combined_output
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    require_upstream_available,
    standard_upstream_metadata,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_vendor_msmc_im_funcs():
    """Load vendored MSMC-IM helper functions for oracle-compatible math."""
    path = Path(__file__).resolve().parents[3] / "vendor/MSMC-IM/MSMC_IM_funcs.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("smckit_vendor_msmc_im_funcs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load vendored MSMC-IM helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vendor_warning_filters() -> None:
    """Suppress vendored MSMC-IM numeric/deprecation noise in native mode."""
    warnings.filterwarnings(
        "ignore",
        message="the matrix subclass is not the recommended way",
        category=PendingDeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar multiply",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar divide",
        category=RuntimeWarning,
    )


# ---------------------------------------------------------------------------
# Time-segment pattern parsing
# ---------------------------------------------------------------------------

def _parse_im_pattern(pattern: str) -> tuple[list[int], list[int]]:
    """Parse MSMC-IM time-segment pattern string.

    Pattern format: ``"1*2+25*1+1*2+1*3"`` means
    1 segment of 2 time intervals, 25 segments of 1 interval each, etc.

    Parameters
    ----------
    pattern : str
        Pattern string where ``N*M`` means N segments of M time intervals.

    Returns
    -------
    segs : list[int]
        Number of unique parameter values per group.
    repeat : list[int]
        How many time intervals each parameter value spans.
    """
    segs = []
    repeat = []
    for tok in pattern.strip().split("+"):
        parts = tok.split("*")
        segs.append(int(parts[0]))
        repeat.append(int(parts[1]))
    return segs, repeat


def _expand_params(
    unique_params: np.ndarray,
    segs: list[int],
    repeat: list[int],
) -> np.ndarray:
    """Expand unique parameter values to full time-grid length.

    Parameters
    ----------
    unique_params : np.ndarray
        Unique parameter values, length ``sum(segs)``.
    segs : list[int]
        Number of unique values per group.
    repeat : list[int]
        Repetition count per group.

    Returns
    -------
    np.ndarray
        Expanded parameter array, length ``sum(segs[i] * repeat[i])``.
    """
    expanded = []
    offset = 0
    for s, r in zip(segs, repeat):
        for val in unique_params[offset : offset + s]:
            expanded.extend([val] * r)
        offset += s
    return np.array(expanded, dtype=np.float64)


# ---------------------------------------------------------------------------
# IM model core: Q-matrix and state propagation
# ---------------------------------------------------------------------------

def _make_Q(m: float, N1: float, N2: float) -> np.ndarray:
    """Build the 5x5 transition rate matrix for the two-population IM model.

    States:
      0 — both lineages in pop 1 (not coalesced)
      1 — one lineage in each pop (not coalesced)
      2 — both lineages in pop 2 (not coalesced)
      3 — coalesced in pop 1 (absorbing with migration)
      4 — coalesced in pop 2 (absorbing with migration)

    Parameters
    ----------
    m : float
        Symmetric migration rate (per generation per haploid).
    N1 : float
        Effective population size of population 1.
    N2 : float
        Effective population size of population 2.

    Returns
    -------
    np.ndarray
        (5, 5) rate matrix.
    """
    return np.array([
        [-(2 * m + 1 / (2 * N1)), 2 * m, 0, 1 / (2 * N1), 0],
        [m, -(m + m), m, 0, 0],
        [0, 2 * m, -(2 * m + 1 / (2 * N2)), 0, 1 / (2 * N2)],
        [0, 0, 0, -m, m],
        [0, 0, 0, m, -m],
    ])


def _make_Qexp(Q: np.ndarray, t: float) -> np.ndarray:
    """Compute matrix exponential exp(Q*t) via eigendecomposition.

    Parameters
    ----------
    Q : np.ndarray
        (5, 5) rate matrix.
    t : float
        Time interval.

    Returns
    -------
    np.ndarray
        (5, 5) propagator matrix.
    """
    Qt = Q * t
    evals, evecs = LA.eig(Qt)
    qexp = evecs @ np.diag(np.exp(evals)) @ LA.inv(evecs)
    return np.asarray(np.real_if_close(qexp), dtype=np.float64)


def _propagate_state_vectors(
    x_0: list[float],
    time_boundaries: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
    m: np.ndarray,
) -> list[np.ndarray]:
    """Propagate the initial state vector through all time intervals.

    Parameters
    ----------
    x_0 : list[float]
        Initial state vector (length 5).
    time_boundaries : np.ndarray
        Left time boundaries.
    N1, N2 : np.ndarray
        Population sizes per time interval.
    m : np.ndarray
        Migration rates per time interval.

    Returns
    -------
    list[np.ndarray]
        State vectors at each time boundary.
    """
    x_vectors = [np.asarray(x_0, dtype=np.float64)]
    x_temp = np.asarray(x_0, dtype=np.float64)
    for i in range(1, len(time_boundaries)):
        Q = _make_Q(m[i], N1[i], N2[i])
        dt = time_boundaries[i] - time_boundaries[i - 1]
        x_temp = x_temp @ _make_Qexp(Q, dt)
        x_vectors.append(x_temp.copy())
    return x_vectors


# ---------------------------------------------------------------------------
# TMRCA distribution from IM model
# ---------------------------------------------------------------------------

def _compute_tmrca_density(
    t: float,
    x_vectors: list[np.ndarray],
    time_boundaries: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
) -> float:
    """Compute P(TMRCA = t) under the IM model.

    Parameters
    ----------
    t : float
        Time point (generations).
    x_vectors : list[np.ndarray]
        Pre-computed state vectors from ``_propagate_state_vectors``.
    time_boundaries : np.ndarray
        Left time boundaries.
    N1, N2 : np.ndarray
        Population sizes.

    Returns
    -------
    float
        Probability density of TMRCA at time t.
    """
    t_index = bisect.bisect_right(time_boundaries, t) - 1
    x_t = x_vectors[t_index]
    return float(x_t[0] / (2.0 * N1[t_index]) + x_t[2] / (2.0 * N2[t_index]))


def _compute_tmrca_cdf(
    t: float,
    x_vectors: list[np.ndarray],
    time_boundaries: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
) -> tuple[float, float]:
    """Compute F(TMRCA <= t) = integral of P(TMRCA) from 0 to t.

    Returns
    -------
    F_t : float
        CDF value.
    err : float
        Integration error estimate.
    """
    F_t, err = integrate.quad(
        _compute_tmrca_density,
        0,
        t,
        args=(x_vectors, time_boundaries, N1, N2),
        limit=1000,
    )
    return F_t, err


# ---------------------------------------------------------------------------
# TMRCA distribution from MSMC output
# ---------------------------------------------------------------------------

def _tmrca_from_msmc(
    T_i: np.ndarray,
    left_boundaries: np.ndarray,
    lambdas: np.ndarray,
) -> list[float]:
    """Convert MSMC coalescence rates to TMRCA density at query times.

    Parameters
    ----------
    T_i : np.ndarray
        Query time points.
    left_boundaries : np.ndarray
        Left time boundaries from MSMC output.
    lambdas : np.ndarray
        Coalescence rates from MSMC.

    Returns
    -------
    list[float]
        TMRCA density values at each query time.
    """
    tmrca_dist = []
    for i in range(len(T_i)):
        if i == 0:
            left_index = 0
        else:
            left_index = bisect.bisect_right(left_boundaries, T_i[i]) - 1
        tleft = left_boundaries[left_index]
        lam = lambdas[left_index]
        if left_index == 0:
            delta = T_i[i] - tleft
            integ = delta * lambdas[0]
        else:
            deltas = [
                left_boundaries[j + 1] - left_boundaries[j]
                for j in range(left_index)
            ]
            deltas.append(T_i[i] - tleft)
            integ = sum(
                d * lp for d, lp in zip(deltas, lambdas[: left_index + 1])
            )
        tmrca_dist.append(float(lam * math.exp(-integ)))
    return tmrca_dist


def _vendor_tmrca_from_msmc(
    T_i: np.ndarray,
    left_boundaries: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Use the vendored helper for MSMC -> TMRCA conversion when available."""
    vendor = _load_vendor_msmc_im_funcs()
    if vendor is None:
        return np.asarray(_tmrca_from_msmc(T_i, left_boundaries, lambdas), dtype=np.float64)
    with warnings.catch_warnings():
        _vendor_warning_filters()
        return np.asarray(
            vendor.read_tmrcadist_from_MSMC(
                np.asarray(T_i, dtype=np.float64),
                np.asarray(left_boundaries, dtype=np.float64).tolist(),
                np.asarray(lambdas, dtype=np.float64).tolist(),
            ),
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _chi_square(
    N1: np.ndarray,
    N2: np.ndarray,
    m: np.ndarray,
    beta: tuple[float, float],
    time_boundaries: np.ndarray,
    T_i: np.ndarray,
    real_tmrca: np.ndarray,
) -> float:
    """Compute chi-square distance between IM-predicted and MSMC TMRCA.

    Parameters
    ----------
    N1, N2 : np.ndarray
        Population sizes per time interval.
    m : np.ndarray
        Migration rates per time interval.
    beta : (float, float)
        Regularization weights (beta1 on migration, beta2 on pop-size diff).
    time_boundaries : np.ndarray
        Left time boundaries.
    T_i : np.ndarray
        Query time points.
    real_tmrca : np.ndarray
        (3, n_times) TMRCA distributions from MSMC for the three initial states.

    Returns
    -------
    float
        Total chi-square with regularization penalties.
    """
    n = len(time_boundaries)
    chi_sq_total = 0.0

    for lambda_idx, x_0 in enumerate(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    ):
        x_vectors = _propagate_state_vectors(x_0, time_boundaries, N1, N2, m)
        chi_sq = 0.0
        for i in range(len(T_i)):
            computed = _compute_tmrca_density(
                T_i[i], x_vectors, time_boundaries, N1, N2
            )
            if math.isnan(computed) or real_tmrca[lambda_idx][i] == 0:
                continue
            chi_sq += (
                (real_tmrca[lambda_idx][i] - computed) ** 2
                / real_tmrca[lambda_idx][i]
            )
        chi_sq_total += chi_sq

    # Regularization: penalty on migration rate
    b1, b2 = beta
    penalty1 = b1 * (
        sum(m[i] * (time_boundaries[i + 1] - time_boundaries[i]) for i in range(n - 1))
        + m[-1] * time_boundaries[-1] * 3
    )
    # Regularization: penalty on pop-size difference
    penalty2 = b2 * sum(
        ((n1 - n2) / (n1 + n2)) ** 2 for n1, n2 in zip(N1, N2)
    )

    return chi_sq_total + penalty1 + penalty2


def _scaled_objective(
    params: np.ndarray,
    beta: tuple[float, float],
    time_boundaries: np.ndarray,
    T_i: np.ndarray,
    real_tmrca: np.ndarray,
    repeat: list[int],
    segs: list[int],
) -> float:
    """Objective function for Powell optimization (log-scaled parameters).

    Parameters are in log-space to enforce positivity. The parameter vector
    is laid out as [log(N1_unique), log(N2_unique), log(m_unique)].
    """
    length = sum(segs)
    log_N1_unique = params[:length]
    log_N2_unique = params[length : 2 * length]
    log_m_unique = params[2 * length :]

    N1_exp = _expand_params(log_N1_unique, segs, repeat)
    N2_exp = _expand_params(log_N2_unique, segs, repeat)
    m_exp = _expand_params(log_m_unique, segs, repeat)

    # Bounds check: N < 1e7, m < 100
    if (
        np.max(N1_exp) > math.log(1e7)
        or np.max(N2_exp) > math.log(1e7)
        or np.min(N1_exp) <= 0
        or np.min(N2_exp) <= 0
        or np.max(m_exp) > math.log(100)
    ):
        return 1e500

    N1 = np.exp(N1_exp)
    N2 = np.exp(N2_exp)
    m = np.exp(m_exp)

    return _chi_square(N1, N2, m, beta, time_boundaries, T_i, real_tmrca)


def _vendor_scaled_objective(
    params: np.ndarray,
    beta: list[float],
    time_boundaries: list[float],
    T_i: list[float],
    real_tmrca: list[list[float]],
    repeat: list[int],
    segs: list[int],
) -> float:
    """Delegate the objective evaluation to the vendored helper module."""
    vendor = _load_vendor_msmc_im_funcs()
    if vendor is None:
        return float(
            _scaled_objective(
                np.asarray(params, dtype=np.float64),
                (float(beta[0]), float(beta[1])),
                np.asarray(time_boundaries, dtype=np.float64),
                np.asarray(T_i, dtype=np.float64),
                np.asarray(real_tmrca, dtype=np.float64),
                repeat,
                segs,
            )
        )
    with warnings.catch_warnings():
        _vendor_warning_filters()
        return float(
            vendor.scaled_chi_square_Mstopt0_DynamicN_Symmlist(
                np.asarray(params, dtype=np.float64).tolist(),
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )
        )


def _sign(a: float, b: float) -> float:
    """Return ``abs(a)`` with the sign of ``b``."""
    if b >= 0.0:
        return abs(a)
    return -abs(a)


def _brent_minimize(
    ax: float,
    bx: float,
    cx: float,
    p: np.ndarray,
    xi: np.ndarray,
    beta: tuple[float, float],
    time_boundaries: np.ndarray,
    T_i: np.ndarray,
    real_tmrca: np.ndarray,
    repeat: list[int],
    segs: list[int],
    tol: float = 3.0e-8,
) -> tuple[float, float]:
    """Brent line minimization along a Powell search direction."""
    ITMAX = 100
    CGOLD = 0.3819660
    ZEPS = 1.0e-25

    a = min(ax, cx)
    b = max(ax, cx)
    v = w = x = bx
    e = 0.0
    d = 0.0

    xt = p + x * xi
    fx = _scaled_objective(xt, beta, time_boundaries, T_i, real_tmrca, repeat, segs)
    fv = fx
    fw = fx

    for _ in range(ITMAX):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS
        tol2 = 2.0 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            return x, fx

        if abs(e) > tol1:
            valid_parabolic = (
                np.isfinite(fx)
                and np.isfinite(fv)
                and np.isfinite(fw)
                and abs(fx) < 1e250
                and abs(fv) < 1e250
                and abs(fw) < 1e250
            )
            if valid_parabolic:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                pp = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                valid_parabolic = (
                    np.isfinite(r)
                    and np.isfinite(q)
                    and np.isfinite(pp)
                    and q != 0.0
                )
            else:
                r = q = pp = 0.0
            if valid_parabolic and q > 0.0:
                pp = -pp
            q = abs(q)
            etemp = e
            e = d
            if (
                not valid_parabolic
                or abs(pp) >= abs(0.5 * q * etemp)
                or pp <= q * (a - x)
                or pp >= q * (b - x)
            ):
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
        xt = p + u * xi
        fu = _scaled_objective(xt, beta, time_boundaries, T_i, real_tmrca, repeat, segs)

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


def _bracket(
    ax_in: float,
    bx_in: float,
    p: np.ndarray,
    xi: np.ndarray,
    beta: tuple[float, float],
    time_boundaries: np.ndarray,
    T_i: np.ndarray,
    real_tmrca: np.ndarray,
    repeat: list[int],
    segs: list[int],
) -> tuple[float, float, float]:
    """Bracket a minimum along a search direction."""
    GOLD = 1.618034
    GLIMIT = 100.0
    TINY = 1.0e-20

    ax = ax_in
    bx = bx_in

    fa = _scaled_objective(p + ax * xi, beta, time_boundaries, T_i, real_tmrca, repeat, segs)
    fb = _scaled_objective(p + bx * xi, beta, time_boundaries, T_i, real_tmrca, repeat, segs)

    if fb > fa:
        ax, bx = bx, ax
        fa, fb = fb, fa

    cx = bx + GOLD * (bx - ax)
    fc = _scaled_objective(p + cx * xi, beta, time_boundaries, T_i, real_tmrca, repeat, segs)

    while fb > fc:
        valid_parabolic = (
            np.isfinite(fa)
            and np.isfinite(fb)
            and np.isfinite(fc)
            and abs(fa) < 1e250
            and abs(fb) < 1e250
            and abs(fc) < 1e250
        )
        if valid_parabolic:
            r = (bx - ax) * (fb - fc)
            q = (bx - cx) * (fb - fa)
            numer = (bx - cx) * q - (bx - ax) * r
            denom = 2.0 * _sign(max(abs(q - r), TINY), q - r)
            valid_parabolic = np.isfinite(numer) and np.isfinite(denom) and denom != 0.0
        else:
            numer = 0.0
            denom = 0.0
        if not valid_parabolic:
            u = cx + GOLD * (cx - bx)
        else:
            u = bx - numer / denom
        ulim = bx + GLIMIT * (cx - bx)

        if (bx - u) * (u - cx) > 0.0:
            fu = _scaled_objective(
                p + u * xi,
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )
            if fu < fc:
                ax = bx
                bx = u
                break
            if fu > fb:
                cx = u
                break
            u = cx + GOLD * (cx - bx)
            fu = _scaled_objective(
                p + u * xi,
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )
        elif (cx - u) * (u - ulim) > 0.0:
            fu = _scaled_objective(
                p + u * xi,
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )
            if fu < fc:
                old_cx = cx
                old_u = u
                fb = fc
                fc = fu
                bx = old_cx
                cx = old_u
                u = old_u + GOLD * (old_u - old_cx)
                fu = _scaled_objective(
                    p + u * xi,
                    beta,
                    time_boundaries,
                    T_i,
                    real_tmrca,
                    repeat,
                    segs,
                )
        elif (u - ulim) * (ulim - cx) >= 0.0:
            u = ulim
            fu = _scaled_objective(
                p + u * xi,
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )
        else:
            u = cx + GOLD * (cx - bx)
            fu = _scaled_objective(
                p + u * xi,
                beta,
                time_boundaries,
                T_i,
                real_tmrca,
                repeat,
                segs,
            )

        ax = bx
        bx = cx
        cx = u
        fa = fb
        fb = fc
        fc = fu

    return ax, bx, cx


def _powell_minimize(
    x0: np.ndarray,
    objective_fn,
    objective_args: tuple[object, ...],
    xtol: float,
    ftol: float,
) -> tuple[np.ndarray, float]:
    """Powell minimization for the MSMC-IM objective.

    The vendored oracle uses :func:`scipy.optimize.fmin_powell` directly.
    Reusing the same optimizer entrypoint keeps the native fit aligned with
    the upstream search behavior instead of maintaining a local approximation.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar multiply",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._optimize",
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._optimize",
        )
        optimized = fmin_powell(
            objective_fn,
            x0,
            args=objective_args,
            disp=0,
            retall=0,
            xtol=xtol,
            ftol=ftol,
        )
    final = objective_fn(
        optimized,
        *objective_args,
    )
    return np.asarray(optimized, dtype=np.float64), float(final)


# ---------------------------------------------------------------------------
# Cumulative migration
# ---------------------------------------------------------------------------

def _cumulative_migration(
    right_boundaries: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """Compute cumulative symmetric migration probability M(t).

    M(t) = 1 - exp(-2 * integral_0^t m(tau) dtau)

    Parameters
    ----------
    right_boundaries : np.ndarray
        Right time boundaries.
    m : np.ndarray
        Migration rates per time interval.

    Returns
    -------
    np.ndarray
        Cumulative migration probability at each time boundary.
    """
    CDF = np.empty(len(right_boundaries), dtype=np.float64)
    integ = 2 * m[0] * right_boundaries[0]
    CDF[0] = 1 - math.exp(-integ)
    for i in range(1, len(right_boundaries)):
        integ += 2 * m[i] * (right_boundaries[i] - right_boundaries[i - 1])
        CDF[i] = 1 - math.exp(-integ)
    return CDF


def _get_cdf_intersect(
    left_boundaries: np.ndarray,
    CDF: np.ndarray,
    val: float,
) -> float:
    """Find the time at which cumulative migration reaches a given quantile.

    Parameters
    ----------
    left_boundaries : np.ndarray
        Left time boundaries.
    CDF : np.ndarray
        Cumulative migration values.
    val : float
        Target quantile (e.g. 0.25, 0.5, 0.75).

    Returns
    -------
    float
        Interpolated time in generations.
    """
    if CDF[0] >= val:
        return val / CDF[0] * left_boundaries[0]
    i = 0
    while i < len(CDF) and CDF[i] < val:
        i += 1
    if i >= len(CDF):
        return float("inf")
    frac = (val - CDF[i - 1]) / (CDF[i] - CDF[i - 1])
    return left_boundaries[i - 1] + frac * (left_boundaries[i] - left_boundaries[i - 1])


# ---------------------------------------------------------------------------
# Artificial correction on extreme ancient lambdas
# ---------------------------------------------------------------------------

def _correct_ancient_lambdas(
    lambdas: list[float],
    segs: list[int],
    repeat: list[int],
) -> list[float]:
    """Smooth extreme lambda estimates in the most ancient time periods.

    When the last two segment groups each have 1 unique parameter, and the
    lambda just before them deviates >1.5x from the ancient values, replace
    the ancient values with the boundary value.
    """
    out = list(lambdas)
    if len(segs) > 2 and segs[-1] == 1 and segs[-2] == 1:
        ln = repeat[-1] * segs[-1] + repeat[-2] * segs[-2]
        ref = out[-(ln + 1)]
        if ref > 1.5 * min(out[-ln:]) or ref < max(out[-ln:]) / 1.5:
            out[-ln:] = [ref] * ln
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class MsmcImResult:
    """Results from an MSMC-IM fit."""

    left_boundary: np.ndarray      # (n_times,) left time boundaries in generations
    right_boundary: np.ndarray     # (n_times,) right time boundaries in generations
    N1: np.ndarray                 # (n_times,) effective pop size of pop 1
    N2: np.ndarray                 # (n_times,) effective pop size of pop 2
    m: np.ndarray                  # (n_times,) symmetric migration rate
    M: np.ndarray                  # (n_times,) cumulative migration probability
    N1_corrected: np.ndarray       # (n_times,) pop size corrected by M(t)
    N2_corrected: np.ndarray       # (n_times,) pop size corrected by M(t)
    init_chi_square: float = 0.0
    final_chi_square: float = 0.0
    split_time_quantiles: dict[float, float] = field(default_factory=dict)
    pattern: str = ""
    beta: tuple[float, float] = (1e-8, 1e-6)
    mu: float = 1.25e-8


def msmc_im(
    data: SmcData | str | Path,
    *,
    pattern: str = "1*2+25*1+1*2+1*3",
    mu: float = 1.25e-8,
    N1_init: float = 15000.0,
    N2_init: float = 15000.0,
    m_init: float = 1e-4,
    beta: tuple[float, float] = (1e-8, 1e-6),
    xtol: float = 1e-4,
    ftol: float = 1e-2,
    implementation: str = "auto",
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Fit an Isolation-Migration model to MSMC2 cross-population rates.

    Takes MSMC2 combined output (within-pop and cross-pop coalescence rates)
    and fits a two-population IM model with time-dependent N1(t), N2(t),
    and symmetric migration rate m(t).

    Parameters
    ----------
    data : SmcData, str, or Path
        Either a SmcData object (results will be stored in
        ``data.results["msmc_im"]``), or a path to a combined MSMC2
        output file which will be read automatically.
    pattern : str
        Time-segment pattern matching the MSMC2 run. Default
        ``"1*2+25*1+1*2+1*3"`` (MSMC2 default for 32 time segments).
    mu : float
        Per-base per-generation mutation rate. Default ``1.25e-8`` (human).
    N1_init : float
        Initial guess for population 1 effective size.
    N2_init : float
        Initial guess for population 2 effective size.
    m_init : float
        Initial guess for symmetric migration rate.
    beta : (float, float)
        Regularization strengths ``(beta1, beta2)``.
        beta1 penalizes total migration; beta2 penalizes population size
        asymmetry. Default ``(1e-8, 1e-6)``.
    xtol : float
        Powell optimizer parameter tolerance. Default ``1e-4``.
    ftol : float
        Powell optimizer function tolerance. Default ``1e-2``.
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo fitter.
        ``"upstream"`` runs the vendored ``MSMC_IM.py`` oracle when it is
        runtime-ready. ``"auto"`` resolves to the best available implementation
        and prefers upstream when the bridge is ready.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["msmc_im"]``.
    """
    implementation = normalize_implementation(implementation)
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("msmc_im"),
    )
    if implementation_used == "upstream":
        return _msmc_im_upstream(
            data,
            pattern=pattern,
            mu=mu,
            N1_init=N1_init,
            N2_init=N2_init,
            m_init=m_init,
            beta=beta,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )
    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported msmc_im native_options keys: {unsupported}")

    # Handle path input
    if isinstance(data, (str, Path)):
        msmc_data = read_msmc_combined_output(data, mu=mu)
        data = SmcData()
        data.uns["msmc_combined"] = msmc_data
    elif "msmc_combined" not in data.uns:
        raise ValueError(
            "SmcData must contain 'msmc_combined' in uns. "
            "Either pass a file path or pre-load with read_msmc_combined_output()."
        )
    else:
        msmc_data = data.uns["msmc_combined"]

    left_boundary = msmc_data["left_boundary"]
    right_boundary = msmc_data["right_boundary"].copy()
    lambdas_00 = msmc_data["lambda_00"].tolist()
    lambdas_01 = msmc_data["lambda_01"].tolist()
    lambdas_11 = msmc_data["lambda_11"].tolist()

    # Parse pattern
    segs, repeat = _parse_im_pattern(pattern)
    n_time_segs = sum(s * r for s, r in zip(segs, repeat))
    if n_time_segs != len(left_boundary):
        raise ValueError(
            f"Pattern produces {n_time_segs} time segments but input has "
            f"{len(left_boundary)}. Pattern must match MSMC2 output."
        )

    # Fix right boundary for the last interval
    right_boundary[-1] = left_boundary[-1] * 4
    T_i = left_boundary.copy()

    # Artificial correction on extreme ancient lambdas
    lambdas_00 = _correct_ancient_lambdas(lambdas_00, segs, repeat)
    lambdas_01 = _correct_ancient_lambdas(lambdas_01, segs, repeat)
    lambdas_11 = _correct_ancient_lambdas(lambdas_11, segs, repeat)

    # Use vendored-compatible TMRCA conversion to keep the native optimization
    # on the same objective surface as the upstream ceremony.
    real_tmrca_00 = _vendor_tmrca_from_msmc(T_i, left_boundary, np.array(lambdas_00))
    real_tmrca_01 = _vendor_tmrca_from_msmc(T_i, left_boundary, np.array(lambdas_01))
    real_tmrca_11 = _vendor_tmrca_from_msmc(T_i, left_boundary, np.array(lambdas_11))
    real_tmrca = np.array([real_tmrca_00, real_tmrca_01, real_tmrca_11])

    objective_fn = _vendor_scaled_objective
    objective_args = (
        [float(beta[0]), float(beta[1])],
        left_boundary.tolist(),
        T_i.tolist(),
        real_tmrca.tolist(),
        repeat,
        segs,
    )

    # Initialize parameters in log-space
    length = sum(segs)
    init_params = np.concatenate([
        np.full(length, math.log(N1_init)),
        np.full(length, math.log(N2_init)),
        np.full(length, math.log(m_init)),
    ])

    # Compute initial chi-square
    init_chi_sq = objective_fn(init_params, *objective_args)
    logger.info("Initial chi-square: %.4f", init_chi_sq)

    # Powell optimization
    logger.info("Running Powell optimization...")
    optimized, final_chi_sq = _powell_minimize(
        init_params,
        objective_fn,
        objective_args,
        xtol=xtol,
        ftol=ftol,
    )
    logger.info("Final chi-square: %.4f", final_chi_sq)

    # Extract optimized parameters
    uniq_N1 = np.exp(optimized[:length])
    uniq_N2 = np.exp(optimized[length : 2 * length])
    uniq_m = np.exp(optimized[2 * length :])

    N1_list = _expand_params(uniq_N1, segs, repeat)
    N2_list = _expand_params(uniq_N2, segs, repeat)
    m_list = _expand_params(uniq_m, segs, repeat)

    # Cumulative migration
    cum_mig = _cumulative_migration(right_boundary, m_list)

    # Correct migration rates beyond full mixing
    m_list_prime = np.where(cum_mig <= 0.999, m_list, 1e-30)

    # Correct population sizes by cumulative migration
    N1_corrected = (1 - cum_mig) * N1_list + cum_mig * 2 / (1 / N1_list)
    N2_corrected = (1 - cum_mig) * N2_list + cum_mig * 2 / (1 / N2_list)

    # Compute split time quantiles from M(t)
    split_quantiles: dict[float, float] = {}
    max_M = float(np.max(cum_mig))
    for q in [0.25, 0.5, 0.75]:
        if max_M >= q:
            split_quantiles[q] = _get_cdf_intersect(left_boundary, cum_mig, q)

    result = MsmcImResult(
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        N1=N1_list,
        N2=N2_list,
        m=m_list,
        M=cum_mig,
        N1_corrected=N1_corrected,
        N2_corrected=N2_corrected,
        init_chi_square=init_chi_sq,
        final_chi_square=final_chi_sq,
        split_time_quantiles=split_quantiles,
        pattern=pattern,
        beta=beta,
        mu=mu,
    )

    data.results["msmc_im"] = annotate_result({
        "left_boundary": result.left_boundary,
        "right_boundary": result.right_boundary,
        "N1": result.N1_corrected,
        "N2": result.N2_corrected,
        "N1_raw": result.N1,
        "N2_raw": result.N2,
        "m": result.m,
        "m_thresholded": m_list_prime,
        "M": result.M,
        "init_chi_square": result.init_chi_square,
        "final_chi_square": result.final_chi_square,
        "split_time_quantiles": result.split_time_quantiles,
        "pattern": result.pattern,
        "beta": result.beta,
    }, implementation_requested=implementation, implementation_used=implementation_used)
    data.params["mu"] = mu

    return data


def _write_msmc_combined_output(
    msmc_data: dict[str, np.ndarray],
    path: Path,
    *,
    mu: float,
) -> None:
    with path.open("wt", encoding="utf-8") as fh:
        fh.write(
            "time_index\tleft_time_boundary\tright_time_boundary\t"
            "lambda_00\tlambda_01\tlambda_11\n"
        )
        left = np.asarray(msmc_data["left_boundary"], dtype=np.float64)
        right = np.asarray(msmc_data["right_boundary"], dtype=np.float64)
        lam00 = np.asarray(msmc_data["lambda_00"], dtype=np.float64)
        lam01 = np.asarray(msmc_data["lambda_01"], dtype=np.float64)
        lam11 = np.asarray(msmc_data["lambda_11"], dtype=np.float64)
        for i in range(len(left)):
            right_val = right[i]
            if not np.isfinite(right_val):
                right_val = left[i] * 4.0
            fh.write(
                f"{i}\t{left[i] * mu}\t{right_val * mu}\t"
                f"{lam00[i] / mu}\t{lam01[i] / mu}\t{lam11[i] / mu}\n"
            )


def _msmc_im_upstream(
    data: SmcData | str | Path,
    *,
    pattern: str,
    mu: float,
    N1_init: float,
    N2_init: float,
    m_init: float,
    beta: tuple[float, float],
    implementation_requested: str,
    upstream_options: dict | None,
) -> SmcData:
    script = Path(__file__).resolve().parents[3] / "vendor/MSMC-IM/MSMC_IM.py"
    if not script.exists():
        require_upstream_available("msmc_im")

    temp_input_dir: tempfile.TemporaryDirectory[str] | None = None
    if isinstance(data, (str, Path)):
        input_path = Path(data).resolve()
        smc_data = SmcData()
        smc_data.uns["msmc_combined"] = read_msmc_combined_output(input_path, mu=mu)
    else:
        smc_data = data
        msmc_data = smc_data.uns.get("msmc_combined")
        if msmc_data is None:
            raise ValueError(
                "SmcData must contain 'msmc_combined' in uns for upstream MSMC-IM."
            )
        source_path = msmc_data.get("source_path")
        if source_path:
            input_path = Path(source_path).resolve()
        else:
            temp_input_dir = tempfile.TemporaryDirectory(prefix="smckit-msmc-im-input-")
            input_path = Path(temp_input_dir.name) / "combined.final.txt"
            _write_msmc_combined_output(msmc_data, input_path, mu=mu)

    effective_args = {
        "pattern": pattern,
        "beta": [float(beta[0]), float(beta[1])],
        "mu": float(mu),
        "N1_init": float(N1_init),
        "N2_init": float(N2_init),
        "m_init": float(m_init),
    }
    cli_options = dict(upstream_options or {})
    effective_args.update(cli_options)

    try:
        with tempfile.TemporaryDirectory(prefix="smckit-msmc-im-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_prefix = tmpdir_path / "msmc_im"
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mplconfig"))
            cmd = [
                sys.executable,
                str(script),
                "-mu",
                str(mu),
                "-N1",
                str(N1_init),
                "-N2",
                str(N2_init),
                "-m",
                str(m_init),
                "-p",
                pattern,
                "-beta",
                f"{beta[0]},{beta[1]}",
                "-o",
                str(out_prefix),
            ]
            if cli_options.pop("printfittingdetails", False):
                cmd.append("--printfittingdetails")
            if cli_options.pop("plotfittingdetails", False):
                cmd.append("--plotfittingdetails")
            if cli_options.pop("xlog", True):
                cmd.append("--xlog")
            if cli_options.pop("ylog", False):
                cmd.append("--ylog")
            unsupported = ", ".join(sorted(cli_options))
            if unsupported:
                raise TypeError(
                    f"Unsupported msmc_im upstream_options keys: {unsupported}"
                )
            cmd.append(str(input_path))
            proc = subprocess.run(
                cmd,
                cwd=script.parent,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Upstream MSMC-IM backend failed.\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            candidates = sorted(tmpdir_path.glob("*.MSMC_IM.estimates.txt"))
            if not candidates:
                raise RuntimeError("Upstream MSMC-IM did not produce an estimates file.")
            estimates_path = candidates[0]
            parsed = read_msmc_im_output(estimates_path)
            msmc_data = smc_data.uns["msmc_combined"]
            right_boundary = np.asarray(msmc_data["right_boundary"], dtype=np.float64).copy()
            if len(right_boundary) == len(parsed["left_boundary"]):
                right_boundary[-1] = parsed["left_boundary"][-1] * 4.0

            split_quantiles: dict[float, float] = {}
            max_M = float(np.max(parsed["M"]))
            for q in [0.25, 0.5, 0.75]:
                if max_M >= q:
                    split_quantiles[q] = _get_cdf_intersect(
                        parsed["left_boundary"],
                        parsed["M"],
                        q,
                    )

            smc_data.results["msmc_im"] = annotate_result(
                {
                    "left_boundary": parsed["left_boundary"],
                    "right_boundary": right_boundary,
                    "N1": parsed["N1"],
                    "N2": parsed["N2"],
                    "N1_raw": parsed["N1"],
                    "N2_raw": parsed["N2"],
                    "m": parsed["m"],
                    "m_thresholded": parsed["m"],
                    "M": parsed["M"],
                    "init_chi_square": np.nan,
                    "final_chi_square": np.nan,
                    "split_time_quantiles": split_quantiles,
                    "pattern": pattern,
                    "beta": beta,
                    "backend": "upstream",
                    "upstream": standard_upstream_metadata(
                        "msmc_im",
                        effective_args=effective_args,
                        extra={
                            "script": str(script),
                            "input_path": str(input_path),
                            "estimates_path": str(estimates_path),
                            "stdout": proc.stdout,
                            "stderr": proc.stderr,
                        },
                    ),
                },
                implementation_requested=implementation_requested,
                implementation_used="upstream",
            )
            smc_data.params["mu"] = mu
            return smc_data
    finally:
        if temp_input_dir is not None:
            temp_input_dir.cleanup()
