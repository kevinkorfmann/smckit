"""Numba JIT backend for HMM operations, replacing CuPy/CUDA GPU backends.

All functions use @numba.njit(cache=True) with explicit loops instead of
NumPy vectorized operations. Numba compiles these loops to native machine
code, achieving performance close to handwritten C without requiring a GPU.

First call to each function triggers JIT compilation. The compiled code is
cached to disk (via cache=True) so subsequent imports are fast.
"""

from __future__ import annotations

import math

import numba
import numpy as np

HMM_TINY = 1e-25
PSMC_T_INF = 1000.0
PSMC_N_PARAMS = 3


# ---------------------------------------------------------------------------
# 1. Scaled forward algorithm
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def forward_jit(a, e, a0, seq):
    """Scaled forward algorithm.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix
    a0 : (n,) initial state distribution
    seq : (L,) int8 observation sequence

    Returns
    -------
    f : (L, n) scaled forward probabilities
    s : (L,) scaling factors
    """
    n = a.shape[0]
    L = seq.shape[0]
    f = np.zeros((L, n), dtype=np.float64)
    s = np.ones(L, dtype=np.float64)

    # Transpose a once for cache-friendly matvec
    at = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            at[i, j] = a[j, i]

    # f[0] = a0 * e[seq[0]]
    sym0 = seq[0]
    total = 0.0
    for k in range(n):
        val = a0[k] * e[sym0, k]
        f[0, k] = val
        total += val
    s[0] = total
    if total > 0.0:
        for k in range(n):
            f[0, k] /= total

    # f[u] for u = 1..L-1
    for u in range(1, L):
        sym = seq[u]
        total = 0.0
        for k in range(n):
            # dot product: at[k,:] @ f[u-1,:]
            dot = 0.0
            for j in range(n):
                dot += at[k, j] * f[u - 1, j]
            val = e[sym, k] * dot
            f[u, k] = val
            total += val
        s[u] = total
        if total > 0.0:
            for k in range(n):
                f[u, k] /= total

    return f, s


# ---------------------------------------------------------------------------
# 2. Scaled backward algorithm
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def backward_jit(a, e, seq, s):
    """Scaled backward algorithm.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix
    seq : (L,) observation sequence
    s : (L,) scaling factors from forward algorithm

    Returns
    -------
    b : (L, n) scaled backward probabilities
    """
    n = a.shape[0]
    m_plus_1 = e.shape[0]
    L = seq.shape[0]
    b = np.zeros((L, n), dtype=np.float64)

    # Precompute ae[sym, k, l] = a[k, l] * e[sym, l]
    ae = np.empty((m_plus_1, n, n), dtype=np.float64)
    for sym in range(m_plus_1):
        for k in range(n):
            for l in range(n):
                ae[sym, k, l] = a[k, l] * e[sym, l]

    # b[L-1]
    if s[L - 1] > 0.0:
        inv_s = 1.0 / s[L - 1]
        for k in range(n):
            b[L - 1, k] = inv_s
    # else: b[L-1] stays 0.0

    # b[u] for u = L-2..0
    for u in range(L - 2, -1, -1):
        sym_next = seq[u + 1]
        for k in range(n):
            dot = 0.0
            for l in range(n):
                dot += ae[sym_next, k, l] * b[u + 1, l]
            b[u, k] = dot
        if s[u] > 0.0:
            for k in range(n):
                b[u, k] /= s[u]

    return b


# ---------------------------------------------------------------------------
# 3. Log-likelihood from scaling factors
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def log_likelihood_jit(s):
    """Compute log P(observations) from scaling factors.

    Parameters
    ----------
    s : (L,) scaling factors

    Returns
    -------
    float
        log P(seq) = sum of log(s[u]) where s[u] > 0
    """
    total = 0.0
    for u in range(s.shape[0]):
        if s[u] > 0.0:
            total += math.log(s[u])
    return total


# ---------------------------------------------------------------------------
# 4. Expected sufficient statistics (single pass)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def expected_counts_jit(a, e, seq, f, b, s, a0):
    """Compute expected sufficient statistics for EM in a single pass over L.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix
    seq : (L,) observation sequence
    f : (L, n) forward probabilities
    b : (L, n) backward probabilities
    s : (L,) scaling factors
    a0 : (n,) initial distribution

    Returns
    -------
    A : (n, n) expected transition counts
    E : (m+1, n) expected emission counts
    A0 : (n,) expected initial state counts
    """
    n = a.shape[0]
    m_plus_1 = e.shape[0]
    L = seq.shape[0]

    # Precompute ae[sym, k, l] = a[k, l] * e[sym, l]
    ae = np.empty((m_plus_1, n, n), dtype=np.float64)
    for sym in range(m_plus_1):
        for k in range(n):
            for l in range(n):
                ae[sym, k, l] = a[k, l] * e[sym, l]

    # Initialize with HMM_TINY to avoid log(0)
    A = np.empty((n, n), dtype=np.float64)
    for k in range(n):
        for l in range(n):
            A[k, l] = HMM_TINY

    E = np.empty((m_plus_1, n), dtype=np.float64)
    for bb in range(m_plus_1):
        for k in range(n):
            E[bb, k] = HMM_TINY

    # Single pass: u = 0..L-2
    for u in range(L - 1):
        sym_next = seq[u + 1]
        sym_cur = seq[u]
        su = s[u]

        for k in range(n):
            fuk = f[u, k]
            # Transition counts: A[k, l] += f[u, k] * ae[seq[u+1], k, l] * b[u+1, l]
            for l in range(n):
                A[k, l] += fuk * ae[sym_next, k, l] * b[u + 1, l]

            # Emission counts: E[seq[u], k] += f[u, k] * b[u, k] * s[u]
            E[sym_cur, k] += fuk * b[u, k] * su

    # A0[k] = a0[k] * e[seq[0], k] * b[0, k]
    A0 = np.empty(n, dtype=np.float64)
    sym0 = seq[0]
    for k in range(n):
        A0[k] = a0[k] * e[sym0, k] * b[0, k]

    return A, E, A0


# ---------------------------------------------------------------------------
# 5. EM Q-function
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def q_function_jit(a, e, A, E, Q0):
    """Compute the EM Q-function value.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix (only rows 0..m-1 used)
    A : (n, n) expected transition counts
    E : (m+1, n) expected emission counts
    Q0 : float, baseline Q value

    Returns
    -------
    float
        Q(params) - Q0
    """
    n = a.shape[0]
    m = e.shape[0] - 1  # exclude missing-data row

    # Check positivity
    for bb in range(m):
        for k in range(n):
            if e[bb, k] <= 0.0:
                return -1e300
    for k in range(n):
        for l in range(n):
            if a[k, l] <= 0.0:
                return -1e300

    # Emission term: sum over b=0..m-1, k=0..n-1 of E[b,k] * log(e[b,k])
    total = 0.0
    for bb in range(m):
        for k in range(n):
            total += E[bb, k] * math.log(e[bb, k])

    # Transition term: sum of A[k,l] * log(a[k,l])
    for k in range(n):
        for l in range(n):
            total += A[k, l] * math.log(a[k, l])

    return total - Q0


# ---------------------------------------------------------------------------
# 6. Q0 baseline from counts
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def q0_from_counts_jit(A, E, m):
    """Compute the Q0 baseline from expected counts.

    Parameters
    ----------
    A : (n, n) expected transition counts
    E : (m+1, n) expected emission counts
    m : int, number of real symbols (excluding missing)

    Returns
    -------
    float
        Q0 value
    """
    n = A.shape[0]
    n_e = E.shape[1]

    # Emission entropy: sum_k sum_b E[b,k] * log(E[b,k] / col_sum)
    emission_entropy = 0.0
    for k in range(n_e):
        col_sum = 0.0
        for bb in range(m):
            col_sum += E[bb, k]
        if col_sum <= 0.0:
            col_sum = 1.0
        for bb in range(m):
            if E[bb, k] > 0.0:
                emission_entropy += E[bb, k] * math.log(E[bb, k] / col_sum)

    # Transition entropy: sum_k sum_l A[k,l] * log(A[k,l] / row_sum)
    transition_entropy = 0.0
    for k in range(n):
        row_sum = 0.0
        for l in range(n):
            row_sum += A[k, l]
        if row_sum <= 0.0:
            row_sum = 1.0
        for l in range(n):
            if A[k, l] > 0.0:
                transition_entropy += A[k, l] * math.log(A[k, l] / row_sum)

    return emission_entropy + transition_entropy


# ---------------------------------------------------------------------------
# 7. Time interval computation
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def compute_time_intervals_jit(n, max_t, alpha):
    """Compute time boundaries t_0, ..., t_{n+1}.

    Parameters
    ----------
    n : int
        Number of intervals - 1 (n+1 states, n+2 boundaries).
    max_t : float
        Maximum coalescent time.
    alpha : float
        Controls spacing of intervals near t=0.

    Returns
    -------
    t : (n+2,) array of time boundaries. t[n+1] = PSMC_T_INF.
    """
    t = np.empty(n + 2, dtype=np.float64)
    beta = math.log(1.0 + max_t / alpha) / n
    for k in range(n):
        t[k] = alpha * (math.exp(beta * k) - 1.0)
    t[n] = max_t
    t[n + 1] = PSMC_T_INF
    return t


# ---------------------------------------------------------------------------
# 8. Coalescent params -> HMM params
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def compute_hmm_params_jit(params, par_map, n, t, divergence):
    """Map population parameters to HMM transition/emission matrices.

    Parameters
    ----------
    params : (n_params,) free parameters [theta, rho, max_t, lam_0, ..., (dt)]
    par_map : (n+1,) int array mapping state -> free lambda index
    n : int, n value (n+1 states)
    t : (n+2,) time boundaries
    divergence : bool

    Returns
    -------
    a_mat : (n+1, n+1) transition matrix
    e_mat : (3, n+1) emission matrix
    sigma : (n+1,) initial distribution
    C_pi : float
    C_sigma : float
    """
    n_states = n + 1
    theta = params[0]
    rho = params[1]

    # Lambda for each state
    lam = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        lam[k] = params[par_map[k] + PSMC_N_PARAMS]

    # Divergence time
    if divergence:
        dt = params[params.shape[0] - 1]
        if dt < 0.0:
            dt = 0.0
    else:
        dt = 0.0

    # Interval widths: tau[k] = t[k+1] - t[k]
    tau = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        tau[k] = t[k + 1] - t[k]

    # log survival per interval
    log_surv = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        log_surv[k] = -tau[k] / lam[k]

    # Alpha: survival probabilities via cumulative product
    alpha = np.empty(n_states + 1, dtype=np.float64)
    alpha[0] = 1.0
    cumsum_log = 0.0
    for k in range(n_states - 1):
        cumsum_log += log_surv[k]
        alpha[k + 1] = math.exp(cumsum_log)
    alpha[n_states] = 0.0

    # Safe alpha for divisions
    alpha_safe = np.empty(n_states + 1, dtype=np.float64)
    for k in range(n_states + 1):
        alpha_safe[k] = alpha[k] if alpha[k] > 1e-300 else 1e-300

    # Beta via cumulative sum
    beta = np.empty(n_states, dtype=np.float64)
    beta[0] = 0.0
    cumsum_beta = 0.0
    for k in range(n_states - 1):
        cumsum_beta += lam[k] * (1.0 / alpha_safe[k + 1] - 1.0 / alpha_safe[k])
        beta[k + 1] = cumsum_beta

    # q_aux[l] for l=0..n-1
    q_aux = np.empty(n, dtype=np.float64)
    for l in range(n):
        q_aux[l] = ((alpha[l] - alpha[l + 1])
                     * (beta[l] - lam[l] / alpha_safe[l])
                     + tau[l])

    # ak1[k] = alpha[k] - alpha[k+1]
    ak1 = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        ak1[k] = alpha[k] - alpha[k + 1]

    # Normalization constant C_pi
    C_pi = 0.0
    for k in range(n_states):
        C_pi += lam[k] * ak1[k]

    C_sigma = 1.0 / (C_pi * rho) + 0.5

    # Cumulative sum_t
    sum_t = np.empty(n_states, dtype=np.float64)
    sum_t[0] = 0.0
    for k in range(n_states - 1):
        sum_t[k + 1] = sum_t[k] + tau[k]

    # cpik, pik, sigma
    cpik = np.empty(n_states, dtype=np.float64)
    pik = np.empty(n_states, dtype=np.float64)
    sigma = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        cpik[k] = ak1[k] * (sum_t[k] + lam[k]) - alpha[k + 1] * tau[k]
        pik[k] = cpik[k] / C_pi
        sigma[k] = (ak1[k] / (C_pi * rho) + pik[k] / 2.0) / C_sigma

    # avg_t with fallback
    avg_t = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        denom = C_sigma * sigma[k]
        if denom > 0.0:
            ratio = pik[k] / denom
        else:
            ratio = 2.0  # triggers fallback

        if ratio > 0.0 and ratio < 1.0:
            arg = 1.0 - ratio
            if arg < 1e-300:
                arg = 1e-300
            val = -math.log(arg) / rho
        else:
            val = math.nan

        if math.isnan(val) or val < sum_t[k] or val > sum_t[k] + tau[k]:
            # Fallback
            if ak1[k] > 0.0:
                val = sum_t[k] + lam[k] - tau[k] * alpha[k + 1] / ak1[k]
            else:
                val = sum_t[k]
        avg_t[k] = val

    # --- Build transition matrix ---
    a_mat = np.zeros((n_states, n_states), dtype=np.float64)

    cpik_safe = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        cpik_safe[k] = cpik[k] if cpik[k] > 1e-300 else 1e-300

    # ratio_k[k] = ak1[k] / cpik_safe[k]
    ratio_k = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        ratio_k[k] = ak1[k] / cpik_safe[k]

    # ratio_k2[k] = q_aux[k] / cpik_safe[k] for k < n
    ratio_k2 = np.empty(n, dtype=np.float64)
    for k in range(n):
        ratio_k2[k] = q_aux[k] / cpik_safe[k]

    for k in range(n_states):
        if cpik[k] > 0.0:
            # Lower triangle: l < k
            for l in range(k):
                a_mat[k, l] = ratio_k[k] * q_aux[l]

            # Diagonal: l == k
            ak_s = alpha_safe[k]
            a_mat[k, k] = ((ak1[k] * ak1[k] * (beta[k] - lam[k] / ak_s)
                            + 2.0 * lam[k] * ak1[k]
                            - 2.0 * alpha[k + 1] * tau[k])
                           / cpik_safe[k])

            # Upper triangle: l > k (only if k < n)
            if k < n:
                for l in range(k + 1, n_states):
                    a_mat[k, l] = ak1[l] * ratio_k2[k]
        else:
            a_mat[k, k] = 1.0

    # Convert q -> transition probabilities
    # tmp_p[k] = pik[k] / (C_sigma * sigma[k])
    tmp_p = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        denom = C_sigma * sigma[k]
        if denom > 0.0:
            tmp_p[k] = pik[k] / denom
        else:
            tmp_p[k] = 0.0

    # a_mat[k,:] *= tmp_p[k]; a_mat[k,k] += 1 - tmp_p[k]
    for k in range(n_states):
        for l in range(n_states):
            a_mat[k, l] *= tmp_p[k]
        a_mat[k, k] += 1.0 - tmp_p[k]

    # --- Emission matrix ---
    e_mat = np.empty((3, n_states), dtype=np.float64)
    for k in range(n_states):
        e_mat[0, k] = math.exp(-theta * (avg_t[k] + dt))
        e_mat[1, k] = 1.0 - e_mat[0, k]
        e_mat[2, k] = 1.0  # missing data

    sigma_out = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        sigma_out[k] = sigma[k]

    return a_mat, e_mat, sigma_out, C_pi, C_sigma


# ---------------------------------------------------------------------------
# 9. Negative Q-function evaluator (for optimizer)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _eval_neg_Q(x, par_map, n, A_sum, E_sum, Q0_val, divergence):
    """Evaluate negative Q-function for optimization.

    Parameters
    ----------
    x : (n_params,) current parameter vector
    par_map : (n+1,) int array
    n : int
    A_sum : (n+1, n+1) aggregated transition counts
    E_sum : (3, n+1) aggregated emission counts
    Q0_val : float
    divergence : bool

    Returns
    -------
    float
        -Q(|x|)
    """
    n_params = x.shape[0]
    x_abs = np.empty(n_params, dtype=np.float64)
    for i in range(n_params):
        x_abs[i] = abs(x[i])

    t = compute_time_intervals_jit(n, x_abs[2], 0.1)
    a, e, _, _, _ = compute_hmm_params_jit(x_abs, par_map, n, t, divergence)
    return -q_function_jit(a, e, A_sum, E_sum, Q0_val)


# ---------------------------------------------------------------------------
# 10. Hooke-Jeeves optimizer (port of kmin_hj)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def kmin_hj_jit(params, par_map, n, A_sum, E_sum, Q0_val, divergence):
    """Hooke-Jeeves nonlinear minimization of -Q (port of Heng Li's kmin_hj).

    Parameters
    ----------
    params : (n_params,) initial parameter vector
    par_map : (n+1,) int array
    n : int
    A_sum : (n+1, n+1) aggregated transition counts
    E_sum : (3, n+1) aggregated emission counts
    Q0_val : float
    divergence : bool

    Returns
    -------
    optimized_params : (n_params,) optimized parameter vector (absolute values)
    Q_val : float, Q-function value at optimum
    """
    r = 0.5
    eps = 1e-7
    max_calls = 50000

    n_dim = params.shape[0]
    x = np.empty(n_dim, dtype=np.float64)
    for i in range(n_dim):
        x[i] = params[i]

    x1 = np.empty(n_dim, dtype=np.float64)
    x_old = np.empty(n_dim, dtype=np.float64)
    dx = np.empty(n_dim, dtype=np.float64)

    for k in range(n_dim):
        dx[k] = abs(x[k]) * r
        if dx[k] == 0.0:
            dx[k] = r

    fx = _eval_neg_Q(x, par_map, n, A_sum, E_sum, Q0_val, divergence)
    n_calls = 1
    radius = r

    while True:
        # x1 = x.copy()
        for k in range(n_dim):
            x1[k] = x[k]

        # Exploratory moves from x1
        fx1 = fx
        for k in range(n_dim):
            x1[k] += dx[k]
            ft = _eval_neg_Q(x1, par_map, n, A_sum, E_sum, Q0_val, divergence)
            n_calls += 1
            if ft < fx1:
                fx1 = ft
            else:
                dx[k] = -dx[k]
                x1[k] += 2.0 * dx[k]
                ft = _eval_neg_Q(x1, par_map, n, A_sum, E_sum, Q0_val, divergence)
                n_calls += 1
                if ft < fx1:
                    fx1 = ft
                else:
                    x1[k] -= dx[k]

        # Pattern moves
        while fx1 < fx:
            for k in range(n_dim):
                t_val = x[k]
                if x1[k] > x[k]:
                    dx[k] = abs(dx[k])
                else:
                    dx[k] = -abs(dx[k])
                x_old[k] = x[k]
                x[k] = x1[k]
                x1[k] = 2.0 * x1[k] - t_val
            fx = fx1

            if n_calls >= max_calls:
                break

            fx1 = _eval_neg_Q(x1, par_map, n, A_sum, E_sum, Q0_val, divergence)
            n_calls += 1

            # Explore from pattern-moved point
            for k in range(n_dim):
                x1[k] += dx[k]
                ft = _eval_neg_Q(x1, par_map, n, A_sum, E_sum, Q0_val, divergence)
                n_calls += 1
                if ft < fx1:
                    fx1 = ft
                else:
                    dx[k] = -dx[k]
                    x1[k] += 2.0 * dx[k]
                    ft = _eval_neg_Q(x1, par_map, n, A_sum, E_sum, Q0_val, divergence)
                    n_calls += 1
                    if ft < fx1:
                        fx1 = ft
                    else:
                        x1[k] -= dx[k]

            if fx1 >= fx:
                break

            # Check convergence
            converged = True
            for k in range(n_dim):
                if abs(x1[k] - x[k]) > 0.5 * abs(dx[k]):
                    converged = False
                    break
            if converged:
                break

        if radius >= eps:
            if n_calls >= max_calls:
                break
            radius *= r
            for k in range(n_dim):
                dx[k] *= r
        else:
            break

    # Return absolute values and the Q value (negate back)
    result = np.empty(n_dim, dtype=np.float64)
    for i in range(n_dim):
        result[i] = abs(x[i])

    return result, -fx
