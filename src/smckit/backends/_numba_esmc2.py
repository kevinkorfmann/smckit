"""Numba JIT backend for eSMC2 HMM operations.

Implements the HMM construction from Sellinger et al. (2020) for
ecological SMC with dormancy (germination rate β) and self-fertilization
(selfing rate σ). The coalescent rate is scaled by 2β²/(2−σ) and
recombination by 2β(1−σ)/(2−σ).

Reference: Sellinger et al., "Inference of past demography, dormancy and
self-fertilization rates from whole genome sequence data", PLoS Genetics, 2020.
"""

from __future__ import annotations

import math

import numba
import numpy as np

HMM_TINY = 1e-25


# ---------------------------------------------------------------------------
# 1. Time discretization
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_build_time_boundaries(n, beta, sigma):
    """Build time boundaries Tc for eSMC2 hidden states.

    Uses the default eSMC2 time window (Big_Window=0):
        Tc[k] = -0.5 * (2 - σ) * log(1 - k/n) / β²

    Parameters
    ----------
    n : int
        Number of hidden states.
    beta : float
        Germination rate (0 < β ≤ 1). β = 1 means no dormancy.
    sigma : float
        Self-fertilization rate (0 ≤ σ < 1). σ = 0 means random mating.

    Returns
    -------
    Tc : (n,) float64
        Time boundaries for hidden states.
    """
    Tc = np.empty(n, dtype=np.float64)
    b2 = beta * beta
    for k in range(n):
        if k == 0:
            Tc[k] = 0.0
        else:
            Tc[k] = -0.5 * (2.0 - sigma) * math.log(1.0 - k / n) / b2
    return Tc


# ---------------------------------------------------------------------------
# 2. Expected coalescent times per state
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_expected_times(Tc, Xi, beta, sigma):
    """Compute expected coalescent times for each hidden state.

    Parameters
    ----------
    Tc : (n,) float64
        Time boundaries.
    Xi : (n,) float64
        Relative population sizes per state.
    beta : float
        Germination rate.
    sigma : float
        Self-fertilization rate.

    Returns
    -------
    t : (n,) float64
        Expected coalescent times.
    """
    n = Tc.shape[0]
    t = np.empty(n, dtype=np.float64)
    b2 = beta * beta
    coal_scale = (2.0 - sigma) / (b2 * 2.0)

    for k in range(n):
        if k < n - 1:
            D_k = Tc[k + 1] - Tc[k]
            rate = b2 * 2.0 / (Xi[k] * (2.0 - sigma))
            exp_neg = math.exp(-rate * D_k)
            t[k] = Xi[k] * coal_scale + (Tc[k] - Tc[k + 1] * exp_neg) / (1.0 - exp_neg)
        else:
            t[k] = Xi[k] * coal_scale + Tc[k]
    return t


# ---------------------------------------------------------------------------
# 3. Equilibrium probabilities
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_equilibrium_probs(Tc, Xi, beta, sigma):
    """Compute equilibrium (stationary) probabilities for each hidden state.

    Parameters
    ----------
    Tc : (n,) float64
        Time boundaries.
    Xi : (n,) float64
        Relative population sizes per state.
    beta : float
        Germination rate.
    sigma : float
        Self-fertilization rate.

    Returns
    -------
    q : (n,) float64
        Equilibrium probabilities (sums to 1).
    """
    n = Tc.shape[0]
    q = np.empty(n, dtype=np.float64)
    b2 = beta * beta

    # D[k] = Tc[k+1] - Tc[k] for k < n-1
    D = np.empty(n - 1, dtype=np.float64)
    for k in range(n - 1):
        D[k] = Tc[k + 1] - Tc[k]

    # q[0] = 1 - exp(-D[0] * 2β² / (Xi[0] * (2 - σ)))
    rate0 = 2.0 * b2 / (Xi[0] * (2.0 - sigma))
    q[0] = 1.0 - math.exp(-D[0] * rate0)

    # q[k] = exp(-2β²/(2-σ) * sum(D[j]/Xi[j], j=0..k-1)) * (1 - exp(-D[k]*rate_k))
    for k in range(1, n - 1):
        cum_sum = 0.0
        for j in range(k):
            cum_sum += D[j] / Xi[j]
        cum_sum *= 2.0 * b2 / (2.0 - sigma)
        rate_k = 2.0 * b2 / (Xi[k] * (2.0 - sigma))
        q[k] = math.exp(-cum_sum) * (1.0 - math.exp(-D[k] * rate_k))

    # q[n-1] = exp(-2β²/(2-σ) * sum(D[j]/Xi[j], j=0..n-2))
    if n > 1:
        cum_sum = 0.0
        for j in range(n - 1):
            cum_sum += D[j] / Xi[j]
        cum_sum *= 2.0 * b2 / (2.0 - sigma)
        q[n - 1] = math.exp(-cum_sum)

    # Normalize to ensure sum = 1
    total = 0.0
    for k in range(n):
        total += q[k]
    if total > 0.0:
        for k in range(n):
            q[k] /= total

    return q


# ---------------------------------------------------------------------------
# 4. Emission matrix
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_build_emission_matrix(mu, mu_b, Tc, t, beta, n):
    """Build emission matrix for eSMC2.

    The emission probability accounts for mutations accumulating along
    the branch, with a different mutation rate in the seed bank.

    Parameters
    ----------
    mu : float
        Mutation rate per bp per 2*Ne generations (= theta / (2*L)).
    mu_b : float
        Ratio of mutation rate in seed bank over active mutation rate.
        mu_b = 1 means same rate in seed bank (standard case).
    Tc : (n,) float64
        Time boundaries.
    t : (n,) float64
        Expected coalescent times.
    beta : float
        Germination rate.
    n : int
        Number of hidden states.

    Returns
    -------
    e : (3, n) float64
        Emission matrix: e[0, k] = P(homozygous | state k),
        e[1, k] = P(heterozygous | state k), e[2, k] = 1 (missing).
    """
    e = np.ones((3, n), dtype=np.float64)

    # Effective mutation rate incorporating seed bank
    # mu_eff = beta + (1 - beta) * mu_b
    for k in range(n):
        if k == 0:
            mu_eff = beta + (1.0 - beta) * mu_b
            e[0, k] = math.exp(-2.0 * mu * mu_eff * t[k])
        else:
            # Sum contributions from each time interval
            branch_length = 0.0
            for j in range(k):
                mu_eff_j = beta + (1.0 - beta) * mu_b
                D_j = Tc[j + 1] - Tc[j]
                branch_length += mu_eff_j * D_j

            mu_eff_k = beta + (1.0 - beta) * mu_b
            branch_length += mu_eff_k * (t[k] - Tc[k])

            e[0, k] = math.exp(-2.0 * mu * branch_length)

        e[1, k] = 1.0 - e[0, k]
        # e[2, k] = 1.0  (already initialized)

    return e


# ---------------------------------------------------------------------------
# 5. Transition matrix
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L):
    """Build transition matrix Q for eSMC2.

    This implements the full SMC transition matrix from Sellinger et al. (2020)
    with dormancy and selfing modifications to the coalescent process.

    Parameters
    ----------
    Tc : (n,) float64
        Time boundaries.
    Xi : (n,) float64
        Relative population sizes per state.
    t : (n,) float64
        Expected coalescent times.
    beta : float
        Germination rate.
    sigma : float
        Self-fertilization rate.
    rho : float
        Recombination rate per sequence (= rho_per_bp * 2 * L).
    L : int
        Sequence length in base pairs.

    Returns
    -------
    Q : (n, n) float64
        Transition matrix in upstream orientation:
        ``Q[next_state, current_state]`` with columns summing to 1.
    """
    n = Tc.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)
    b2 = beta * beta

    r = rho / (2.0 * (L - 1))

    D = np.empty(n - 1, dtype=np.float64)
    for k in range(n - 1):
        D[k] = Tc[k + 1] - Tc[k]

    rec_scale = beta * 2.0 * (1.0 - sigma) / (2.0 - sigma)

    # --- Row i = 0: first state ---
    if n > 1:
        for j in range(1, n):
            rec_prob = 1.0 - math.exp(-2.0 * r * t[j] * rec_scale)
            # Integral term for state 0
            rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
            integral = D[0] - (1.0 - math.exp(-D[0] * rate_0)) / rate_0
            Q[0, j] = rec_prob * (1.0 / (2.0 * t[j])) * integral

    # --- Rows i = 1 to n-2: interior states ---
    for i in range(1, n - 1):
        # Forward uses 4β² rate; backward uses 2β² rate
        fwd_coal_i = 1.0 - math.exp(-D[i] * b2 * 4.0 / (Xi[i] * (2.0 - sigma)))
        bwd_coal_i = 1.0 - math.exp(-D[i] * b2 * 2.0 / (Xi[i] * (2.0 - sigma)))

        # Forward transitions: i → j where j > i
        # Compute the "truc" integral term
        truc_sum = 0.0
        for eta in range(i - 1):
            rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
            term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
            # Multiply by exp of cumulative rates from eta+1 to i-1
            cum_exp = 0.0
            for kk in range(eta + 1, i):
                cum_exp += D[kk] / (Xi[kk] * (2.0 - sigma))
            cum_exp *= 4.0 * b2
            truc_sum += term * math.exp(-cum_exp)

        # Last eta = i-1: integral without exp decay factor (always present)
        eta = i - 1
        rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
        truc_sum += (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta

        rate_i = 4.0 * b2 / (Xi[i] * (2.0 - sigma))
        diag_integral = D[i] - (1.0 - math.exp(-D[i] * rate_i)) / rate_i

        forward_base = truc_sum * fwd_coal_i + diag_integral

        for j in range(i + 1, n):
            rec_prob = 1.0 - math.exp(-2.0 * r * t[j] * rec_scale)
            Q[i, j] = rec_prob * (1.0 / (2.0 * t[j])) * forward_base

        # Backward transitions: i → j where j < i
        for gamma in range(i):
            # Compute exp_truc[gamma]
            if gamma < i - 1:
                cum_rate = 0.0
                for kk in range(gamma + 1, i):
                    cum_rate += D[kk] / (Xi[kk] * (2.0 - sigma))
                cum_rate *= b2 * 2.0
                cum_rate += b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                exp_truc_g = math.exp(-cum_rate)
            else:
                arg = -b2 * 2.0 * (Tc[gamma + 1] - t[gamma])
                arg /= Xi[gamma] * (2.0 - sigma)
                exp_truc_g = math.exp(arg)

            # Compute truc[gamma]
            if gamma == 0:
                rate_g = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                truc_g = (1.0 - math.exp((Tc[0] - t[0]) * rate_g)) / rate_g
            elif gamma == 1:
                rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                sub = (1.0 - math.exp(-D[0] * rate_0)) / rate_0
                sub *= math.exp(-rate_g * (t[gamma] - Tc[gamma]))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g
            else:
                sub = 0.0
                for eta in range(gamma - 1):
                    rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                    term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
                    cum_exp = 0.0
                    for kk in range(eta + 1, gamma):
                        cum_exp += D[kk] / (Xi[kk] * (2.0 - sigma))
                    cum_exp *= 4.0 * b2
                    cum_exp += 4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma))
                    sub += term * math.exp(-cum_exp)

                eta = gamma - 1
                rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                sub += (
                    (1.0 - math.exp(-D[eta] * rate_eta))
                    / rate_eta
                    * math.exp(-4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma)))
                )

                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g

            rec_prob_g = 1.0 - math.exp(-2.0 * r * t[gamma] * rec_scale)
            Q[i, gamma] = rec_prob_g * (1.0 / t[gamma]) * bwd_coal_i * exp_truc_g * truc_g

    # --- Row n-1: last state ---
    if n > 1:
        for gamma in range(n - 1):
            if gamma < n - 2:
                cum_rate = 0.0
                for kk in range(gamma + 1, n - 1):
                    cum_rate += D[kk] / (Xi[kk] * (2.0 - sigma))
                cum_rate *= b2 * 2.0
                cum_rate += b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                exp_truc_g = math.exp(-cum_rate)
            else:
                arg = -b2 * 2.0 * (Tc[gamma + 1] - t[gamma])
                arg /= Xi[gamma] * (2.0 - sigma)
                exp_truc_g = math.exp(arg)

            if gamma == 0:
                rate_g = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                truc_g = (1.0 - math.exp((Tc[0] - t[0]) * rate_g)) / rate_g
            elif gamma == 1:
                rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                sub = (1.0 - math.exp(-D[0] * rate_0)) / rate_0
                sub *= math.exp(-rate_g * (t[gamma] - Tc[gamma]))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g
            else:
                sub = 0.0
                for eta in range(gamma - 1):
                    rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                    term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
                    cum_exp = 0.0
                    for kk in range(eta + 1, gamma):
                        cum_exp += D[kk] / (Xi[kk] * (2.0 - sigma))
                    cum_exp *= 4.0 * b2
                    cum_exp += 4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma))
                    sub += term * math.exp(-cum_exp)

                eta = gamma - 1
                rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                sub += (
                    (1.0 - math.exp(-D[eta] * rate_eta))
                    / rate_eta
                    * math.exp(-4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma)))
                )

                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g

            rec_prob_g = 1.0 - math.exp(-2.0 * r * t[gamma] * rec_scale)
            Q[n - 1, gamma] = rec_prob_g * (1.0 / t[gamma]) * exp_truc_g * truc_g

    # Upstream build_HMM_matrix normalizes columns: Q[next, current].
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            if i != j:
                col_sum += Q[i, j]
        Q[j, j] = 1.0 - col_sum

    return Q


# ---------------------------------------------------------------------------
# 6. Complete HMM construction
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _esmc2_build_hmm_impl(
    n,
    Xi,
    beta,
    sigma,
    rho,
    mu,
    mu_b,
    L,
    beta_hidden,
    sigma_hidden,
):
    """Build complete HMM matrices for eSMC2 with split hidden-state params.

    Parameters
    ----------
    n : int
        Number of hidden states.
    Xi : (n,) float64
        Relative population sizes per state.
    beta : float
        Germination rate.
    sigma : float
        Self-fertilization rate.
    rho : float
        Recombination rate per sequence.
    mu : float
        Mutation rate per bp per 2*Ne.
    mu_b : float
        Seed bank mutation rate ratio.
    L : int
        Sequence length.

    Returns
    -------
    Q : (n, n) transition matrix with ``Q[next, current]``
    q : (n,) equilibrium probabilities
    t : (n,) expected coalescent times
    Tc : (n,) time boundaries
    e : (3, n) emission matrix
    """
    Tc = esmc2_build_time_boundaries(n, beta_hidden, sigma_hidden)
    t = esmc2_expected_times(Tc, Xi, beta, sigma)
    q = esmc2_equilibrium_probs(Tc, Xi, beta, sigma)
    Q = esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)
    e = esmc2_build_emission_matrix(mu, mu_b, Tc, t, beta, n)
    return Q, q, t, Tc, e


def esmc2_build_hmm(
    n,
    Xi,
    beta,
    sigma,
    rho,
    mu,
    mu_b,
    L,
    beta_hidden=None,
    sigma_hidden=None,
):
    """Build complete HMM matrices for eSMC2.

    When ``beta_hidden`` / ``sigma_hidden`` are supplied, the hidden-state time
    grid matches upstream eSMC2's Baum-Welch semantics: hidden states are built
    from the run's fixed ``Beta`` / ``Self`` values, while transition and
    emission terms still use the candidate ``beta`` / ``sigma``.
    """
    if beta_hidden is None:
        beta_hidden = beta
    if sigma_hidden is None:
        sigma_hidden = sigma
    return _esmc2_build_hmm_impl(
        n,
        Xi,
        beta,
        sigma,
        rho,
        mu,
        mu_b,
        L,
        beta_hidden,
        sigma_hidden,
    )


# ---------------------------------------------------------------------------
# 7. Negative log-likelihood (for optimization)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_forward_loglik(Q, e, q, seq):
    """Compute log-likelihood of a sequence using scaled forward algorithm.

    Parameters
    ----------
    Q : (n, n) transition matrix
    e : (3, n) emission matrix
    q : (n,) initial distribution
    seq : (L,) int8 observation sequence (0=hom, 1=het, 2=missing)

    Returns
    -------
    loglik : float
        Total log-likelihood.
    """
    n = Q.shape[0]
    L_seq = seq.shape[0]

    f_prev = np.empty(n, dtype=np.float64)
    f_curr = np.empty(n, dtype=np.float64)

    # Initialize
    sym = seq[0]
    total = 0.0
    for k in range(n):
        f_prev[k] = e[sym, k] * q[k]
        total += f_prev[k]
    loglik = math.log(total + HMM_TINY)
    for k in range(n):
        f_prev[k] /= total

    # Forward pass
    for u in range(1, L_seq):
        sym = seq[u]
        total = 0.0
        for k in range(n):
            s = 0.0
            for j in range(n):
                s += Q[k, j] * f_prev[j]
            f_curr[k] = e[sym, k] * s
            total += f_curr[k]
        loglik += math.log(total + HMM_TINY)
        for k in range(n):
            f_prev[k] = f_curr[k] / total

    return loglik


# ---------------------------------------------------------------------------
# 8. Forward algorithm (returns full forward matrix + scaling)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_forward(Q, e, q, seq):
    """Scaled forward algorithm returning full matrix.

    Parameters
    ----------
    Q : (n, n) transition matrix with ``Q[next, current]``
    e : (3, n) emission matrix
    q : (n,) initial distribution
    seq : (L,) int8 observation sequence

    Returns
    -------
    f : (L, n) scaled forward probabilities
    s : (L,) scaling factors (raw, not log)
    """
    n = Q.shape[0]
    L_seq = seq.shape[0]
    f = np.zeros((L_seq, n), dtype=np.float64)
    s = np.ones(L_seq, dtype=np.float64)

    sym = seq[0]
    total = 0.0
    for k in range(n):
        f[0, k] = e[sym, k] * q[k]
        total += f[0, k]
    s[0] = total
    if total > 0.0:
        for k in range(n):
            f[0, k] /= total

    for u in range(1, L_seq):
        sym = seq[u]
        total = 0.0
        for k in range(n):
            v = 0.0
            for j in range(n):
                v += Q[k, j] * f[u - 1, j]
            f[u, k] = e[sym, k] * v
            total += f[u, k]
        s[u] = total
        if total > 0.0:
            for k in range(n):
                f[u, k] /= total

    return f, s


# ---------------------------------------------------------------------------
# 9. Backward algorithm
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_backward(Q, e, seq, s):
    """Scaled backward algorithm.

    Parameters
    ----------
    Q : (n, n) transition matrix with ``Q[next, current]``
    e : (3, n) emission matrix
    seq : (L,) int8 observation sequence
    s : (L,) scaling factors from forward algorithm

    Returns
    -------
    b : (L, n) scaled backward probabilities
    """
    n = Q.shape[0]
    L_seq = seq.shape[0]
    b = np.zeros((L_seq, n), dtype=np.float64)

    for k in range(n):
        b[L_seq - 1, k] = 1.0

    for u in range(L_seq - 2, -1, -1):
        sym = seq[u + 1]
        for k in range(n):
            v = 0.0
            for j in range(n):
                v += Q[j, k] * e[sym, j] * b[u + 1, j]
            b[u, k] = v / s[u + 1]

    return b


# ---------------------------------------------------------------------------
# 10. Expected counts from forward-backward
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def esmc2_expected_counts(Q, e, seq, f, b, s, q):
    """Compute expected transition and emission counts.

    Parameters
    ----------
    Q : (n, n) transition matrix with ``Q[next, current]``
    e : (3, n) emission matrix
    seq : (L,) int8 observation sequence
    f : (L, n) forward probabilities
    b : (L, n) backward probabilities
    s : (L,) scaling factors
    q : (n,) initial distribution

    Returns
    -------
    N : (n, n) expected transition counts
    M : (3, n) expected emission counts
    """
    n = Q.shape[0]
    L_seq = seq.shape[0]
    N_counts = np.zeros((n, n), dtype=np.float64)
    M_counts = np.zeros((3, n), dtype=np.float64)

    # Expected transitions: sum_u P(z_u=i, z_{u+1}=j | obs)
    for u in range(L_seq - 1):
        sym_next = seq[u + 1]
        for i in range(n):
            for j in range(n):
                val = f[u, i] * Q[j, i] * e[sym_next, j] * b[u + 1, j] / s[u + 1]
                N_counts[i, j] += val

    # Expected emissions: sum_u P(z_u=k | obs)
    for u in range(L_seq):
        sym = seq[u]
        # Posterior gamma[u, k] = f[u, k] * b[u, k]
        total = 0.0
        for k in range(n):
            total += f[u, k] * b[u, k]
        if total > 0.0:
            for k in range(n):
                M_counts[sym, k] += f[u, k] * b[u, k] / total

    return N_counts, M_counts
