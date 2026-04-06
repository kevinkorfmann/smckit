"""NumPy CPU reference kernels for HMM operations.

Fully vectorized — no Python loops over sequence length L or state count n
except where sequential dependencies require it (forward/backward along L).

Conventions (matching khmm.c):
    n  — number of hidden states
    m  — number of observation symbols (excluding missing=m)
    L  — sequence length
    a  — transition matrix, shape (n, n), a[k, l] = P(next=l | cur=k)
    e  — emission matrix, shape (m+1, n), e[b, k] = P(obs=b | state=k);
         e[m, :] = 1 (missing data)
    a0 — initial distribution, shape (n,)
    seq — observation sequence, shape (L,), values in 0..m (int8)
"""

from __future__ import annotations

import numpy as np

HMM_TINY = 1e-25


def forward(
    a: np.ndarray,
    e: np.ndarray,
    a0: np.ndarray,
    seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward algorithm.

    The loop over L is inherently sequential (each step depends on the previous).
    The inner n-state operations are fully vectorized as matrix-vector products.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix
    a0 : (n,) initial state distribution
    seq : (L,) observation sequence (int8, 0-indexed, m=missing)

    Returns
    -------
    f : (L, n) scaled forward probabilities
    s : (L,) scaling factors
    """
    n = a.shape[0]
    L = seq.shape[0]
    f = np.zeros((L, n), dtype=np.float64)
    s = np.ones(L, dtype=np.float64)

    # Precompute: emission vectors for entire sequence and transposed transition
    e_seq = e[seq]  # (L, n) — emission for each position, one lookup
    at = a.T.copy()  # (n, n) — transposed for cache-friendly matvec

    # f[0]
    f[0] = a0 * e_seq[0]
    s[0] = f[0].sum()
    if s[0] > 0:
        f[0] /= s[0]

    # f[u] for u = 1..L-1  (sequential dependency: f[u] depends on f[u-1])
    for u in range(1, L):
        f[u] = e_seq[u] * (at @ f[u - 1])
        s[u] = f[u].sum()
        if s[u] > 0:
            f[u] /= s[u]

    return f, s


def backward(
    a: np.ndarray,
    e: np.ndarray,
    seq: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
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
    L = seq.shape[0]
    b = np.zeros((L, n), dtype=np.float64)

    # Precompute ae[b, k, l] = a[k, l] * e[b, l] for each symbol b
    # Vectorized: broadcast (m+1, 1, n) * (1, n, n) → (m+1, n, n)
    ae = a[np.newaxis, :, :] * e[:, np.newaxis, :]  # (m+1, n, n)

    # b[L-1]
    b[L - 1] = 1.0 / s[L - 1] if s[L - 1] > 0 else 0.0

    # b[u] for u = L-2 .. 0  (sequential dependency)
    for u in range(L - 2, -1, -1):
        b[u] = ae[seq[u + 1]] @ b[u + 1]
        if s[u] > 0:
            b[u] /= s[u]

    return b


def log_likelihood(s: np.ndarray) -> float:
    """Compute log P(observations) from scaling factors.

    Parameters
    ----------
    s : (L,) scaling factors from forward algorithm

    Returns
    -------
    float
        log P(seq) = sum of log(s[u])
    """
    return float(np.log(s[s > 0]).sum())


def posterior_decode(
    f: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Posterior decoding: most likely state at each position.

    Parameters
    ----------
    f : (L, n) forward probabilities
    b : (L, n) backward probabilities
    s : (L,) scaling factors

    Returns
    -------
    path : (L,) most likely state index at each position
    """
    gamma = f * b * s[:, np.newaxis]
    return gamma.argmax(axis=1)


def posterior_probabilities(
    f: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Full posterior state probabilities at each position.

    Parameters
    ----------
    f : (L, n) forward probabilities
    b : (L, n) backward probabilities
    s : (L,) scaling factors

    Returns
    -------
    gamma : (L, n) posterior probabilities P(state=k | observations)
    """
    gamma = f * b * s[:, np.newaxis]
    row_sums = gamma.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return gamma / row_sums


def expected_counts(
    a: np.ndarray,
    e: np.ndarray,
    seq: np.ndarray,
    f: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    a0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected sufficient statistics for EM.

    Fully vectorized: the O(L) Python loop is replaced by grouping positions
    by their observation symbol and using matrix multiplications.

    A[k,l] = sum_u f[u,k] * a[k,l] * e[seq[u+1],l] * b[u+1,l]

    For each symbol b, the contribution from all positions where seq[u+1]=b is:
        A += ae[b] * (f_sub.T @ b_sub)
    where f_sub/b_sub are the f/b vectors at matching positions.

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

    # Precompute ae[b, k, l] = a[k, l] * e[b, l]
    ae = a[np.newaxis, :, :] * e[:, np.newaxis, :]  # (m+1, n, n)

    # Initialize with HMM_TINY to avoid log(0)
    A = np.full((n, n), HMM_TINY, dtype=np.float64)
    E = np.full((m_plus_1, n), HMM_TINY, dtype=np.float64)

    # --- Transition counts A[k,l] ---
    # Group by next-symbol: for each b, gather all u where seq[u+1] == b
    seq_next = seq[1:]  # symbols at u+1, for u = 0..L-2
    f_curr = f[:-1]     # f[u] for u = 0..L-2,  shape (L-1, n)
    b_next = b[1:]      # b[u+1] for u = 0..L-2, shape (L-1, n)

    for bb in range(m_plus_1):
        mask = seq_next == bb
        if mask.any():
            f_sub = f_curr[mask]  # (count, n)
            b_sub = b_next[mask]  # (count, n)
            # sum_u f[u,k] * b[u+1,l] = (f_sub.T @ b_sub)[k,l]
            A += ae[bb] * (f_sub.T @ b_sub)

    # --- Emission counts E[b,k] ---
    # E[b, k] = sum_{u: seq[u]=b} f[u,k] * b[u,k] * s[u]
    gamma = f[:-1] * b[:-1] * s[:-1, np.newaxis]  # (L-1, n)
    seq_curr = seq[:-1]

    for bb in range(m_plus_1):
        mask = seq_curr == bb
        if mask.any():
            E[bb] += gamma[mask].sum(axis=0)

    # A0[l] = a0[l] * e[seq[0], l] * b[0, l]
    A0 = a0 * e[seq[0]] * b[0]

    return A, E, A0


def q_function(
    a: np.ndarray,
    e: np.ndarray,
    A: np.ndarray,
    E: np.ndarray,
    Q0: float,
) -> float:
    """Compute the EM Q-function value. Fully vectorized.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix (only rows 0..m-1 used, not the missing row)
    A : (n, n) expected transition counts
    E : (m+1, n) expected emission counts
    Q0 : float, baseline Q value from hmm_Q0

    Returns
    -------
    float
        Q(params) - Q0
    """
    m = e.shape[0] - 1  # exclude missing-data row

    # Check positivity
    if np.any(e[:m] <= 0) or np.any(a <= 0):
        return -1e300

    # Emission term: sum over b=0..m-1, k=0..n-1 of E[b,k] * log(e[b,k])
    total = np.sum(E[:m] * np.log(e[:m]))

    # Transition term: sum of A[k,l] * log(a[k,l])
    total += np.sum(A * np.log(a))

    return float(total - Q0)


def q0_from_counts(A: np.ndarray, E: np.ndarray, m: int) -> float:
    """Compute the Q0 baseline from expected counts. Fully vectorized.

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
    # Emission entropy: sum_k sum_b E[b,k] * log(E[b,k] / sum_b' E[b',k])
    E_real = E[:m]  # (m, n)
    col_sums = E_real.sum(axis=0)  # (n,)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    E_safe = np.where(E_real > 0, E_real, 1.0)  # avoid log(0), masked below
    mask_e = E_real > 0
    emission_entropy = np.sum(
        np.where(mask_e, E_real * np.log(E_safe / col_sums[np.newaxis, :]), 0.0)
    )

    # Transition entropy: sum_k sum_l A[k,l] * log(A[k,l] / sum_l' A[k,l'])
    row_sums = A.sum(axis=1)  # (n,)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    A_safe = np.where(A > 0, A, 1.0)
    mask_a = A > 0
    transition_entropy = np.sum(
        np.where(mask_a, A * np.log(A_safe / row_sums[:, np.newaxis]), 0.0)
    )

    return float(emission_entropy + transition_entropy)


def viterbi(
    a: np.ndarray,
    e: np.ndarray,
    a0: np.ndarray,
    seq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Viterbi algorithm for most likely state sequence.

    Inner loop over states is vectorized; outer loop over L is sequential.

    Parameters
    ----------
    a : (n, n) transition matrix
    e : (m+1, n) emission matrix
    a0 : (n,) initial distribution
    seq : (L,) observation sequence

    Returns
    -------
    path : (L,) most likely state sequence
    log_prob : float, log probability of the best path
    """
    n = a.shape[0]
    L = seq.shape[0]

    # Work in log space
    la = np.log(a + 1e-300)    # (n, n)
    le = np.log(e + 1e-300)    # (m+1, n)
    la0 = np.log(a0 + 1e-300)  # (n,)

    V = np.empty((L, n), dtype=np.float64)
    bt = np.empty((L, n), dtype=np.int32)

    # V[0]
    V[0] = le[seq[0]] + la0
    bt[0] = 0

    # V[u] for u = 1..L-1
    # For each state k: V[u,k] = le[seq[u],k] + max_l(V[u-1,l] + la[l,k])
    for u in range(1, L):
        # scores[l, k] = V[u-1, l] + la[l, k]  →  shape (n, n)
        scores = V[u - 1, :, np.newaxis] + la  # (n, n): row l, col k
        bt[u] = scores.argmax(axis=0)           # best predecessor for each k
        V[u] = le[seq[u]] + scores[bt[u], np.arange(n)]

    # Backtrace
    path = np.empty(L, dtype=np.int32)
    path[L - 1] = V[L - 1].argmax()
    log_prob = V[L - 1, path[L - 1]]
    for u in range(L - 2, -1, -1):
        path[u] = bt[u + 1, path[u + 1]]

    return path, float(log_prob)
