"""NumPy/EM fallback fitting for SSM, wrapping existing Numba code."""

from __future__ import annotations

import logging

import numpy as np

from smckit.backends._numba import (
    backward_jit,
    compute_hmm_params_jit,
    compute_time_intervals_jit,
    expected_counts_jit,
    forward_jit,
    kmin_hj_jit,
    log_likelihood_jit,
    q0_from_counts_jit,
    q_function_jit,
)
from smckit.ext.ssm._base import FitResult

logger = logging.getLogger(__name__)


def fit_em(
    ssm,
    observations: list[np.ndarray],
    params_init: np.ndarray,
    n_iterations: int = 30,
) -> FitResult:
    """Fit via EM with Hooke-Jeeves M-step, reusing existing Numba backend.

    Parameters
    ----------
    ssm : PsmcSSM
        The state-space model instance.
    observations : list of np.ndarray
        Observation sequences (int8 arrays).
    params_init : np.ndarray
        Initial parameter vector.
    n_iterations : int
        Number of EM iterations.

    Returns
    -------
    FitResult
    """
    params = params_init.copy()
    par_map = ssm.par_map
    n = ssm.n
    n_states = ssm.n_states
    divergence = ssm.divergence

    t = compute_time_intervals_jit(n, float(params[2]), ssm.alpha)

    history: list[dict] = []

    for i in range(n_iterations):
        # E-step
        a, e, sigma, C_pi, C_sigma = compute_hmm_params_jit(
            params, par_map, n, t, divergence,
        )

        A_sum = np.zeros((n_states, n_states), dtype=np.float64)
        E_sum = np.zeros((3, n_states), dtype=np.float64)
        LL = 0.0

        for seq in observations:
            f_arr, s_arr = forward_jit(a, e, sigma, seq)
            b_arr = backward_jit(a, e, seq, s_arr)
            LL += log_likelihood_jit(s_arr)
            A, E, A0 = expected_counts_jit(a, e, seq, f_arr, b_arr, s_arr, sigma)
            A_sum += A
            E_sum += E

        Q0_val = q0_from_counts_jit(A_sum, E_sum, 2)
        Q_before = q_function_jit(a, e, A_sum, E_sum, Q0_val)

        # M-step
        params, Q_after = kmin_hj_jit(
            params, par_map, n, A_sum, E_sum, Q0_val, divergence,
        )
        t[:] = compute_time_intervals_jit(n, params[2], ssm.alpha)

        history.append({
            "iteration": i + 1,
            "log_likelihood": LL,
            "Q_before": Q_before,
            "Q_after": Q_after,
            "params": params.copy(),
        })
        logger.info(
            "EM iter %d/%d  LL=%.2f Q=%.4f->%.4f",
            i + 1, n_iterations, LL, Q_before, Q_after,
        )

    # Final log-likelihood
    a, e, sigma, _, _ = compute_hmm_params_jit(params, par_map, n, t, divergence)
    final_ll = 0.0
    for seq in observations:
        f_arr, s_arr = forward_jit(a, e, sigma, seq)
        final_ll += log_likelihood_jit(s_arr)

    return FitResult(
        params=params,
        log_likelihood=final_ll,
        n_iterations=n_iterations,
        converged=True,
        history=history,
    )
