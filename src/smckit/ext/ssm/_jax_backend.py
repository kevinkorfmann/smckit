"""JAX differentiable backend for PSMC-SSM.

Provides:
- ``compute_time_intervals_jax`` — time boundary computation
- ``compute_hmm_params_jax`` — coalescent-to-HMM mapping (differentiable)
- ``forward_log`` — log-space forward algorithm via ``jax.lax.scan``
- ``fit_gradient`` — gradient-based fitting with optax
"""

from __future__ import annotations

import logging

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy.special
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import numpy as np

logger = logging.getLogger(__name__)

PSMC_T_INF = 1000.0
PSMC_N_PARAMS = 3
_TINY = 1e-300


def _check_jax():
    if not HAS_JAX:
        raise ImportError(
            "JAX is required for gradient-based fitting. "
            "Install with: pip install smckit[jax]"
        )


# ---------------------------------------------------------------------------
# Time intervals
# ---------------------------------------------------------------------------

def compute_time_intervals_jax(n: int, max_t, alpha: float):
    """Compute time boundaries (JAX-compatible).

    Parameters
    ----------
    n : int
        Number of intervals minus 1 (n+1 states).
    max_t : jax scalar
        Maximum coalescent time.
    alpha : float
        Spacing parameter.

    Returns
    -------
    t : (n+2,) jax array
    """
    beta = jnp.log(1.0 + max_t / alpha) / n
    k = jnp.arange(n, dtype=jnp.float64)
    t_inner = alpha * (jnp.exp(beta * k) - 1.0)
    t = jnp.concatenate([t_inner, jnp.array([max_t, PSMC_T_INF])])
    return t


# ---------------------------------------------------------------------------
# Coalescent -> HMM params (differentiable)
# ---------------------------------------------------------------------------

def compute_hmm_params_jax(params, par_map, n: int, t, divergence: bool = False):
    """Map population parameters to HMM matrices — JAX differentiable.

    Translation of ``backends/_numba.py:compute_hmm_params_jit``.

    Parameters
    ----------
    params : (n_params,) jax array
        [theta, rho, max_t, lam_0, ..., lam_{n_free-1}].
    par_map : (n+1,) int array
        Maps each state to its free parameter index.
    n : int
        n value (n+1 states).
    t : (n+2,) jax array
        Time boundaries.
    divergence : bool
        Whether to use divergence-time parameter.

    Returns
    -------
    a_mat : (n+1, n+1) transition matrix
    e_mat : (3, n+1) emission matrix
    sigma : (n+1,) initial distribution
    """
    n_states = n + 1
    theta = params[0]
    rho = params[1]

    # Lambda for each state (via par_map gather — differentiable in JAX)
    lam = params[par_map + PSMC_N_PARAMS]

    # Divergence time
    if divergence:
        dt = jnp.maximum(params[-1], 0.0)
    else:
        dt = 0.0

    # Interval widths
    tau = t[1: n_states + 1] - t[:n_states]

    # Log survival per interval
    log_surv = -tau / lam

    # Alpha: survival probabilities via cumulative sum of log survival
    # alpha[0] = 1.0, alpha[k] = exp(sum(log_surv[0:k])), alpha[n_states] = 0.0
    cum_log = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(log_surv[:-1])])
    alpha = jnp.concatenate([jnp.exp(cum_log), jnp.array([0.0])])

    # Safe alpha for divisions
    alpha_safe = jnp.maximum(alpha, _TINY)

    # Beta: cumulative weighted divergence
    # beta[0] = 0.0
    # beta[k] = sum_{j<k} lam[j] * (1/alpha_safe[j+1] - 1/alpha_safe[j])
    beta_increments = lam[:-1] * (1.0 / alpha_safe[1:n_states] - 1.0 / alpha_safe[:n_states - 1])
    beta = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(beta_increments)])

    # q_aux[l] for l=0..n-1
    ak1_full = alpha[:n_states] - alpha[1: n_states + 1]  # (n_states,)
    ak1_n = ak1_full[:n]  # first n elements
    q_aux = ak1_n * (beta[:n] - lam[:n] / alpha_safe[:n]) + tau[:n]

    # ak1 = alpha[k] - alpha[k+1] for all n_states
    ak1 = ak1_full

    # C_pi normalization
    C_pi = jnp.sum(lam * ak1)

    C_sigma = 1.0 / (C_pi * rho) + 0.5

    # Cumulative time: sum_t[k] = sum(tau[0:k])
    sum_t = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(tau[:-1])])

    # cpik, pik, sigma
    cpik = ak1 * (sum_t + lam) - alpha[1: n_states + 1] * tau
    pik = cpik / C_pi
    sigma = (ak1 / (C_pi * rho) + pik / 2.0) / C_sigma

    # avg_t with fallback
    denom = C_sigma * sigma
    denom_safe = jnp.maximum(denom, _TINY)
    ratio = pik / denom_safe

    # Primary path: -log(1 - ratio) / rho
    arg = jnp.clip(1.0 - ratio, _TINY, None)
    val_primary = -jnp.log(arg) / rho

    # Fallback path
    ak1_safe = jnp.maximum(ak1, _TINY)
    val_fallback = sum_t + lam - tau * alpha[1: n_states + 1] / ak1_safe

    # Select: use primary if valid and in bounds
    in_bounds = (
        jnp.isfinite(val_primary)
        & (val_primary >= sum_t)
        & (val_primary <= sum_t + tau)
        & (ratio > 0.0)
        & (ratio < 1.0)
    )
    avg_t = jnp.where(in_bounds, val_primary, val_fallback)

    # --- Transition matrix ---
    cpik_safe = jnp.maximum(cpik, _TINY)

    # ratio_k[k] = ak1[k] / cpik_safe[k]  — shape (n_states,)
    ratio_k = ak1 / cpik_safe

    # q_aux padded to n_states (last element unused for upper triangle)
    q_aux_padded = jnp.concatenate([q_aux, jnp.array([0.0])])

    # ratio_k2[k] = q_aux[k] / cpik_safe[k] for k < n
    ratio_k2 = q_aux / cpik_safe[:n]
    ratio_k2_padded = jnp.concatenate([ratio_k2, jnp.array([0.0])])

    # Lower triangle: a[k, l] = ratio_k[k] * q_aux[l]  for l < k
    idx = jnp.arange(n_states)
    lower_mask = (idx[None, :] < idx[:, None]).astype(jnp.float64)
    lower = jnp.outer(ratio_k, q_aux_padded[:n_states]) * lower_mask

    # Upper triangle: a[k, l] = ak1[l] * ratio_k2[k]  for l > k, k < n
    upper_mask = (idx[None, :] > idx[:, None]).astype(jnp.float64)
    upper = jnp.outer(ratio_k2_padded[:n_states], ak1) * upper_mask

    # Diagonal
    ak_s = alpha_safe[:n_states]
    diag_vals = (
        ak1 * ak1 * (beta - lam / ak_s)
        + 2.0 * lam * ak1
        - 2.0 * alpha[1: n_states + 1] * tau
    ) / cpik_safe

    q_mat = lower + upper + jnp.diag(diag_vals)

    # Where cpik <= 0, set row to identity (diagonal = 1)
    valid = (cpik > 0.0).astype(jnp.float64)
    identity_row = jnp.eye(n_states)
    q_mat = q_mat * valid[:, None] + identity_row * (1.0 - valid[:, None])

    # Scale by recombination probability
    tmp_p = jnp.where(denom > 0.0, pik / denom_safe, 0.0)
    a_mat = q_mat * tmp_p[:, None] + jnp.diag(1.0 - tmp_p)

    # --- Emission matrix ---
    e0 = jnp.exp(-theta * (avg_t + dt))
    e1 = 1.0 - e0
    e_missing = jnp.ones(n_states)
    e_mat = jnp.stack([e0, e1, e_missing])  # (3, n_states)

    return a_mat, e_mat, sigma


# ---------------------------------------------------------------------------
# Log-space forward algorithm
# ---------------------------------------------------------------------------

def forward_log(log_a, log_e, log_a0, seq):
    """Log-space forward algorithm using ``jax.lax.scan``.

    Parameters
    ----------
    log_a : (n, n) log transition matrix
    log_e : (m+1, n) log emission matrix
    log_a0 : (n,) log initial distribution
    seq : (L,) int32 observation sequence

    Returns
    -------
    log_likelihood : float
    """
    # Initialize
    log_f0_unnorm = log_a0 + log_e[seq[0]]
    log_norm0 = jax.scipy.special.logsumexp(log_f0_unnorm)
    log_f0 = log_f0_unnorm - log_norm0

    def scan_fn(log_f_prev, obs):
        # log_f[k] = log_e[obs, k] + logsumexp_j(log_a[j, k] + log_f_prev[j])
        log_f_unnorm = log_e[obs] + jax.scipy.special.logsumexp(
            log_a.T + log_f_prev[None, :], axis=1
        )
        log_norm = jax.scipy.special.logsumexp(log_f_unnorm)
        log_f = log_f_unnorm - log_norm
        return log_f, log_norm

    _, log_norms = jax.lax.scan(scan_fn, log_f0, seq[1:])
    return log_norm0 + jnp.sum(log_norms)


# ---------------------------------------------------------------------------
# Negative log-likelihood (differentiable)
# ---------------------------------------------------------------------------

def _neg_log_likelihood(x, par_map, n, alpha, observations, divergence=False):
    """Negative log-likelihood in log-parameterized space.

    Parameters
    ----------
    x : unconstrained parameter vector (params_pos = exp(x))
    par_map : int array mapping states to free params
    n : int
    alpha : float
    observations : list of (L_i,) int32 arrays
    divergence : bool

    Returns
    -------
    float : -log P(observations | exp(x))
    """
    params_pos = jnp.exp(x)

    t = compute_time_intervals_jax(n, params_pos[2], alpha)
    a, e, sigma = compute_hmm_params_jax(params_pos, par_map, n, t, divergence)

    log_a = jnp.log(jnp.maximum(a, _TINY))
    log_e = jnp.log(jnp.maximum(e, _TINY))
    log_a0 = jnp.log(jnp.maximum(sigma, _TINY))

    total_ll = 0.0
    for seq in observations:
        total_ll = total_ll + forward_log(log_a, log_e, log_a0, seq)

    return -total_ll


# ---------------------------------------------------------------------------
# Gradient-based fitting
# ---------------------------------------------------------------------------

def fit_gradient(
    ssm,
    observations: list[np.ndarray],
    params_init: np.ndarray,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    n_iterations: int = 500,
    verbose: bool = True,
    log_every: int = 50,
):
    """Fit PSMC-SSM via gradient-based optimization of marginal log-likelihood.

    Parameters
    ----------
    ssm : PsmcSSM
        The state-space model instance.
    observations : list of np.ndarray
        Observation sequences (int8 arrays).
    params_init : np.ndarray
        Initial parameter vector (positive values).
    optimizer : str
        Optimizer name: ``"adam"`` or ``"sgd"``.
    learning_rate : float
        Learning rate for the optimizer.
    n_iterations : int
        Number of optimization steps.
    verbose : bool
        Whether to log progress.
    log_every : int
        Log every N iterations.

    Returns
    -------
    FitResult
    """
    _check_jax()
    import optax

    from smckit.ext.ssm._base import FitResult

    par_map = jnp.array(ssm.par_map, dtype=jnp.int32)
    n = ssm.n
    alpha = ssm.alpha
    divergence = ssm.divergence

    # Convert observations to jax arrays
    obs_jax = [jnp.array(s, dtype=jnp.int32) for s in observations]

    # Log-space reparameterization
    x = jnp.log(jnp.maximum(jnp.array(params_init, dtype=jnp.float64), 1e-10))

    # Build loss+grad function (single forward+backward pass per step)
    @jax.jit
    def loss_and_grad(x):
        return jax.value_and_grad(
            lambda x: _neg_log_likelihood(x, par_map, n, alpha, obs_jax, divergence)
        )(x)

    # Optimizer
    if optimizer == "adam":
        opt = optax.adam(learning_rate)
    elif optimizer == "sgd":
        opt = optax.sgd(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Use 'adam' or 'sgd'.")

    opt_state = opt.init(x)

    # JIT warmup (compilation happens here)
    if verbose:
        logger.info("JIT compiling loss+grad (this may take a moment)...")
    loss, grads = loss_and_grad(x)
    jax.block_until_ready(loss)
    if verbose:
        logger.info("JIT compilation done. Starting optimization...")

    history: list[dict] = []
    best_loss = float("inf")
    best_x = x

    for i in range(n_iterations):
        loss, grads = loss_and_grad(x)
        updates, opt_state = opt.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)

        loss_val = float(loss)
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x

        history.append({
            "iteration": i + 1,
            "loss": loss_val,
            "log_likelihood": -loss_val,
            "grad_norm": float(jnp.linalg.norm(grads)),
            "params": np.asarray(jnp.exp(x)),
        })

        if verbose and (i + 1) % log_every == 0:
            logger.info(
                "Gradient iter %d/%d  LL=%.2f  |grad|=%.4e",
                i + 1, n_iterations, -loss_val,
                float(jnp.linalg.norm(grads)),
            )

    final_params = np.asarray(jnp.exp(best_x))
    return FitResult(
        params=final_params,
        log_likelihood=-best_loss,
        n_iterations=n_iterations,
        converged=True,
        history=history,
    )
