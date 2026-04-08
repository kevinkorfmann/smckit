"""PSMC: Pairwise Sequentially Markovian Coalescent.

Reimplementation of Li & Durbin (2011).
See docs/psmc_internals.md for the full mathematical reference.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smckit._core import SmcData
from smckit.io._psmc_output import read_psmc_output
from smckit.backends._numba import (
    forward_jit,
    backward_jit,
    log_likelihood_jit,
    expected_counts_jit,
    q0_from_counts_jit,
    compute_hmm_params_jit,
    compute_time_intervals_jit,
    kmin_hj_jit,
)
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

PSMC_T_INF = 1000.0
PSMC_N_PARAMS = 3  # theta, rho, max_t


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def parse_pattern(pattern: str) -> tuple[np.ndarray, int, int]:
    """Parse a PSMC pattern string like ``"4+5*3+4"``.

    Parameters
    ----------
    pattern : str
        Pattern string. ``+`` separates groups, ``N*M`` means N repeats of M.

    Returns
    -------
    par_map : (n+1,) int array
        Maps each state index to its free parameter group index.
    n_free : int
        Number of free λ parameters.
    n : int
        Number of time intervals minus one (n_states = n+1).
    """
    stack: list[int] = []
    tokens = pattern.replace("+", " ").split()
    for tok in tokens:
        if "*" in tok:
            parts = tok.split("*")
            repeats = int(parts[0])
            value = int(parts[1])
            stack.extend([value] * repeats)
        else:
            stack.append(int(tok))

    n_free = len(stack)
    total = sum(stack)
    n = total - 1  # number of intervals - 1

    par_map = np.empty(n + 1, dtype=np.int32)
    idx = 0
    for group_id, group_size in enumerate(stack):
        for _ in range(group_size):
            par_map[idx] = group_id
            idx += 1

    return par_map, n_free, n


# ---------------------------------------------------------------------------
# Time intervals (Python wrapper)
# ---------------------------------------------------------------------------

def compute_time_intervals(
    n: int,
    max_t: float = 15.0,
    alpha: float = 0.1,
    inp_ti: np.ndarray | None = None,
) -> np.ndarray:
    """Compute time boundaries t_0, ..., t_{n+1}."""
    if inp_ti is not None:
        t = np.empty(n + 2, dtype=np.float64)
        t[: n + 1] = inp_ti[: n + 1]
        t[n + 1] = PSMC_T_INF
        return t
    return compute_time_intervals_jit(n, max_t, alpha)


# ---------------------------------------------------------------------------
# HMM params (Python wrapper around JIT)
# ---------------------------------------------------------------------------

@dataclass
class PsmcHmmParams:
    """Container for HMM parameters derived from population parameters."""

    a: np.ndarray        # (n+1, n+1) transition matrix
    e: np.ndarray        # (3, n+1) emission matrix (sym 0, 1, missing)
    a0: np.ndarray       # (n+1,) initial distribution (sigma)
    sigma: np.ndarray    # (n+1,) sigma_k
    C_pi: float = 0.0
    C_sigma: float = 0.0


def compute_hmm_params(
    params: np.ndarray,
    par_map: np.ndarray,
    n: int,
    t: np.ndarray,
    divergence: bool = False,
) -> PsmcHmmParams:
    """Map population parameters to HMM transition/emission matrices."""
    a, e, sigma, C_pi, C_sigma = compute_hmm_params_jit(
        params, par_map, n, t, divergence,
    )
    return PsmcHmmParams(
        a=a, e=e, a0=sigma.copy(), sigma=sigma,
        C_pi=C_pi, C_sigma=C_sigma,
    )


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def _em_step(
    records: list[dict],
    params: np.ndarray,
    par_map: np.ndarray,
    n: int,
    t: np.ndarray,
    divergence: bool = False,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """Run one EM iteration.

    Returns
    -------
    params : updated parameter vector
    LL : log-likelihood
    Q0_val : Q before optimization
    Q1_val : Q after optimization
    post_sigma : posterior state distribution
    """
    n_states = n + 1

    # --- E-step: compute expected counts ---
    a, e, sigma, C_pi, C_sigma = compute_hmm_params_jit(
        params, par_map, n, t, divergence,
    )

    A_sum = np.zeros((n_states, n_states), dtype=np.float64)
    E_sum = np.zeros((3, n_states), dtype=np.float64)
    LL = 0.0

    for rec in records:
        seq = rec["codes"]
        f_arr, s_arr = forward_jit(a, e, sigma, seq)
        b_arr = backward_jit(a, e, seq, s_arr)
        LL += log_likelihood_jit(s_arr)

        A, E, A0 = expected_counts_jit(a, e, seq, f_arr, b_arr, s_arr, sigma)
        A_sum += A
        E_sum += E

    # Q0 baseline
    Q0_val = q0_from_counts_jit(A_sum, E_sum, 2)

    # Q before optimization (using JIT compute_hmm_params output)
    from smckit.backends._numba import q_function_jit
    Q_before = q_function_jit(a, e, A_sum, E_sum, Q0_val)

    # --- M-step: Hooke-Jeeves optimizer (all JIT) ---
    optimized_params, Q1_val = kmin_hj_jit(
        params, par_map, n, A_sum, E_sum, Q0_val, divergence,
    )

    # Update time intervals with optimized max_t
    t[:] = compute_time_intervals_jit(n, optimized_params[2], 0.1)

    # Posterior sigma
    col_sums = E_sum[0] + E_sum[1]
    total = col_sums.sum()
    post_sigma = col_sums / total if total > 0 else np.ones(n_states) / n_states

    return optimized_params, LL, Q_before, Q1_val, post_sigma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class PsmcResult:
    """Results from a PSMC run."""

    time: np.ndarray           # (n+1,) time boundaries t_k
    lambda_k: np.ndarray       # (n+1,) relative pop sizes per state
    ne: np.ndarray             # (n+1,) N_e(t_k) = lambda_k * N0
    time_years: np.ndarray     # (n+1,) time in years
    theta: float = 0.0
    rho: float = 0.0
    n0: float = 0.0            # N0 = theta / (4 * mu * window_size)
    log_likelihood: float = 0.0
    n_iterations: int = 0
    pattern: str = ""
    rounds: list[dict] = field(default_factory=list)


def psmc(
    data: SmcData,
    pattern: str = "4+5*3+4",
    n_iterations: int = 30,
    max_t: float = 15.0,
    tr_ratio: float = 4.0,
    alpha: float = 0.1,
    mu: float = 1.25e-8,
    generation_time: float = 25.0,
    random_init: float = 0.01,
    seed: int | None = None,
    implementation: str = "auto",
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run PSMC demographic inference.

    Parameters
    ----------
    data : SmcData
        Input data from ``smckit.io.read_psmcfa()``.
    pattern : str
        Parameter pattern (default ``"4+5*3+4"``).
    n_iterations : int
        Number of EM iterations.
    max_t : float
        Maximum coalescent time (units of 2N₀).
    tr_ratio : float
        Initial θ/ρ ratio.
    alpha : float
        Time interval spacing parameter.
    mu : float
        Per-base per-generation mutation rate.
    generation_time : float
        Generation time in years.
    random_init : float
        Amplitude of random initialization noise for λ.
    seed : int, optional
        Random seed.
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo port.
        ``"upstream"`` requests the original implementation and currently
        raises because no public upstream bridge is exposed yet. ``"auto"``
        resolves to the best available implementation.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["psmc"]``.
    """
    implementation = normalize_implementation(implementation)
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("psmc"),
    )
    if implementation_used == "upstream":
        return _psmc_upstream(
            data,
            pattern=pattern,
            n_iterations=n_iterations,
            max_t=max_t,
            tr_ratio=tr_ratio,
            mu=mu,
            generation_time=generation_time,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )
    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported psmc native_options keys: {unsupported}")

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    records = data.uns["records"]
    sum_L = data.uns["sum_L"]
    sum_n = data.uns["sum_n"]
    window_size = data.window_size

    # Parse pattern
    par_map, n_free, n = parse_pattern(pattern)
    n_states = n + 1
    logger.info("pattern=%s n=%d n_free=%d n_states=%d", pattern, n, n_free, n_states)

    # Initialize parameters
    n_params = n_free + PSMC_N_PARAMS
    params = np.zeros(n_params, dtype=np.float64)

    theta = -np.log(1.0 - sum_n / sum_L) if sum_L > sum_n else sum_n / sum_L
    params[0] = theta           # theta_0
    params[1] = theta / tr_ratio  # rho_0
    params[2] = max_t           # max_t

    for k in range(PSMC_N_PARAMS, n_params):
        params[k] = 1.0 + (rng.random() * 2.0 - 1.0) * random_init
        if params[k] < 0.1:
            params[k] = 0.1

    # Time intervals
    t = compute_time_intervals(n, max_t, alpha)

    # EM loop
    rounds: list[dict] = []

    # Record round 0 (initial parameters)
    hp0 = compute_hmm_params(params, par_map, n, t)
    lam0 = params[par_map + PSMC_N_PARAMS]
    rounds.append({
        "round": 0,
        "theta": params[0],
        "rho": params[1],
        "max_t": params[2],
        "time": t[:n_states].copy(),
        "lambda": lam0,
        "sigma": hp0.sigma.copy(),
        "params": params.copy(),
    })

    for i in range(n_iterations):
        logger.info("EM iteration %d/%d", i + 1, n_iterations)

        params, LL, Q0, Q1, post_sigma = _em_step(
            records, params, par_map, n, t,
        )

        lam_i = params[par_map + PSMC_N_PARAMS]
        hp_i = compute_hmm_params(params, par_map, n, t)

        rd = {
            "round": i + 1,
            "log_likelihood": LL,
            "Q0": Q0,
            "Q1": Q1,
            "theta": params[0],
            "rho": params[1],
            "max_t": params[2],
            "time": t[:n_states].copy(),
            "lambda": lam_i,
            "sigma": hp_i.sigma.copy(),
            "post_sigma": post_sigma,
            "params": params.copy(),
        }
        rounds.append(rd)
        logger.info(
            "  LL=%.2f Q=%.4f->%.4f theta=%.6f rho=%.6f",
            LL, Q0, Q1, params[0], params[1],
        )

    # Final results
    theta_final = params[0]
    rho_final = params[1]
    n0 = theta_final / (4.0 * mu * window_size)

    lam_final = params[par_map + PSMC_N_PARAMS]
    ne = lam_final * n0
    time_years = t[:n_states] * 2.0 * n0 * generation_time

    result = PsmcResult(
        time=t[:n_states].copy(),
        lambda_k=lam_final,
        ne=ne,
        time_years=time_years,
        theta=theta_final,
        rho=rho_final,
        n0=n0,
        log_likelihood=rounds[-1].get("log_likelihood", 0.0),
        n_iterations=n_iterations,
        pattern=pattern,
        rounds=rounds,
    )

    data.results["psmc"] = annotate_result({
        "time": result.time,
        "lambda": result.lambda_k,
        "ne": result.ne,
        "time_years": result.time_years,
        "theta": result.theta,
        "rho": result.rho,
        "n0": result.n0,
        "log_likelihood": result.log_likelihood,
        "pattern": result.pattern,
        "rounds": result.rounds,
    }, implementation_requested=implementation, implementation_used=implementation_used)
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time

    return data


def _psmc_binary_path() -> Path | None:
    status = upstream_status("psmc")
    cache_path = Path(status["cache_path"]) / "bin/psmc"
    if cache_path.exists():
        return cache_path
    vendor_path = Path(status["vendor_path"]) / "psmc"
    if vendor_path.exists():
        return vendor_path
    return None


def _write_psmcfa(records: list[dict], path: Path) -> None:
    decode = np.array(["T", "K", "N"], dtype=object)
    with path.open("wt", encoding="utf-8") as fh:
        for idx, record in enumerate(records):
            name = record.get("name", f"sequence_{idx}")
            codes = np.asarray(record["codes"], dtype=np.int8)
            seq = "".join(decode[int(x)] if 0 <= int(x) < 3 else "N" for x in codes)
            fh.write(f">{name}\n")
            for start in range(0, len(seq), 60):
                fh.write(seq[start:start + 60] + "\n")


def _psmc_upstream(
    data: SmcData,
    *,
    pattern: str,
    n_iterations: int,
    max_t: float,
    tr_ratio: float,
    mu: float,
    generation_time: float,
    implementation_requested: str,
    upstream_options: dict | None,
) -> SmcData:
    status = upstream_status("psmc")
    if not status["source_present"] or not status["runtime_ready"]:
        require_upstream_available("psmc")
    if not status["cache_ready"]:
        bootstrap_upstream("psmc")

    binary = _psmc_binary_path()
    if binary is None:
        raise RuntimeError("Upstream PSMC binary is unavailable after bootstrap.")

    source_path = data.uns.get("source_path")
    effective_args = {
        "pattern": pattern,
        "n_iterations": int(n_iterations),
        "max_t": float(max_t),
        "tr_ratio": float(tr_ratio),
    }
    if upstream_options:
        effective_args.update(upstream_options)

    with tempfile.TemporaryDirectory(prefix="smckit-psmc-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.psmcfa"
        if source_path:
            input_path = Path(source_path)
        else:
            _write_psmcfa(data.uns["records"], input_path)

        output_path = tmpdir_path / "result.psmc"
        cmd = [
            str(binary),
            "-N",
            str(int(n_iterations)),
            "-t",
            repr(float(max_t)),
            "-r",
            repr(float(tr_ratio)),
            "-p",
            pattern,
            "-o",
            str(output_path),
            str(input_path),
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream PSMC backend failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        rounds = read_psmc_output(output_path)
        if not rounds:
            raise RuntimeError("Upstream PSMC produced no rounds.")
        final = rounds[-1]
        window_size = data.window_size
        theta = float(final["theta"])
        rho = float(final["rho"])
        time = np.asarray(final["time"], dtype=np.float64)
        lam = np.asarray(final["lambda"], dtype=np.float64)
        n0 = theta / (4.0 * mu * window_size)
        ne = lam * n0
        time_years = time * 2.0 * n0 * generation_time

        data.results["psmc"] = annotate_result(
            {
                "time": time,
                "lambda": lam,
                "ne": ne,
                "time_years": time_years,
                "theta": theta,
                "rho": rho,
                "n0": n0,
                "log_likelihood": float(final.get("log_likelihood", 0.0)),
                "pattern": pattern,
                "rounds": rounds,
                "backend": "upstream",
                "upstream": standard_upstream_metadata(
                    "psmc",
                    effective_args=effective_args,
                    extra={
                        "binary": str(binary),
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "input_path": str(input_path),
                    },
                ),
            },
            implementation_requested=implementation_requested,
            implementation_used="upstream",
        )
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time
    return data
