"""PSMC: Pairwise Sequentially Markovian Coalescent.

Reimplementation of Li & Durbin (2011).
See docs/psmc_internals.md for the full mathematical reference.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from smckit._core import SmcData
from smckit._provenance import sha256_file
from smckit.backends._numba import (
    backward_jit,
    compute_hmm_params_jit,
    compute_time_intervals_jit,
    expected_counts_jit,
    forward_jit,
    kmin_hj_jit,
    log_likelihood_jit,
    q0_from_counts_jit,
)
from smckit.io._psmc_output import read_psmc_output, write_psmc_output
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    require_upstream_available,
    standard_upstream_metadata,
    warn_if_native_not_trusted,
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

    a: np.ndarray  # (n+1, n+1) transition matrix
    e: np.ndarray  # (3, n+1) emission matrix (sym 0, 1, missing)
    a0: np.ndarray  # (n+1,) initial distribution (sigma)
    sigma: np.ndarray  # (n+1,) sigma_k
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
        params,
        par_map,
        n,
        t,
        divergence,
    )
    return PsmcHmmParams(
        a=a,
        e=e,
        a0=sigma.copy(),
        sigma=sigma,
        C_pi=C_pi,
        C_sigma=C_sigma,
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
    alpha: float = 0.1,
    fixed_time_intervals: bool = False,
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
        params,
        par_map,
        n,
        t,
        divergence,
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
        params,
        par_map,
        n,
        A_sum,
        E_sum,
        Q0_val,
        divergence,
    )

    # Upstream preserves explicit input intervals; otherwise it recomputes
    # boundaries using the requested skipping factor.
    if not fixed_time_intervals:
        t[:] = compute_time_intervals_jit(n, optimized_params[2], alpha)

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

    time: np.ndarray  # (n+1,) time boundaries t_k
    lambda_k: np.ndarray  # (n+1,) relative pop sizes per state
    ne: np.ndarray  # (n+1,) N_e(t_k) = lambda_k * N0
    time_years: np.ndarray  # (n+1,) time in years
    theta: float = 0.0
    rho: float = 0.0
    n0: float = 0.0  # N0 = theta / (4 * mu * window_size)
    log_likelihood: float = 0.0
    n_iterations: int = 0
    pattern: str = ""
    rounds: list[dict] = field(default_factory=list)


def _record_stats(records: list[dict[str, Any]]) -> tuple[int, int]:
    callable_sites = sum(int(np.count_nonzero(np.asarray(rec["codes"]) < 2)) for rec in records)
    heterozygous_sites = sum(
        int(np.count_nonzero(np.asarray(rec["codes"]) == 1)) for rec in records
    )
    return callable_sites, heterozygous_sites


def split_psmc_records(
    records: list[dict[str, Any]],
    segment_length: int = 500_000,
) -> list[dict[str, Any]]:
    """Split PSMCFA records with the exact upstream ``splitfa`` boundary rule."""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive.")
    split: list[dict[str, Any]] = []
    for record in records:
        codes = np.asarray(record["codes"], dtype=np.int8)
        start = 0
        segment_id = 1
        while start < len(codes):
            remaining = len(codes) - start
            end = len(codes) if remaining < segment_length * 3 / 2 else start + segment_length
            chunk = codes[start:end].copy()
            callable_mask = chunk < 2
            split.append(
                {
                    "name": f"{record.get('name', 'sequence')}_{segment_id}",
                    "codes": chunk,
                    "L": len(chunk),
                    "L_e": int(callable_mask.sum()),
                    "n_e": int((chunk == 1).sum()),
                    "source_name": record.get("name", "sequence"),
                    "source_start": start,
                    "source_end": end,
                }
            )
            start = end
            segment_id += 1
    return split


def bootstrap_psmc_records(
    records: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Resample PSMC segments using the length-matching rule in upstream PSMC."""
    if not records:
        raise ValueError("Cannot bootstrap an empty PSMC record collection.")
    target_length = sum(len(np.asarray(record["codes"])) for record in records)
    sampled: list[dict[str, Any]] = []
    sampled_length = 0
    while True:
        record = records[int(rng.integers(0, len(records)))]
        record_length = len(np.asarray(record["codes"]))
        under = target_length - sampled_length
        over = sampled_length + record_length - target_length
        if over <= 0 or (under > 0 and 0 < over < under):
            clone = dict(record)
            clone["codes"] = np.asarray(record["codes"], dtype=np.int8).copy()
            clone["name"] = f"{record.get('name', 'sequence')}_resampled_{len(sampled) + 1}"
            sampled.append(clone)
            sampled_length += record_length
        if under >= 0 and over >= 0:
            break
    if not sampled:
        # The upstream rule can reject a single overlong draw. Retain the
        # closest segment so the Python surface fails safely rather than looping.
        record = min(records, key=lambda item: abs(len(np.asarray(item["codes"])) - target_length))
        sampled = [dict(record)]
        sampled[0]["codes"] = np.asarray(record["codes"], dtype=np.int8).copy()
    return sampled


def _cap_transition_matrix(matrix: np.ndarray, cap_state: int | None) -> np.ndarray:
    if cap_state is None:
        return matrix
    if not 0 < cap_state < matrix.shape[0] - 1:
        raise ValueError(f"transition_cap must be between 1 and {matrix.shape[0] - 2}.")
    capped = matrix.copy()
    capped[:, cap_state] = capped[:, cap_state:].sum(axis=1)
    capped[:, cap_state + 1 :] = 0.0
    return capped


def _decode_records(
    records: list[dict[str, Any]],
    hp: PsmcHmmParams,
    theta: float,
    *,
    mode: str,
    transition_cap: int | None,
) -> dict[str, Any]:
    if mode not in {"posterior", "full"}:
        raise ValueError("decode must be one of: None, 'posterior', 'full'.")
    transition = _cap_transition_matrix(hp.a, transition_cap)
    expected_time = -np.log(np.maximum(hp.e[0], 1e-300)) / max(theta, 1e-300)
    decoded_records: list[dict[str, Any]] = []
    for record in records:
        sequence = np.asarray(record["codes"], dtype=np.int8)
        forward, scales = forward_jit(transition, hp.e, hp.a0, sequence)
        backward = backward_jit(transition, hp.e, sequence, scales)
        posterior = forward * backward * scales[:, np.newaxis]
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)
        states = posterior.argmax(axis=1).astype(np.int32)
        payload: dict[str, Any] = {
            "name": record.get("name", "sequence"),
            "length": len(sequence),
            "log_likelihood": float(log_likelihood_jit(scales)),
        }
        if mode == "posterior":
            segments: list[dict[str, Any]] = []
            start = 0
            while start < len(states):
                state = int(states[start])
                end = start + 1
                while end < len(states) and int(states[end]) == state:
                    end += 1
                segments.append(
                    {
                        "start": start + 1,
                        "end": end,
                        "state": state,
                        "scaled_time": float(expected_time[state] * theta),
                        "max_probability": float(posterior[start:end, state].max()),
                    }
                )
                start = end
            payload["segments"] = segments
        else:
            recombination = np.zeros(len(sequence), dtype=np.float64)
            for position in range(len(sequence) - 1):
                diagonal_mass = np.sum(
                    forward[position]
                    * np.diag(transition)
                    * backward[position + 1]
                    * hp.e[sequence[position + 1]]
                )
                recombination[position] = np.clip(1.0 - diagonal_mass, 0.0, 1.0)
            payload.update(
                {
                    "position": np.arange(1, len(sequence) + 1, dtype=np.int64),
                    "recombination_probability": recombination,
                    "posterior": posterior,
                    "state": states,
                }
            )
        decoded_records.append(payload)
    return {
        "mode": mode,
        "transition_cap": transition_cap,
        "state_expected_time": expected_time,
        "records": decoded_records,
    }


def _sequence_probabilities(
    records: list[dict[str, Any]],
    hp: PsmcHmmParams,
) -> list[dict[str, Any]]:
    probabilities: list[dict[str, Any]] = []
    for record in records:
        sequence = np.asarray(record["codes"], dtype=np.int8)
        _, scales = forward_jit(hp.a, hp.e, hp.a0, sequence)
        probabilities.append(
            {
                "name": record.get("name", "sequence"),
                "length": len(sequence),
                "scale": scales,
                "log_likelihood": float(log_likelihood_jit(scales)),
            }
        )
    return probabilities


def _simulate_records(
    records: list[dict[str, Any]],
    hp: PsmcHmmParams,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    simulated: list[dict[str, Any]] = []
    for record in records:
        template = np.asarray(record["codes"], dtype=np.int8)
        sequence = np.empty_like(template)
        state = int(rng.choice(len(hp.a0), p=hp.a0 / hp.a0.sum()))
        for position, observed in enumerate(template):
            if observed == 2:
                sequence[position] = 2
            else:
                emission = hp.e[:2, state]
                sequence[position] = int(rng.choice(2, p=emission / emission.sum()))
            if position + 1 < len(template):
                transition = hp.a[state]
                state = int(rng.choice(len(transition), p=transition / transition.sum()))
        simulated.append(
            {
                "name": record.get("name", "sequence"),
                "codes": sequence,
                "L": len(sequence),
                "L_e": int((sequence < 2).sum()),
                "n_e": int((sequence == 1).sum()),
            }
        )
    return simulated


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
    bootstrap: bool = False,
    bootstrap_segment_length: int = 500_000,
    decode: str | None = None,
    transition_cap: int | None = None,
    initial_params: np.ndarray | list[float] | None = None,
    time_intervals: np.ndarray | list[float] | None = None,
    divergence_time: float | None = None,
    sequence_probability: bool = False,
    simulate: bool = False,
    output_path: str | Path | None = None,
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
    bootstrap : bool
        Resample pre-split sequence segments with replacement.
    bootstrap_segment_length : int
        Length used to split records before native bootstrap resampling.
    decode : {None, "posterior", "full"}
        Return posterior path segments or full per-window posterior values.
    transition_cap : int, optional
        Collapse transition probability at and above this state for decoding.
    initial_params : array-like, optional
        Initial ``theta, rho, max_t, lambda...[, divergence]`` parameters.
    time_intervals : array-like, optional
        Explicit state boundaries, excluding the infinity sentinel.
    divergence_time : float, optional
        Enable the divergence model with this initial scaled divergence time.
    sequence_probability : bool
        Return the per-window forward scaling values printed by upstream ``-s``.
    simulate : bool
        Simulate PSMCFA records from the final fitted HMM while preserving masks.
    output_path : str or Path, optional
        Write an upstream-compatible PSMC result file.
    implementation : {"auto", "native", "upstream"}
        Algorithm provenance selector. ``"native"`` runs the in-repo port.
        ``"upstream"`` executes the preserved original implementation.
        ``"auto"`` resolves according to the promoted capability registry.

    Returns
    -------
    SmcData
        Input data with results stored in ``data.results["psmc"]``.
    """
    started = time.perf_counter()
    implementation = normalize_implementation(implementation)
    requested_capabilities = {
        name
        for name, enabled in {
            "bootstrap": bootstrap,
            "decode": decode is not None,
            "transition_cap": transition_cap is not None,
            "initial_params": initial_params is not None,
            "time_intervals": time_intervals is not None,
            "divergence": divergence_time is not None,
            "sequence_probability": sequence_probability,
            "simulation": simulate,
            "output": output_path is not None,
            "upstream_options": bool(upstream_options),
        }.items()
        if enabled
    }
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("psmc"),
        method_name="psmc",
        requested_capabilities=requested_capabilities or None,
    )
    warn_if_native_not_trusted("psmc", implementation_used)
    if implementation_used == "upstream":
        return _psmc_upstream(
            data,
            pattern=pattern,
            n_iterations=n_iterations,
            max_t=max_t,
            tr_ratio=tr_ratio,
            alpha=alpha,
            random_init=random_init,
            bootstrap=bootstrap,
            bootstrap_segment_length=bootstrap_segment_length,
            decode=decode,
            transition_cap=transition_cap,
            initial_params=initial_params,
            time_intervals=time_intervals,
            divergence_time=divergence_time,
            sequence_probability=sequence_probability,
            simulate=simulate,
            output_path=output_path,
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

    records = deepcopy(data.uns["records"])
    original_record_count = len(records)
    if bootstrap:
        records = split_psmc_records(records, bootstrap_segment_length)
        records = bootstrap_psmc_records(records, rng)
    sum_L, sum_n = _record_stats(records)
    if sum_L <= 0:
        raise ValueError("PSMC input contains no callable windows.")
    if sum_n >= sum_L:
        raise ValueError("PSMC requires at least one callable homozygous window.")
    window_size = data.window_size

    # Parse pattern
    par_map, n_free, n = parse_pattern(pattern)
    n_states = n + 1
    logger.info("pattern=%s n=%d n_free=%d n_states=%d", pattern, n, n_free, n_states)

    # Initialize parameters
    divergence = divergence_time is not None
    n_params = n_free + PSMC_N_PARAMS + int(divergence)
    if initial_params is not None:
        params = np.asarray(initial_params, dtype=np.float64).copy()
        if params.shape != (n_params,):
            raise ValueError(
                f"initial_params must contain exactly {n_params} values for pattern {pattern!r}."
            )
        if divergence and divergence_time is not None:
            params[-1] = divergence_time
    else:
        params = np.zeros(n_params, dtype=np.float64)
        theta = -np.log1p(-sum_n / sum_L)
        params[0] = theta
        params[1] = theta / tr_ratio
        params[2] = max_t
        lambda_stop = n_params - int(divergence)
        for k in range(PSMC_N_PARAMS, lambda_stop):
            params[k] = 1.0 + (rng.random() * 2.0 - 1.0) * random_init
            if params[k] < 0.1:
                params[k] = 0.1
        if divergence:
            params[-1] = float(divergence_time)

    # Time intervals
    explicit_intervals = None
    if time_intervals is not None:
        explicit_intervals = np.asarray(time_intervals, dtype=np.float64)
        if explicit_intervals.shape != (n + 1,):
            raise ValueError(f"time_intervals must contain exactly {n + 1} boundaries.")
        if explicit_intervals[0] != 0 or np.any(np.diff(explicit_intervals) <= 0):
            raise ValueError("time_intervals must start at zero and be strictly increasing.")
    t = compute_time_intervals(
        n,
        params[2],
        alpha,
        inp_ti=explicit_intervals,
    )

    # EM loop
    rounds: list[dict] = []

    # Record round 0 (initial parameters)
    hp0 = compute_hmm_params(params, par_map, n, t, divergence)
    lam0 = params[par_map + PSMC_N_PARAMS]
    n_recomb0 = sum_L / hp0.C_sigma
    rounds.append(
        {
            "round": 0,
            "theta": params[0],
            "rho": params[1],
            "max_t": params[2],
            "time": t[:n_states].copy(),
            "lambda": lam0,
            "pi": n_recomb0 * hp0.a0,
            "sigma": hp0.sigma.copy(),
            "params": params.copy(),
            **({"divergence_time": params[-1]} if divergence else {}),
        }
    )

    for i in range(n_iterations):
        logger.info("EM iteration %d/%d", i + 1, n_iterations)

        params, LL, Q0, Q1, post_sigma = _em_step(
            records,
            params,
            par_map,
            n,
            t,
            divergence=divergence,
            alpha=alpha,
            fixed_time_intervals=explicit_intervals is not None,
        )

        lam_i = params[par_map + PSMC_N_PARAMS]
        hp_i = compute_hmm_params(params, par_map, n, t, divergence)
        n_recomb_i = sum_L / hp_i.C_sigma

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
            "pi": n_recomb_i * hp_i.a0,
            "sigma": hp_i.sigma.copy(),
            "post_sigma": post_sigma,
            "params": params.copy(),
            **({"divergence_time": params[-1]} if divergence else {}),
        }
        rounds.append(rd)
        logger.info(
            "  LL=%.2f Q=%.4f->%.4f theta=%.6f rho=%.6f",
            LL,
            Q0,
            Q1,
            params[0],
            params[1],
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

    final_hmm = compute_hmm_params(params, par_map, n, t, divergence)
    decoded = (
        _decode_records(
            records,
            final_hmm,
            theta_final,
            mode=decode,
            transition_cap=transition_cap,
        )
        if decode is not None
        else None
    )
    sequence_probabilities = (
        _sequence_probabilities(records, final_hmm) if sequence_probability else None
    )
    simulated_records = _simulate_records(records, final_hmm, rng) if simulate else None
    artifacts: list[dict[str, Any]] = []
    if output_path is not None:
        native_output_path = Path(output_path)
        native_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_psmc_output(
            native_output_path,
            rounds,
            pattern=pattern,
            metadata={"implementation": "smckit-native"},
            decoded=decoded,
            sequence_probabilities=sequence_probabilities,
            simulated_records=simulated_records,
        )
        artifacts.append(
            {
                "path": str(native_output_path),
                "sha256": sha256_file(native_output_path),
                "kind": "psmc-output",
            }
        )
    source_path = data.uns.get("source_path")
    effective_args = {
        "pattern": pattern,
        "n_iterations": int(n_iterations),
        "max_t": float(max_t),
        "tr_ratio": float(tr_ratio),
        "alpha": float(alpha),
        "random_init": float(random_init),
        "bootstrap": bool(bootstrap),
        "bootstrap_segment_length": int(bootstrap_segment_length),
        "decode": decode,
        "transition_cap": transition_cap,
        "divergence_time": divergence_time,
        "sequence_probability": bool(sequence_probability),
        "simulate": bool(simulate),
        "time_intervals": explicit_intervals.tolist() if explicit_intervals is not None else None,
    }
    data.results["psmc"] = annotate_result(
        {
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
            "decode": decoded,
            "sequence_probabilities": sequence_probabilities,
            "simulated_records": simulated_records,
            "divergence_time": float(params[-1]) if divergence else None,
            "input_summary": {
                "records": len(records),
                "original_records": original_record_count,
                "callable_windows": sum_L,
                "heterozygous_windows": sum_n,
                "missing_windows": sum(len(record["codes"]) for record in records) - sum_L,
                "bootstrap": bootstrap,
            },
        },
        method_name="psmc",
        implementation_requested=implementation,
        implementation_used=implementation_used,
        effective_args=effective_args,
        input_paths=[str(source_path)] if source_path else None,
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        artifacts=artifacts,
    )
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
                fh.write(seq[start : start + 60] + "\n")


def _decoded_from_psmc_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rounds:
        return None
    final = rounds[-1]
    if final.get("decoded_segments"):
        return {
            "mode": "posterior",
            "time_centres": final.get("time_centres", []),
            "segments": final["decoded_segments"],
        }
    if final.get("decoded_full"):
        return {
            "mode": "full",
            "time_centres": final.get("time_centres", []),
            "rows": final["decoded_full"],
        }
    return None


def _psmc_upstream(
    data: SmcData,
    *,
    pattern: str,
    n_iterations: int,
    max_t: float,
    tr_ratio: float,
    alpha: float,
    random_init: float,
    bootstrap: bool,
    bootstrap_segment_length: int,
    decode: str | None,
    transition_cap: int | None,
    initial_params: np.ndarray | list[float] | None,
    time_intervals: np.ndarray | list[float] | None,
    divergence_time: float | None,
    sequence_probability: bool,
    simulate: bool,
    output_path: str | Path | None,
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
        "alpha": float(alpha),
        "random_init": float(random_init),
        "bootstrap": bool(bootstrap),
        "bootstrap_segment_length": int(bootstrap_segment_length),
        "decode": decode,
        "transition_cap": transition_cap,
        "divergence_time": divergence_time,
        "sequence_probability": bool(sequence_probability),
        "simulate": bool(simulate),
    }
    if upstream_options:
        effective_args.update(upstream_options)

    with tempfile.TemporaryDirectory(prefix="smckit-psmc-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.psmcfa"
        transformed_records = data.uns["records"]
        if bootstrap:
            transformed_records = split_psmc_records(
                deepcopy(transformed_records),
                bootstrap_segment_length,
            )
        if source_path and not bootstrap:
            input_path = Path(source_path)
        else:
            _write_psmcfa(transformed_records, input_path)

        temporary_output_path = tmpdir_path / "result.psmc"
        persistent_output_path = Path(output_path) if output_path is not None else None
        command_output_path = persistent_output_path or temporary_output_path
        if persistent_output_path is not None:
            persistent_output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(binary),
            "-N",
            str(int(n_iterations)),
            "-t",
            repr(float(max_t)),
            "-r",
            repr(float(tr_ratio)),
            "-l",
            repr(float(alpha)),
            "-I",
            repr(float(random_init)),
            "-p",
            pattern,
            "-o",
            str(command_output_path),
            str(input_path),
        ]
        if bootstrap:
            cmd.insert(-1, "-b")
        if decode is not None:
            if decode not in {"posterior", "full"}:
                raise ValueError("decode must be one of: None, 'posterior', 'full'.")
            cmd.insert(-1, "-d")
            if decode == "full":
                cmd.insert(-1, "-D")
        if transition_cap is not None:
            cmd[-1:-1] = ["-C", str(int(transition_cap))]
        if divergence_time is not None:
            cmd[-1:-1] = ["-T", repr(float(divergence_time))]
        if sequence_probability:
            cmd.insert(-1, "-s")
        if simulate:
            cmd.insert(-1, "-S")

        if initial_params is not None or time_intervals is not None:
            par_map, n_free, n = parse_pattern(pattern)
            del par_map
            params = (
                np.asarray(initial_params, dtype=np.float64)
                if initial_params is not None
                else None
            )
            expected = n_free + PSMC_N_PARAMS + int(divergence_time is not None)
            if params is None or params.shape != (expected,):
                raise ValueError(
                    "The upstream parameter-file path requires initial_params with "
                    f"exactly {expected} values."
                )
            intervals = (
                np.asarray(time_intervals, dtype=np.float64)
                if time_intervals is not None
                else None
            )
            if intervals is not None and intervals.shape != (n + 1,):
                raise ValueError(f"time_intervals must contain exactly {n + 1} boundaries.")
            parameter_path = tmpdir_path / "parameters.txt"
            file_params = params.copy()
            if intervals is not None:
                file_params[2] = -abs(file_params[2])
            with parameter_path.open("wt", encoding="utf-8") as handle:
                values = [pattern, *(repr(float(value)) for value in file_params[: n_free + 3])]
                if intervals is not None:
                    values.extend(repr(float(value)) for value in intervals)
                if divergence_time is not None:
                    values.append(repr(float(params[-1])))
                handle.write(" ".join(values) + "\n")
            cmd[-1:-1] = ["-i", str(parameter_path)]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Upstream PSMC backend failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        rounds = read_psmc_output(command_output_path)
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

        artifacts = []
        if persistent_output_path is not None:
            artifacts.append(
                {
                    "path": str(persistent_output_path),
                    "sha256": sha256_file(persistent_output_path),
                    "kind": "psmc-output",
                }
            )
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
                "decode": _decoded_from_psmc_rounds(rounds),
                "sequence_probabilities": final.get("sequence_probabilities"),
                "simulated_records": final.get("simulated_records"),
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
            method_name="psmc",
            implementation_requested=implementation_requested,
            implementation_used="upstream",
            effective_args=effective_args,
            input_paths=[str(source_path)] if source_path else None,
            artifacts=artifacts,
        )
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time
    return data


def psmc_bootstrap(
    data: SmcData,
    n_replicates: int = 100,
    *,
    segment_length: int = 500_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> SmcData:
    """Run independent native PSMC bootstrap replicates."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    rng = np.random.default_rng(seed)
    output_directory = Path(output_dir) if output_dir is not None else None
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
    replicates: list[dict[str, Any]] = []
    for index in range(n_replicates):
        replicate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        replicate_data = deepcopy(data)
        replicate_output = (
            output_directory / f"replicate-{index + 1:04d}.psmc"
            if output_directory is not None
            else None
        )
        psmc(
            replicate_data,
            bootstrap=True,
            bootstrap_segment_length=segment_length,
            seed=replicate_seed,
            output_path=replicate_output,
            implementation="native",
            **kwargs,
        )
        replicate_result = replicate_data.results["psmc"]
        replicates.append(
            {
                "replicate": index + 1,
                "seed": replicate_seed,
                "time": replicate_result["time"],
                "lambda": replicate_result["lambda"],
                "ne": replicate_result["ne"],
                "time_years": replicate_result["time_years"],
                "log_likelihood": replicate_result["log_likelihood"],
                "artifact": str(replicate_output) if replicate_output is not None else None,
            }
        )
    data.results["psmc_bootstrap"] = {
        "n_replicates": n_replicates,
        "segment_length": segment_length,
        "seed": seed,
        "replicates": replicates,
    }
    return data
