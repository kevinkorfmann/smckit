"""Numerically explicit validation and performance-promotion metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_vector(values: ArrayLike, name: str, *, minimum_size: int = 1) -> NDArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < minimum_size:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain only finite positive values.")
    return result


def _trajectory(time: ArrayLike, size: ArrayLike, label: str) -> tuple[NDArray, NDArray]:
    times = _positive_vector(time, f"{label}_time", minimum_size=2)
    sizes = _positive_vector(size, f"{label}_size", minimum_size=2)
    if times.size != sizes.size:
        raise ValueError(f"{label} time and size vectors must have equal length.")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{label}_time must be strictly increasing.")
    return times, sizes


def log_integrated_trajectory_error(
    truth_time: ArrayLike,
    truth_size: ArrayLike,
    estimate_time: ArrayLike,
    estimate_size: ArrayLike,
) -> float:
    """Mean absolute log-size error integrated over the shared log-time domain.

    Both trajectories are linearly interpolated in log(time)-log(size) space on
    the union of their knots. The result is zero for identical histories and is
    invariant to a common change in the time or population-size units.
    """
    truth_t, truth_n = _trajectory(truth_time, truth_size, "truth")
    estimate_t, estimate_n = _trajectory(estimate_time, estimate_size, "estimate")
    lower = max(truth_t[0], estimate_t[0])
    upper = min(truth_t[-1], estimate_t[-1])
    if lower >= upper:
        raise ValueError("Truth and estimate trajectories have no shared time interval.")

    knots = np.unique(
        np.concatenate(
            (
                [lower, upper],
                truth_t[(truth_t > lower) & (truth_t < upper)],
                estimate_t[(estimate_t > lower) & (estimate_t < upper)],
            )
        )
    )
    log_knots = np.log(knots)
    truth_log_size = np.interp(log_knots, np.log(truth_t), np.log(truth_n))
    estimate_log_size = np.interp(
        log_knots,
        np.log(estimate_t),
        np.log(estimate_n),
    )
    absolute_error = np.abs(estimate_log_size - truth_log_size)
    return float(np.trapezoid(absolute_error, log_knots) / (log_knots[-1] - log_knots[0]))


def parameter_relative_error(truth: float, estimate: float) -> float:
    """Return absolute relative error for a non-zero finite scalar truth."""
    if not np.isfinite(truth) or truth == 0:
        raise ValueError("truth must be finite and non-zero.")
    if not np.isfinite(estimate):
        raise ValueError("estimate must be finite.")
    return float(abs(estimate - truth) / abs(truth))


def per_site_likelihood_difference(
    native_log_likelihood: float,
    upstream_log_likelihood: float,
    n_sites: int,
) -> float:
    """Return absolute native-versus-upstream log-likelihood difference per site."""
    if not isinstance(n_sites, int) or isinstance(n_sites, bool) or n_sites <= 0:
        raise ValueError("n_sites must be a positive integer.")
    if not np.isfinite(native_log_likelihood) or not np.isfinite(upstream_log_likelihood):
        raise ValueError("Log likelihoods must be finite.")
    return float(abs(native_log_likelihood - upstream_log_likelihood) / n_sites)


def posterior_coverage(
    truth: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """Return the fraction of true values contained in closed credible intervals."""
    truth_array = np.asarray(truth, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if truth_array.ndim != 1 or truth_array.size == 0:
        raise ValueError("truth must be a non-empty one-dimensional vector.")
    if truth_array.shape != lower_array.shape or truth_array.shape != upper_array.shape:
        raise ValueError("truth, lower, and upper must have identical shapes.")
    if not all(np.all(np.isfinite(value)) for value in (truth_array, lower_array, upper_array)):
        raise ValueError("Coverage inputs must be finite.")
    if np.any(lower_array > upper_array):
        raise ValueError("Credible-interval lower bounds must not exceed upper bounds.")
    return float(np.mean((lower_array <= truth_array) & (truth_array <= upper_array)))


def bootstrap_speedup_interval(
    native_seconds: Sequence[float],
    upstream_seconds: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    random_seed: int = 1,
) -> tuple[float, float, float]:
    """Estimate the upstream/native median-runtime ratio and paired bootstrap CI."""
    native = _positive_vector(native_seconds, "native_seconds")
    upstream = _positive_vector(upstream_seconds, "upstream_seconds")
    if native.size != upstream.size:
        raise ValueError("Native and upstream timing vectors must have equal length.")
    if native.size < 2:
        raise ValueError("At least two paired timings are required.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one.")
    if not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")

    ratios = upstream / native
    estimate = float(np.median(ratios))
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, ratios.size, size=(resamples, ratios.size))
    bootstrapped = np.median(ratios[indices], axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrapped, (alpha, 1 - alpha))
    return estimate, float(lower), float(upper)


def promotion_assessment(
    native_seconds: Sequence[float],
    upstream_seconds: Sequence[float],
    *,
    native_peak_memory_bytes: int,
    upstream_peak_memory_bytes: int,
    confidence: float = 0.95,
    resamples: int = 10_000,
    random_seed: int = 1,
) -> dict[str, Any]:
    """Apply the roadmap's speed-confidence and memory promotion gates."""
    for name, value in (
        ("native_peak_memory_bytes", native_peak_memory_bytes),
        ("upstream_peak_memory_bytes", upstream_peak_memory_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    speedup, lower, upper = bootstrap_speedup_interval(
        native_seconds,
        upstream_seconds,
        confidence=confidence,
        resamples=resamples,
        random_seed=random_seed,
    )
    memory_ratio = native_peak_memory_bytes / upstream_peak_memory_bytes
    faster_with_confidence = lower > 1.0
    memory_within_limit = memory_ratio <= 1.25
    return {
        "schema_version": 1,
        "speedup": speedup,
        "speedup_confidence_interval": [lower, upper],
        "confidence": confidence,
        "memory_ratio": memory_ratio,
        "faster_with_confidence": faster_with_confidence,
        "memory_within_25_percent": memory_within_limit,
        "promotable": faster_with_confidence and memory_within_limit,
    }
