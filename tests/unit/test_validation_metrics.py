"""Tests for frozen parity and publication metrics."""

from __future__ import annotations

import numpy as np
import pytest

from smckit.validation import (
    bootstrap_speedup_interval,
    log_integrated_trajectory_error,
    parameter_relative_error,
    per_site_likelihood_difference,
    posterior_coverage,
    promotion_assessment,
)


def test_log_integrated_error_is_zero_for_identical_trajectory() -> None:
    times = [1, 10, 100]
    sizes = [10_000, 5_000, 20_000]
    assert log_integrated_trajectory_error(times, sizes, times, sizes) == pytest.approx(0)


def test_log_integrated_error_has_expected_constant_ratio() -> None:
    error = log_integrated_trajectory_error(
        [1, 10, 100],
        [10_000, 10_000, 10_000],
        [1, 100],
        [20_000, 20_000],
    )
    assert error == pytest.approx(np.log(2))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([1, 1], [1, 1], [1, 2], [1, 1]), "strictly increasing"),
        (([1, 2], [1, 1], [3, 4], [1, 1]), "no shared"),
        (([0, 1], [1, 1], [1, 2], [1, 1]), "positive"),
    ],
)
def test_log_integrated_error_rejects_invalid_trajectories(args, message) -> None:
    with pytest.raises(ValueError, match=message):
        log_integrated_trajectory_error(*args)


def test_scalar_likelihood_and_coverage_metrics() -> None:
    assert parameter_relative_error(100, 110) == pytest.approx(0.1)
    assert per_site_likelihood_difference(-100.0, -100.2, 100) == pytest.approx(0.002)
    assert posterior_coverage([1, 2, 3], [0, 2, 4], [1, 3, 5]) == pytest.approx(2 / 3)


def test_paired_bootstrap_is_deterministic_and_detects_speedup() -> None:
    first = bootstrap_speedup_interval(
        [1.0, 1.1, 0.9, 1.0, 1.05],
        [2.0, 2.1, 1.8, 2.2, 2.0],
        resamples=1_000,
        random_seed=72,
    )
    second = bootstrap_speedup_interval(
        [1.0, 1.1, 0.9, 1.0, 1.05],
        [2.0, 2.1, 1.8, 2.2, 2.0],
        resamples=1_000,
        random_seed=72,
    )
    assert first == second
    assert first[1] > 1


def test_promotion_requires_both_speed_and_memory_gates() -> None:
    accepted = promotion_assessment(
        [1, 1.1, 0.9, 1.0, 1.05],
        [2, 2.1, 1.8, 2.2, 2.0],
        native_peak_memory_bytes=120,
        upstream_peak_memory_bytes=100,
        resamples=1_000,
    )
    assert accepted["promotable"] is True
    rejected = promotion_assessment(
        [1, 1.1, 0.9, 1.0, 1.05],
        [2, 2.1, 1.8, 2.2, 2.0],
        native_peak_memory_bytes=126,
        upstream_peak_memory_bytes=100,
        resamples=1_000,
    )
    assert rejected["faster_with_confidence"] is True
    assert rejected["promotable"] is False


@pytest.mark.parametrize(
    "call",
    [
        lambda: parameter_relative_error(0, 1),
        lambda: per_site_likelihood_difference(1, 2, 0),
        lambda: posterior_coverage([1], [2], [1]),
        lambda: bootstrap_speedup_interval([1], [2]),
    ],
)
def test_metrics_reject_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
