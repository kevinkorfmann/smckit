"""Tests for MSMC2 helpers."""

import numpy as np

from smckit.tl._msmc import (
    _build_segments_for_pair,
    _emission_prob,
    _estimate_theta,
    emission_probs,
    quantile_boundaries,
)


def test_build_segments_preserves_ambiguous_observation():
    positions = np.array([10], dtype=np.int64)
    n_called = np.array([10], dtype=np.int64)
    pair_obs = np.array([1.5], dtype=np.float64)

    seg_pos, seg_obs = _build_segments_for_pair(positions, n_called, pair_obs)

    assert seg_pos.tolist() == [10]
    assert seg_obs.dtype == np.float64
    assert seg_obs.tolist() == [1.5]


def test_build_segments_distinguishes_skipped_ambiguous_from_missing():
    positions = np.array([10], dtype=np.int64)
    n_called = np.array([10], dtype=np.int64)

    skipped_pos, skipped_obs = _build_segments_for_pair(
        positions, n_called, np.array([-1.0], dtype=np.float64)
    )
    missing_pos, missing_obs = _build_segments_for_pair(
        positions, n_called, np.array([0.0], dtype=np.float64)
    )

    assert skipped_pos.tolist() == [10]
    assert skipped_obs.tolist() == [0.0]
    assert missing_pos.tolist() == [9, 10]
    assert missing_obs.tolist() == [1.0, 0.0]


def test_emission_prob_interpolates_ambiguous_observation():
    emission = np.array(
        [
            [1.0, 1.0],
            [0.8, 0.6],
            [0.2, 0.4],
        ]
    )

    assert _emission_prob(1.5, emission, 0) == 0.5
    assert _emission_prob(1.5, emission, 1) == 0.5


def test_estimate_theta_counts_ambiguous_hets_fractionally():
    segments = [
        {
            "positions": np.array([10, 20], dtype=np.int64),
            "n_called": np.array([10, 10], dtype=np.int64),
            "obs": {(0, 1): np.array([1.0, 1.5], dtype=np.float64)},
        }
    ]

    theta = _estimate_theta(segments, [(0, 1)])

    assert theta == 0.1


def test_emission_probs_are_clamped_to_valid_probabilities():
    boundaries = quantile_boundaries(40, 1.0 / 6.0)
    lambda_vec = np.ones(40, dtype=np.float64)
    lambda_vec[-1] = 1e-3
    lambda_vec[-2] = 1e-9

    e = emission_probs(boundaries, lambda_vec, 5.392e-05)

    assert np.all(e[1] >= 0.0)
    assert np.all(e[1] <= 1.0)
    assert np.all(e[2] >= 0.0)
    assert np.all(e[2] <= 1.0)
