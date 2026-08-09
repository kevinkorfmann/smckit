"""Tests for the persistent SMC++ promotion benchmark worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "benchmark_smcpp_split.py"
SPEC = importlib.util.spec_from_file_location("smckit_smcpp_promotion_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summary_accepts_normalized_native_split_result() -> None:
    summary = MODULE._summary(
        {
            "analysis": "split",
            "implementation": "native",
            "split": 0.5,
            "log_likelihood": -12.0,
            "joint_emission_sum": 1.0,
            "population_models": [
                {"population": "a", "values": np.array([1.0, 2.0])},
                {"population": "b", "scale": np.float64(3.0)},
            ],
        }
    )

    assert summary["implementation"] == "native"
    assert summary["joint_emission_sum"] == pytest.approx(1.0)
    assert len(summary["model_sha256"]) == 64


def test_summary_accepts_normalized_upstream_split_result() -> None:
    summary = MODULE._summary(
        {
            "analysis": "split",
            "implementation": "upstream",
            "split": 0.5,
            "log_likelihood": -12.0,
            "population_models": [{"population": "a"}, {"population": "b"}],
        }
    )

    assert summary["implementation"] == "upstream"
    assert "joint_emission_sum" not in summary
    assert "log_scale" not in summary


def test_summary_requires_normalized_population_models() -> None:
    with pytest.raises(KeyError, match="population models"):
        MODULE._summary(
            {
                "analysis": "split",
                "implementation": "native",
                "split": 0.5,
                "log_likelihood": -12.0,
            }
        )
