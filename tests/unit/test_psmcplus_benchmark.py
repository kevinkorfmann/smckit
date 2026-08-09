"""Fairness and provenance helpers for the paired PSMC+ benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_psmcplus_warm_core.py"
SPEC = importlib.util.spec_from_file_location("smckit_psmcplus_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_measurement_pairs_are_warmed_and_counterbalanced(monkeypatch) -> None:
    events: list[str] = []
    clock = iter(float(index) for index in range(100))
    monkeypatch.setattr(MODULE.time, "perf_counter", lambda: next(clock))

    def native() -> str:
        events.append("native")
        return "native-result"

    def upstream() -> str:
        events.append("upstream")
        return "upstream-result"

    native_times, upstream_times, native_value, upstream_value = MODULE._measure_paired(
        native,
        upstream,
        4,
    )

    assert events == [
        "native",
        "upstream",
        "native",
        "upstream",
        "upstream",
        "native",
        "native",
        "upstream",
        "upstream",
        "native",
    ]
    assert native_times == [1.0] * 4
    assert upstream_times == [1.0] * 4
    assert native_value == "native-result"
    assert upstream_value == "upstream-result"


def test_paired_bootstrap_uses_within_pair_speedups() -> None:
    summary = MODULE._paired_summary(
        [1.0, 2.0, 4.0, 8.0, 16.0],
        [2.0, 4.0, 8.0, 16.0, 32.0],
        bootstrap_replicates=1_000,
        seed=17,
    )

    assert summary["pairwise_speedups"] == [2.0] * 5
    assert summary["speedup"] == pytest.approx(2.0)
    assert summary["speedup_confidence_interval"] == pytest.approx([2.0, 2.0])
    assert summary["bootstrap_design"] == "paired median speedup"
    assert summary["faster_with_confidence"] is True


@pytest.mark.parametrize(
    ("native", "upstream"),
    [([1.0], [2.0]), ([1.0, 2.0], [2.0]), ([0.0, 1.0], [1.0, 2.0])],
)
def test_paired_summary_rejects_invalid_design(native, upstream) -> None:
    with pytest.raises(ValueError, match=r"Paired PSMC\+ timings"):
        MODULE._paired_summary(native, upstream, bootstrap_replicates=100, seed=1)
