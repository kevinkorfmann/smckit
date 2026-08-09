"""Contract tests for combined PSMC+ speed and memory evidence."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "aggregate_psmcplus_promotion.py"
SPEC = importlib.util.spec_from_file_location("smckit_psmcplus_promotion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _records() -> tuple[dict, dict]:
    source = {"clean": True, "commit": "abc", "status": []}
    speed = {
        "speedup": 2.0,
        "speedup_confidence_interval": [1.5, 2.5],
        "confidence": 0.95,
        "bootstrap_design": "paired median speedup",
        "faster_with_confidence": True,
    }
    warm = {
        "method": "psmcplus",
        "source": source,
        "upstream_commit": "upstream",
        "input_sha256": "input",
        "threads": 1,
        "repetitions": 5,
        "bootstrap_replicates": 20_000,
        "pair_order": ["native_then_upstream", "upstream_then_native"],
        "environment": {"platform": "test"},
        "fit": deepcopy(speed),
        "decode": deepcopy(speed),
    }
    memory = {
        "method": "psmcplus",
        "source": source,
        "upstream_commit": "upstream",
        "input_sha256": "input",
        "threads": 1,
        "protocol_id": "sha256:protocol",
        "runtime_design": "separate implementation processes; diagnostic only",
        "environment": {"system": "test"},
        "comparisons": [
            {
                "mode": mode,
                "native_cold_wall_seconds": 1.0,
                "native_peak_memory_median_bytes": 100,
                "upstream_peak_memory_median_bytes": 200,
                "memory_ratio": 0.5,
                "memory_within_25_percent": True,
            }
            for mode in ("fit", "decode")
        ],
    }
    return warm, memory


def test_combined_promotion_requires_paired_speed_and_memory_gates() -> None:
    warm, memory = _records()
    result = MODULE.aggregate_promotion(warm, memory)

    assert result["performance_gate_passed"] is True
    assert all(item["promotion_gate_passed"] for item in result["comparisons"])
    assert all(
        item["typed_end_to_end"]["runtime_claim_eligible"] is False
        for item in result["comparisons"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda warm, memory: warm.update(source={"clean": False}), "source state"),
        (lambda warm, memory: warm.update(repetitions=4), "five paired"),
        (lambda warm, memory: warm.update(bootstrap_replicates=19_999), "20,000"),
        (lambda warm, memory: warm.update(threads=2), "one numeric thread"),
        (
            lambda warm, memory: memory.update(runtime_design="whole process speed"),
            "diagnostic",
        ),
    ],
)
def test_combined_promotion_rejects_weakened_protocol(mutation, message) -> None:
    warm, memory = _records()
    mutation(warm, memory)
    with pytest.raises(ValueError, match=message):
        MODULE.aggregate_promotion(warm, memory)


def test_combined_promotion_fails_when_confidence_interval_includes_parity() -> None:
    warm, memory = _records()
    warm["decode"]["speedup_confidence_interval"] = [0.99, 2.0]
    result = MODULE.aggregate_promotion(warm, memory)

    assert result["performance_gate_passed"] is False
    assert result["comparisons"][1]["promotion_gate_passed"] is False
