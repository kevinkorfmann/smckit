"""Tests for the persistent diCal2 introgression benchmark worker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "benchmark_dical2_introgression.py"
SPEC = importlib.util.spec_from_file_location("smckit_dical2_promotion_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EVIDENCE = (
    ROOT / "workflow" / "publication" / "evidence" / "dical2-introgression" / "sha256-c339dbb6"
)


def test_summary_records_fitted_endpoint_and_optimizer_contract() -> None:
    summary = MODULE._summary(
        {
            "implementation": "native",
            "log_likelihood": -12.5,
            "best_params": [0.08, 0.03],
            "core_type": "eigen",
            "resolved_options": {"number_iterations_mstep": 1},
        }
    )

    assert summary == {
        "implementation": "native",
        "log_likelihood": pytest.approx(-12.5),
        "best_params": pytest.approx([0.08, 0.03]),
        "core_type": "eigen",
        "number_iterations_mstep": 1,
    }


def test_deterministic_fixture_contains_variants_and_stable_reference(tmp_path: Path) -> None:
    vcf_path, reference_path = MODULE._write_fixture(tmp_path)

    assert len(vcf_path.read_text(encoding="utf-8").splitlines()) > 5
    assert reference_path.read_text(encoding="utf-8") == "A" * MODULE.SEQUENCE_LENGTH + "\n"
    assert len(MODULE._sha256(vcf_path)) == 64


def test_frozen_introgression_evidence_passes_capability_gate() -> None:
    aggregate = json.loads((EVIDENCE / "dical2-introgression-aggregate.json").read_text())
    comparison = aggregate["performance_comparisons"][0]
    native = json.loads((EVIDENCE / "dical2-introgression-native.json").read_text())
    upstream = json.loads((EVIDENCE / "dical2-introgression-upstream.json").read_text())

    assert comparison["promotable"] is True
    assert comparison["speedup_confidence_interval"][0] > 1.0
    assert comparison["memory_ratio"] <= 1.25
    assert comparison["warm_repetitions"] == 10
    for native_record, upstream_record in zip(
        native["repetitions"], upstream["repetitions"], strict=True
    ):
        native_result = native_record["worker_result"]
        upstream_result = upstream_record["worker_result"]
        assert native_result["best_params"] == pytest.approx(
            upstream_result["best_params"], abs=1e-14
        )
        assert native_result["log_likelihood"] == pytest.approx(
            upstream_result["log_likelihood"], abs=1e-5
        )


def test_frozen_introgression_evidence_checksums() -> None:
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        observed = hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest()
        assert observed == expected
