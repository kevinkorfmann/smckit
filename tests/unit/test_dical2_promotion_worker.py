"""Tests for the persistent diCal2 introgression benchmark worker."""

from __future__ import annotations

import importlib.util
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
