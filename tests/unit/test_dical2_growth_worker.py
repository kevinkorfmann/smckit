"""Tests for the persistent diCal2 exponential-growth benchmark worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "benchmark_dical2_growth.py"
SPEC = importlib.util.spec_from_file_location("smckit_dical2_growth_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summary_records_fitted_growth_endpoint_and_optimizer_contract() -> None:
    summary = MODULE._summary(
        {
            "implementation": "native",
            "log_likelihood": -733.2181358824441,
            "best_params": [1.28, 2.00000003],
            "core_type": "ode",
            "resolved_options": {"number_iterations_mstep": 1},
        }
    )

    assert summary == {
        "implementation": "native",
        "log_likelihood": pytest.approx(-733.2181358824441),
        "best_params": pytest.approx([1.28, 2.00000003]),
        "core_type": "ode",
        "number_iterations_mstep": 1,
    }


def test_deterministic_growth_fixture_has_expected_samples_and_variants(
    tmp_path: Path,
) -> None:
    vcf_path, reference_path = MODULE._write_fixture(tmp_path)
    lines = vcf_path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#CHROM"))

    assert len(lines) > 5
    assert len(header.split("\t")) == 14
    assert reference_path.read_text(encoding="utf-8") == "A" * MODULE.SEQUENCE_LENGTH + "\n"
    assert len(MODULE._sha256(vcf_path)) == 64
