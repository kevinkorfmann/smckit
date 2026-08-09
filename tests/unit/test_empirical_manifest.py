"""Tests for accession-recorded empirical manifests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "validate_empirical_manifest.py"
MANIFEST = ROOT / "workflow" / "publication" / "empirical" / "1000genomes-na12878-grch38.json"
SPEC = importlib.util.spec_from_file_location("smckit_empirical_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_human_manifest_is_valid_deterministic_and_not_overpromoted() -> None:
    first = MODULE.load_and_validate(MANIFEST)
    second = MODULE.load_and_validate(MANIFEST)

    assert first == second
    assert first["manifest_id"].startswith("sha256:")
    assert first["status"] == "source_pinned"
    assert first["samples"] == ["NA12878"]
    assert "psmcplus" in first["methods"]
    assert first["blockers"]


def test_pairwise_manifest_cannot_treat_variant_only_vcf_as_callable() -> None:
    payload = _payload()
    payload["analysis"]["callability"]["variant_only_vcf_allowed"] = True

    with pytest.raises(ValueError, match="reject variant-only VCF"):
        MODULE.validate_manifest(payload)


def test_ready_status_requires_sources_and_zero_blockers() -> None:
    payload = _payload()
    payload["status"] = "ready"

    with pytest.raises(ValueError, match="ready manifests cannot retain blockers"):
        MODULE.validate_manifest(payload)

    payload["blockers"] = []
    payload["sources"] = [
        source for source in payload["sources"] if source["role"] != "aligned_reads_index"
    ]
    with pytest.raises(ValueError, match="lacks roles"):
        MODULE.validate_manifest(payload)


def test_manifest_hash_changes_with_scientific_controls() -> None:
    payload = _payload()
    original = MODULE.validate_manifest(payload)["manifest_id"]
    modified = copy.deepcopy(payload)
    modified["analysis"]["callability"]["minimum_mapping_quality"] = 30

    assert MODULE.validate_manifest(modified)["manifest_id"] != original
