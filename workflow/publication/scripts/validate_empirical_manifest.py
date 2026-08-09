#!/usr/bin/env python3
"""Validate and content-address public empirical dataset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MD5 = re.compile(r"[0-9a-f]{32}")
_STATUSES = {"source_pinned", "acquisition_ready", "ready", "authorization_required"}
_PAIRWISE_METHODS = {"psmc", "psmcplus", "msmc2", "esmc2"}
_REQUIRED_SOURCE_ROLES = {
    "aligned_reads",
    "aligned_reads_index",
    "reference",
    "reference_index",
    "sample_index",
    "sample_metadata",
    "data_reuse",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one empirical manifest and return its content-addressed summary."""
    _require(payload.get("schema_version") == 1, "schema_version must be 1")
    dataset_id = payload.get("dataset_id")
    _require(isinstance(dataset_id, str) and dataset_id.strip(), "dataset_id is required")
    status = payload.get("status")
    _require(status in _STATUSES, f"status must be one of {sorted(_STATUSES)}")

    organism = payload.get("organism", {})
    _require(
        isinstance(organism.get("scientific_name"), str)
        and isinstance(organism.get("ncbi_taxon_id"), int),
        "organism scientific_name and integer ncbi_taxon_id are required",
    )
    assembly = payload.get("assembly", {})
    _require(
        isinstance(assembly.get("name"), str) and isinstance(assembly.get("accession"), str),
        "assembly name and accession are required",
    )

    selection = payload.get("sample_selection", {})
    _require(
        isinstance(selection.get("policy"), str) and selection.get("policy").strip(),
        "sample_selection.policy is required",
    )
    samples = selection.get("samples")
    _require(isinstance(samples, list) and samples, "sample_selection.samples is required")
    sample_ids: list[str] = []
    for sample in samples:
        _require(isinstance(sample, dict), "each sample must be an object")
        sample_id = sample.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id, "each sample needs sample_id")
        accessions = sample.get("accessions")
        _require(isinstance(accessions, dict) and accessions, "each sample needs accessions")
        _require(
            all(
                isinstance(key, str) and isinstance(value, str) and value
                for key, value in accessions.items()
            ),
            "sample accession names and values must be non-empty strings",
        )
        sample_ids.append(sample_id)
    _require(len(sample_ids) == len(set(sample_ids)), "sample IDs must be unique")

    sources = payload.get("sources")
    _require(isinstance(sources, list) and sources, "sources are required")
    roles: set[str] = set()
    for source in sources:
        _require(isinstance(source, dict), "each source must be an object")
        role = source.get("role")
        url = source.get("url")
        _require(isinstance(role, str) and role, "each source needs a role")
        _require(
            isinstance(url, str) and url.startswith("https://"),
            f"source {role!r} must use an https URL",
        )
        roles.add(role)
        sha256 = source.get("sha256")
        md5 = source.get("md5")
        contig_md5 = source.get("contig_md5")
        _require(
            sha256 is not None or md5 is not None or contig_md5 is not None,
            f"source {role!r} needs a cryptographic checksum",
        )
        if sha256 is not None:
            _require(bool(_SHA256.fullmatch(sha256)), f"source {role!r} has invalid sha256")
        if md5 is not None:
            _require(bool(_MD5.fullmatch(md5)), f"source {role!r} has invalid md5")
        if contig_md5 is not None:
            _require(
                isinstance(contig_md5, dict)
                and contig_md5
                and all(
                    isinstance(contig, str) and bool(_MD5.fullmatch(digest))
                    for contig, digest in contig_md5.items()
                ),
                f"source {role!r} has invalid contig_md5",
            )
    if status in {"acquisition_ready", "ready"}:
        missing_roles = _REQUIRED_SOURCE_ROLES - roles
        _require(
            not missing_roles,
            f"execution-ready manifest lacks roles: {sorted(missing_roles)}",
        )

    analysis = payload.get("analysis", {})
    methods = analysis.get("methods")
    _require(isinstance(methods, list) and methods, "analysis.methods is required")
    _require(
        isinstance(analysis.get("smoke_scope"), str)
        and isinstance(analysis.get("publication_scope"), str),
        "analysis smoke_scope and publication_scope are required",
    )
    callability = analysis.get("callability", {})
    if _PAIRWISE_METHODS.intersection(methods):
        _require(
            callability.get("positive_mask_required") is True,
            "pairwise SMC manifests must require a positive callability mask",
        )
        _require(
            callability.get("variant_only_vcf_allowed") is False,
            "pairwise SMC manifests must reject variant-only VCF callability",
        )

    data_use = payload.get("data_use", {})
    _require(isinstance(data_use.get("status"), str), "data_use.status is required")
    blockers = payload.get("blockers")
    _require(isinstance(blockers, list), "blockers must be a list")
    if status == "ready":
        _require(not blockers, "ready manifests cannot retain blockers")
    else:
        _require(blockers, "non-ready manifests must state their blockers")
    if status == "authorization_required":
        _require(
            data_use["status"] == "authorization_required",
            "authorization_required manifests need matching data_use status",
        )

    digest = _canonical_hash(payload)
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "status": status,
        "manifest_id": f"sha256:{digest}",
        "samples": sample_ids,
        "methods": list(methods),
        "source_roles": sorted(roles),
        "blockers": list(blockers),
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    """Load and validate one manifest path."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path}: manifest root must be an object")
    return validate_manifest(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = [load_and_validate(path) for path in args.manifests]
    print(json.dumps({"manifests": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
