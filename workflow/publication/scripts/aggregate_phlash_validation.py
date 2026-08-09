#!/usr/bin/env python3
"""Aggregate frozen per-replicate PHLASH accuracy and coverage records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from smckit._provenance import sha256_file

METRICS = (
    "log_integrated_trajectory_error",
    "log_root_mean_squared_error",
    "log_median_bias",
    "posterior_coverage",
    "log_time_weighted_posterior_coverage",
    "mean_log_credible_interval_width",
)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Aggregate metric values must be a non-empty finite vector.")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "quantile_0.025": float(np.quantile(array, 0.025)),
        "quantile_0.975": float(np.quantile(array, 0.975)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and aggregate a complete set of PHLASH replicate records."""
    if not records:
        raise ValueError("At least one PHLASH validation record is required.")
    if any(record.get("method") != "phlash" for record in records):
        raise ValueError("All validation records must belong to PHLASH.")
    protocol_ids = {record.get("protocol_id") for record in records}
    if len(protocol_ids) != 1:
        raise ValueError("PHLASH records must share one frozen protocol ID.")
    protocol_source_hashes = {record.get("protocol_source_sha256") for record in records}
    if None in protocol_source_hashes or len(protocol_source_hashes) != 1:
        raise ValueError("PHLASH records must share one frozen protocol source hash.")
    versions = {record["inference"]["phlash_version"] for record in records}
    if len(versions) != 1:
        raise ValueError("PHLASH records must use one package version.")
    keys = [(str(record["scenario"]), int(record["replicate"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate PHLASH scenario/replicate records are not allowed.")
    expectations = {
        (
            int(record["protocol_expectations"]["replicates_per_scenario"]),
            tuple(record["protocol_expectations"]["scenarios"]),
        )
        for record in records
    }
    if len(expectations) != 1:
        raise ValueError("PHLASH records disagree about the frozen replicate matrix.")
    expected_replicates, expected_scenarios = expectations.pop()
    expected_keys = {
        (scenario, replicate)
        for scenario in expected_scenarios
        for replicate in range(1, expected_replicates + 1)
    }
    if set(keys) != expected_keys:
        missing = sorted(expected_keys - set(keys))
        unexpected = sorted(set(keys) - expected_keys)
        raise ValueError(
            "PHLASH aggregate requires the complete frozen replicate matrix; "
            f"missing={missing}, unexpected={unexpected}."
        )
    credible_levels = {float(record["posterior"]["credible_level"]) for record in records}
    if len(credible_levels) != 1:
        raise ValueError("PHLASH records must share one nominal credible level.")

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scenario[str(record["scenario"])].append(record)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "replicates": len(group),
            "metrics": {
                metric: _summary([float(record["metrics"][metric]) for record in group])
                for metric in METRICS
            },
            "runtime_seconds": _summary(
                [float(record["inference"]["runtime_seconds"]) for record in group]
            ),
        }

    ordered = sorted(records, key=lambda record: (record["scenario"], record["replicate"]))
    return {
        "schema_version": 1,
        "method": "phlash",
        "protocol_id": protocol_ids.pop(),
        "protocol_source_sha256": protocol_source_hashes.pop(),
        "phlash_version": versions.pop(),
        "records": len(records),
        "complete_frozen_matrix": True,
        "replicates_per_scenario": expected_replicates,
        "nominal_credible_level": credible_levels.pop(),
        "scenarios": {
            scenario: summarize(sorted(group, key=lambda record: record["replicate"]))
            for scenario, group in sorted(by_scenario.items())
        },
        "overall": summarize(ordered),
        "input_records": [
            {"scenario": record["scenario"], "replicate": record["replicate"]}
            for record in ordered
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
    result = aggregate_records(records)
    result["input_records"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "scenario": record["scenario"],
            "replicate": record["replicate"],
        }
        for path, record in sorted(
            zip(args.records, records, strict=True),
            key=lambda item: (item[1]["scenario"], item[1]["replicate"]),
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
