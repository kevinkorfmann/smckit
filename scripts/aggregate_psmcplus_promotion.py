#!/usr/bin/env python3
"""Combine paired core speed and typed peak-memory PSMC+ promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return payload


def aggregate_promotion(
    warm_core: dict[str, Any],
    end_to_end: dict[str, Any],
) -> dict[str, Any]:
    """Validate and combine the two intentionally distinct measurement scopes."""
    if warm_core.get("method") != "psmcplus" or end_to_end.get("method") != "psmcplus":
        raise ValueError("Promotion records must both belong to PSMC+.")
    if warm_core.get("source") != end_to_end.get("source"):
        raise ValueError("Promotion records must share one exact source state.")
    source = warm_core.get("source")
    if not source or source.get("clean") is not True:
        raise ValueError("Promotion evidence requires a clean source checkout.")
    if warm_core.get("upstream_commit") != end_to_end.get("upstream_commit"):
        raise ValueError("Promotion records must share one upstream commit.")
    if warm_core.get("input_sha256") != end_to_end.get("input_sha256"):
        raise ValueError("Promotion records must share one input checksum.")
    if warm_core.get("threads") != 1 or end_to_end.get("threads") != 1:
        raise ValueError("Promotion evidence requires exactly one numeric thread.")
    if int(warm_core.get("repetitions", 0)) < 5:
        raise ValueError("Promotion evidence requires at least five paired core repetitions.")
    if int(warm_core.get("bootstrap_replicates", 0)) < 20_000:
        raise ValueError("Promotion evidence requires at least 20,000 paired bootstrap samples.")
    if warm_core.get("pair_order") != ["native_then_upstream", "upstream_then_native"]:
        raise ValueError("Promotion core timings must use the counterbalanced pair order.")
    if end_to_end.get("runtime_design") != ("separate implementation processes; diagnostic only"):
        raise ValueError("End-to-end runtimes must be explicitly marked diagnostic.")

    memory_by_mode = {item["mode"]: item for item in end_to_end["comparisons"]}
    if set(memory_by_mode) != {"fit", "decode"}:
        raise ValueError("End-to-end evidence must contain fit and decode memory records.")
    comparisons = []
    for mode in ("fit", "decode"):
        speed = warm_core[mode]
        memory = memory_by_mode[mode]
        interval = speed["speedup_confidence_interval"]
        speed_passed = bool(speed["faster_with_confidence"] and float(interval[0]) > 1.0)
        memory_passed = bool(
            memory["memory_within_25_percent"] and float(memory["memory_ratio"]) <= 1.25
        )
        comparisons.append(
            {
                "mode": mode,
                "paired_warm_core": {
                    "repetitions": int(warm_core["repetitions"]),
                    "speedup": float(speed["speedup"]),
                    "speedup_confidence_interval": [float(interval[0]), float(interval[1])],
                    "confidence": float(speed["confidence"]),
                    "bootstrap_design": speed["bootstrap_design"],
                    "speed_gate_passed": speed_passed,
                },
                "typed_end_to_end": {
                    "native_cold_wall_seconds": memory["native_cold_wall_seconds"],
                    "native_peak_memory_median_bytes": memory["native_peak_memory_median_bytes"],
                    "upstream_peak_memory_median_bytes": memory[
                        "upstream_peak_memory_median_bytes"
                    ],
                    "memory_ratio": float(memory["memory_ratio"]),
                    "memory_gate_passed": memory_passed,
                    "runtime_claim_eligible": False,
                },
                "promotion_gate_passed": speed_passed and memory_passed,
            }
        )

    return {
        "schema_version": 1,
        "method": "psmcplus",
        "protocol_id": end_to_end["protocol_id"],
        "source": source,
        "upstream_commit": warm_core["upstream_commit"],
        "input_sha256": warm_core["input_sha256"],
        "threads": 1,
        "environment": {
            "core": warm_core["environment"],
            "typed_end_to_end": end_to_end["environment"],
        },
        "comparisons": comparisons,
        "performance_gate_passed": all(item["promotion_gate_passed"] for item in comparisons),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-core", type=Path, required=True)
    parser.add_argument("--end-to-end", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_promotion(_load(args.warm_core), _load(args.end_to_end))
    result["source_records"] = {
        "warm_core": {
            "path": str(args.warm_core.resolve()),
            "sha256": _sha256(args.warm_core),
        },
        "end_to_end": {
            "path": str(args.end_to_end.resolve()),
            "sha256": _sha256(args.end_to_end),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable benchmark record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["performance_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
