"""Validate and aggregate immutable publication benchmark/evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from smckit.validation import promotion_assessment


def _canonical_hash(payload: dict[str, Any], hash_field: str) -> str:
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_hashed_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"{path} is not a supported schema-version 1 or 2 record.")
    observed = payload.get("record_sha256")
    expected = _canonical_hash(payload, "record_sha256")
    if observed != expected:
        raise ValueError(f"{path} failed its record_sha256 integrity check.")
    return payload


def _platform_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    platform = payload.get("platform")
    if not isinstance(platform, dict):
        raise ValueError("Benchmark record has no platform mapping.")
    platform_fields = ("system", "machine", "processor")
    values = tuple(str(platform.get(key, "")).strip() for key in platform_fields)
    if not values[0] or not values[1]:
        raise ValueError("Benchmark platform must include system and machine.")
    return values


def _benchmark_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("protocol_id"),
        str(payload.get("method", "")),
        str(payload.get("dataset", "")),
        str(payload.get("measurement_component", "")),
        int(payload.get("threads", 0)),
        *_platform_key(payload),
    )


def _validate_benchmark(payload: dict[str, Any], required_warm_repetitions: int) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("Publication benchmarks require schema version 2.")
    if payload.get("measurement_scope") != "in_process_call":
        raise ValueError("Promotion benchmarks require warmed calls in one initialized process.")
    if payload.get("warmup_semantics") != "first_call_then_same_process":
        raise ValueError("Benchmark warmup semantics are not promotion-safe.")
    if payload.get("promotion_eligible") is not True:
        raise ValueError("Benchmark record is not marked promotion eligible.")
    if (
        not isinstance(payload.get("measurement_component"), str)
        or not payload["measurement_component"].strip()
    ):
        raise ValueError("Benchmark measurement_component must be non-empty.")
    startup = payload.get("startup")
    if not isinstance(startup, dict):
        raise ValueError("Promotion benchmark must record process startup separately.")
    for key in ("runtime_seconds", "peak_memory_bytes"):
        value = startup.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Benchmark startup {key} must be positive.")
    if payload.get("implementation") not in {"native", "upstream", "external"}:
        raise ValueError("Benchmark implementation is invalid.")
    if int(payload.get("threads", 0)) < 1:
        raise ValueError("Benchmark threads must be positive.")
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list):
        raise ValueError("Benchmark repetitions must be a list.")
    cold = [item for item in repetitions if item.get("temperature") == "cold"]
    warm = [item for item in repetitions if item.get("temperature") == "warm"]
    if len(cold) != 1 or len(warm) < required_warm_repetitions:
        raise ValueError(
            "Benchmark must contain one cold run and at least "
            f"{required_warm_repetitions} warmed repetitions."
        )
    for repetition in repetitions:
        if repetition.get("return_code") != 0:
            raise ValueError("Failed benchmark repetitions cannot enter publication aggregates.")
        for key in ("runtime_seconds", "peak_memory_bytes"):
            value = repetition.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Benchmark {key} must be positive.")


def _validate_accuracy(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Accuracy evaluations require schema version 1.")
    if payload.get("evaluation_kind") not in {"parity", "simulation", "empirical"}:
        raise ValueError("Accuracy evaluation_kind must be parity, simulation, or empirical.")
    if payload.get("implementation") not in {"native", "upstream", "external"}:
        raise ValueError("Accuracy implementation is invalid.")
    for key in ("method", "dataset"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Accuracy record requires a non-empty {key}.")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Accuracy record must include one or more metrics.")
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Accuracy metrics must be named, finite, and non-negative.")


def aggregate_publication_results(
    *,
    benchmark_paths: list[Path],
    accuracy_paths: list[Path],
    output: Path,
    required_warm_repetitions: int = 5,
    bootstrap_resamples: int = 10_000,
    random_seed: int = 1729,
) -> dict[str, Any]:
    """Create a deterministic aggregate without accepting incomplete timing evidence."""
    if required_warm_repetitions < 2:
        raise ValueError("At least two warmed repetitions are required.")
    if not benchmark_paths and not accuracy_paths:
        raise ValueError("At least one benchmark or accuracy record is required.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable aggregate record: {output}")

    benchmarks = []
    seen_benchmarks: set[tuple[Any, ...]] = set()
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(Path(item) for item in benchmark_paths):
        payload = _read_hashed_record(path)
        _validate_benchmark(payload, required_warm_repetitions)
        key = _benchmark_key(payload)
        identity = (*key, payload["implementation"])
        if identity in seen_benchmarks:
            raise ValueError(f"Duplicate benchmark record for {identity}.")
        seen_benchmarks.add(identity)
        grouped[key][payload["implementation"]] = payload
        benchmarks.append(
            {
                "method": payload["method"],
                "dataset": payload["dataset"],
                "measurement_component": payload["measurement_component"],
                "implementation": payload["implementation"],
                "threads": payload["threads"],
                "platform": payload["platform"],
                "record_sha256": payload["record_sha256"],
            }
        )

    comparisons = []
    for key, implementations in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if not {"native", "upstream"} <= implementations.keys():
            continue
        native = implementations["native"]
        upstream = implementations["upstream"]
        native_warm = [
            float(item["runtime_seconds"])
            for item in native["repetitions"]
            if item["temperature"] == "warm"
        ]
        upstream_warm = [
            float(item["runtime_seconds"])
            for item in upstream["repetitions"]
            if item["temperature"] == "warm"
        ]
        if len(native_warm) != len(upstream_warm):
            raise ValueError("Paired native/upstream benchmarks need equal warmed replication.")
        native_memory = max(
            int(item["peak_memory_bytes"])
            for item in native["repetitions"]
            if item["temperature"] == "warm"
        )
        upstream_memory = max(
            int(item["peak_memory_bytes"])
            for item in upstream["repetitions"]
            if item["temperature"] == "warm"
        )
        assessment = promotion_assessment(
            native_warm,
            upstream_warm,
            native_peak_memory_bytes=native_memory,
            upstream_peak_memory_bytes=upstream_memory,
            resamples=bootstrap_resamples,
            random_seed=random_seed,
        )
        native_cold = next(
            float(item["runtime_seconds"])
            for item in native["repetitions"]
            if item["temperature"] == "cold"
        )
        upstream_cold = next(
            float(item["runtime_seconds"])
            for item in upstream["repetitions"]
            if item["temperature"] == "cold"
        )
        native_startup = float(native["startup"]["runtime_seconds"])
        upstream_startup = float(upstream["startup"]["runtime_seconds"])
        comparisons.append(
            {
                "protocol_id": key[0],
                "method": key[1],
                "dataset": key[2],
                "measurement_component": key[3],
                "threads": key[4],
                "platform": {
                    "system": key[5],
                    "machine": key[6],
                    "processor": key[7],
                },
                "warm_repetitions": len(native_warm),
                "native_startup_seconds": native_startup,
                "upstream_startup_seconds": upstream_startup,
                "startup_runtime_ratio_upstream_over_native": (upstream_startup / native_startup),
                "native_cold_seconds": native_cold,
                "upstream_cold_seconds": upstream_cold,
                "cold_runtime_ratio_upstream_over_native": upstream_cold / native_cold,
                "native_warm_peak_memory_bytes": native_memory,
                "upstream_warm_peak_memory_bytes": upstream_memory,
                **assessment,
            }
        )

    accuracy = []
    for path in sorted(Path(item) for item in accuracy_paths):
        payload = _read_hashed_record(path)
        _validate_accuracy(payload)
        accuracy.append(payload)

    protocols = {
        value.get("protocol_id")
        for value in [*(_read_hashed_record(path) for path in benchmark_paths), *accuracy]
        if value.get("protocol_id") is not None
    }
    if len(protocols) > 1:
        raise ValueError("All publication records must share one protocol_id.")

    result = {
        "schema_version": 1,
        "protocol_id": next(iter(protocols), None),
        "required_warm_repetitions": required_warm_repetitions,
        "benchmark_records": benchmarks,
        "performance_comparisons": comparisons,
        "accuracy_records": sorted(
            accuracy,
            key=lambda item: (
                item["evaluation_kind"],
                item["method"],
                item["dataset"],
                int(item.get("replicate", 0)),
            ),
        ),
    }
    result["input_record_sha256"] = sorted(
        [item["record_sha256"] for item in benchmarks]
        + [item["record_sha256"] for item in accuracy]
    )
    result["aggregate_sha256"] = _canonical_hash(result, "aggregate_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--accuracy", type=Path, action="append", default=[])
    parser.add_argument("--required-warm-repetitions", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate_publication_results(
        benchmark_paths=args.benchmark,
        accuracy_paths=args.accuracy,
        output=args.output,
        required_warm_repetitions=args.required_warm_repetitions,
        bootstrap_resamples=args.bootstrap_resamples,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
