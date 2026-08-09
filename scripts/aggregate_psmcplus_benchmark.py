#!/usr/bin/env python3
"""Aggregate frozen PSMC+ fit/decode benchmark records with bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("fit", "decode"):
        for implementation in ("native", "upstream"):
            parser.add_argument(f"--{implementation}-{mode}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison(native: dict, upstream: dict, *, seed: int, replicates: int) -> dict:
    native_times = np.asarray(
        [item["wall_seconds"] for item in native["measurements"]],
        dtype=np.float64,
    )
    upstream_times = np.asarray(
        [item["wall_seconds"] for item in upstream["measurements"]],
        dtype=np.float64,
    )
    native_memory = np.asarray(
        [item["peak_rss_bytes"] for item in native["measurements"]],
        dtype=np.float64,
    )
    upstream_memory = np.asarray(
        [item["peak_rss_bytes"] for item in upstream["measurements"]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled_native = rng.choice(
        native_times,
        size=(replicates, native_times.size),
        replace=True,
    )
    sampled_upstream = rng.choice(
        upstream_times,
        size=(replicates, upstream_times.size),
        replace=True,
    )
    speedups = np.median(sampled_upstream, axis=1) / np.median(sampled_native, axis=1)
    native_median = float(np.median(native_times))
    upstream_median = float(np.median(upstream_times))
    memory_ratio = float(np.median(native_memory) / np.median(upstream_memory))
    interval = np.quantile(speedups, [0.025, 0.975])
    warmup = native["warmups"][0] if native["warmups"] else None
    return {
        "mode": native["mode"],
        "warm_repetitions": int(native_times.size),
        "native_warm_median_seconds": native_median,
        "upstream_median_seconds": upstream_median,
        "speedup": upstream_median / native_median,
        "speedup_confidence_interval": [float(interval[0]), float(interval[1])],
        "confidence": 0.95,
        "faster_with_confidence": bool(interval[0] > 1.0),
        "native_peak_memory_median_bytes": int(np.median(native_memory)),
        "upstream_peak_memory_median_bytes": int(np.median(upstream_memory)),
        "memory_ratio": memory_ratio,
        "memory_within_25_percent": bool(memory_ratio <= 1.25),
        "native_cold_wall_seconds": None if warmup is None else warmup["wall_seconds"],
        "promotable_performance": bool(interval[0] > 1.0 and memory_ratio <= 1.25),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("bootstrap-replicates must be positive")
    records: dict[str, dict[str, dict]] = {}
    sources: dict[str, dict[str, str]] = {}
    for mode in ("fit", "decode"):
        records[mode] = {}
        sources[mode] = {}
        for implementation in ("native", "upstream"):
            path = getattr(args, f"{implementation}_{mode}").resolve()
            record = _load(path)
            if record["mode"] != mode or record["implementation"] != implementation:
                raise ValueError(f"Unexpected benchmark identity in {path}.")
            records[mode][implementation] = record
            sources[mode][implementation] = _sha256(path)
    comparisons = [
        _comparison(
            records[mode]["native"],
            records[mode]["upstream"],
            seed=args.seed + index,
            replicates=args.bootstrap_replicates,
        )
        for index, mode in enumerate(("fit", "decode"))
    ]
    payload = {
        "schema_version": 1,
        "method": "psmcplus",
        "protocol_id": "sha256:c339dbb68e7ec26c721d909916edea5e388d77a60f03c04847e9daaa5cf560dd",
        "dataset": "pinned-upstream-constant-population",
        "threads": 1,
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "source_sha256": sources,
        "comparisons": comparisons,
        "performance_gate_passed": all(item["promotable_performance"] for item in comparisons),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
