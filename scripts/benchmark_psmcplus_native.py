#!/usr/bin/env python3
"""Benchmark typed native or upstream PSMC+ with peak-RSS sampling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil

import smckit
from smckit._provenance import package_version, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("native", "upstream"), required=True)
    parser.add_argument("--mode", choices=("fit", "decode"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _rss_tree(process: psutil.Process) -> int:
    total = 0
    for candidate in [process, *process.children(recursive=True)]:
        try:
            total += candidate.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _measure(call) -> tuple[Any, float, int]:
    process = psutil.Process()
    peak = _rss_tree(process)
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak
        while not stop.wait(0.005):
            peak = max(peak, _rss_tree(process))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = call()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join()
        peak = max(peak, _rss_tree(process))
    return result, elapsed, peak


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=dirty",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status_lines = status.stdout.splitlines() if status.returncode == 0 else ["unknown"]
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else "unknown",
        "clean": not status_lines,
        "status": status_lines,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise SystemExit("repetitions must be positive and warmups cannot be negative")
    input_path = args.input.expanduser().resolve()
    options = smckit.tl.PSMCPlusOptions(
        mode=args.mode,
        number_time_windows=4,
        bin_size=100,
        iterations=1,
        likelihood_threshold=0,
        lambda_initial=[1, 1, 1, 1] if args.mode == "decode" else None,
        decode_downsample=1000,
        cores=1,
    )

    def run_once():
        data = smckit.io.read_multihetsep(input_path)
        with tempfile.TemporaryDirectory(prefix="smckit-psmcplus-benchmark-") as temporary:
            root = Path(temporary)
            output = root / ("posterior.txt" if args.mode == "decode" else "result_")
            marginal = root / "marginal.txt" if args.mode == "decode" else None
            smckit.tl.psmcplus(
                data,
                options=options,
                output_prefix=output,
                marginal_recombination_path=marginal,
                implementation=args.implementation,
                timeout=240,
            )
            return data.results["psmcplus"]

    warmups: list[dict[str, float | int]] = []
    for _ in range(args.warmups):
        result, elapsed, peak = _measure(run_once)
        warmups.append(
            {
                "wall_seconds": elapsed,
                "reported_seconds": float(result["provenance"]["runtime_seconds"]),
                "peak_rss_bytes": peak,
            }
        )

    measurements: list[dict[str, float | int]] = []
    for _ in range(args.repetitions):
        result, elapsed, peak = _measure(run_once)
        measurements.append(
            {
                "wall_seconds": elapsed,
                "reported_seconds": float(result["provenance"]["runtime_seconds"]),
                "peak_rss_bytes": peak,
            }
        )
    wall = [float(item["wall_seconds"]) for item in measurements]
    memory = [int(item["peak_rss_bytes"]) for item in measurements]
    try:
        input_label = str(input_path.relative_to(ROOT))
    except ValueError:
        input_label = str(input_path)
    payload = {
        "schema_version": 2,
        "method": "psmcplus",
        "source": _git_state(),
        "mode": args.mode,
        "implementation": args.implementation,
        "package_version": package_version(),
        "upstream_commit": "032168f2ceed3c0e46b7f214f890faf83dff41ae",
        "input": input_label,
        "input_sha256": sha256_file(input_path),
        "options": {
            "number_time_windows": 4,
            "bin_size": 100,
            "iterations": 1,
            "decode_downsample": 1000,
            "cores": 1,
        },
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "numeric_threads": 1,
        },
        "warmups": warmups,
        "measurements": measurements,
        "summary": {
            "wall_seconds_median": statistics.median(wall),
            "wall_seconds_min": min(wall),
            "wall_seconds_max": max(wall),
            "peak_rss_bytes_median": statistics.median(memory),
            "peak_rss_bytes_max": max(memory),
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
