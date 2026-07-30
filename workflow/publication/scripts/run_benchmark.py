"""Run a command repeatedly and emit an auditable resource benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resident_bytes(process: psutil.Process) -> int:
    total = 0
    try:
        total += process.memory_info().rss
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return total
    for child in children:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _run_once(
    command: list[str],
    *,
    environment: dict[str, str],
    poll_seconds: float,
) -> dict[str, Any]:
    with (
        tempfile.TemporaryFile() as stdout_handle,
        tempfile.TemporaryFile() as stderr_handle,
    ):
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
        )
        observed = psutil.Process(process.pid)
        peak_memory = _resident_bytes(observed)
        while process.poll() is None:
            peak_memory = max(peak_memory, _resident_bytes(observed))
            time.sleep(poll_seconds)
        return_code = process.wait()
        peak_memory = max(peak_memory, _resident_bytes(observed))
        runtime = time.perf_counter() - started
        stdout_handle.seek(0)
        stderr_handle.seek(0)
        stdout = stdout_handle.read()
        stderr = stderr_handle.read()
    return {
        "return_code": return_code,
        "runtime_seconds": runtime,
        "peak_memory_bytes": peak_memory,
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
    }


def run_benchmark(
    *,
    command: list[str],
    repetitions: int,
    threads: int,
    method: str,
    implementation: str,
    dataset: str,
    output: Path,
    protocol_id: str | None = None,
    poll_seconds: float = 0.01,
) -> dict[str, Any]:
    """Benchmark *command* without shell interpolation and persist every repetition.

    ``repetitions`` is the number of independent warmed measurements. A
    separate, uncounted cold measurement is always recorded first so JIT and
    process-startup costs cannot silently reduce the protocol's replication.
    """
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain one or more non-empty arguments.")
    if repetitions < 2:
        raise ValueError("repetitions must contain at least two warmed measurements.")
    if threads < 1:
        raise ValueError("threads must be positive.")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    if implementation not in {"native", "upstream", "external"}:
        raise ValueError("implementation must be native, upstream, or external.")

    environment = os.environ.copy()
    thread_value = str(threads)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = thread_value

    records = []
    for index in range(repetitions + 1):
        record = _run_once(
            command,
            environment=environment,
            poll_seconds=poll_seconds,
        )
        record.update(
            {
                "repetition": index,
                "temperature": "cold" if index == 0 else "warm",
            }
        )
        records.append(record)

    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "method": method,
        "implementation": implementation,
        "dataset": dataset,
        "command": command,
        "threads": threads,
        "warm_repetitions": repetitions,
        "repetitions": records,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["record_sha256"] = _sha256_bytes(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [record for record in records if record["return_code"] != 0]
    if failed:
        raise RuntimeError(
            f"Benchmark command failed in {len(failed)} of {repetitions + 1} runs; "
            f"evidence was written to {output}."
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--implementation",
        choices=("native", "upstream", "external"),
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--protocol-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    run_benchmark(
        command=command,
        repetitions=args.repetitions,
        threads=args.threads,
        method=args.method,
        implementation=args.implementation,
        dataset=args.dataset,
        output=args.output,
        protocol_id=args.protocol_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
