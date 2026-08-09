"""Run a command repeatedly and emit an auditable resource benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _platform_payload() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }


def _write_hashed_payload(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["record_sha256"] = _sha256_bytes(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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


def _read_worker_message(
    process: subprocess.Popen[bytes],
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[dict[str, Any], bytes, float, int]:
    """Read one JSON-lines worker response while sampling process-tree RSS."""
    if process.stdout is None:
        raise RuntimeError("Persistent benchmark worker has no stdout pipe.")
    started = time.perf_counter()
    peak_memory = _resident_bytes(psutil.Process(process.pid))
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    "Persistent benchmark worker did not respond within "
                    f"{timeout_seconds:g} seconds."
                )
            peak_memory = max(peak_memory, _resident_bytes(psutil.Process(process.pid)))
            events = selector.select(min(poll_seconds, timeout_seconds - elapsed))
            if events:
                raw = process.stdout.readline()
                peak_memory = max(peak_memory, _resident_bytes(psutil.Process(process.pid)))
                if not raw:
                    return_code = process.poll()
                    raise RuntimeError(
                        "Persistent benchmark worker closed stdout before replying "
                        f"(return code {return_code})."
                    )
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "Persistent benchmark worker emitted invalid JSON."
                    ) from error
                if not isinstance(message, dict):
                    raise RuntimeError(
                        "Persistent benchmark worker response must be a JSON object."
                    )
                return message, raw, time.perf_counter() - started, peak_memory
            if process.poll() is not None:
                raise RuntimeError(
                    "Persistent benchmark worker exited before replying "
                    f"(return code {process.returncode})."
                )
    finally:
        selector.close()


def _run_persistent(
    command: list[str],
    *,
    environment: dict[str, str],
    repetitions: int,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Measure repeated calls inside one initialized JSON-lines worker."""
    with tempfile.TemporaryFile() as stderr_handle:
        process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            shell=False,
        )
        if process.stdin is None:
            process.kill()
            raise RuntimeError("Persistent benchmark worker has no stdin pipe.")
        try:
            ready, ready_raw, startup_seconds, startup_peak = _read_worker_message(
                process,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
            if ready.get("event") != "ready":
                raise RuntimeError("Persistent benchmark worker did not begin with a ready event.")
            startup = {
                "runtime_seconds": startup_seconds,
                "peak_memory_bytes": startup_peak,
                "response_sha256": _sha256_bytes(ready_raw),
                "response_bytes": len(ready_raw),
            }

            records = []
            for index in range(repetitions + 1):
                prepare_request = (
                    json.dumps(
                        {"event": "prepare", "repetition": index},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                process.stdin.write(prepare_request)
                process.stdin.flush()
                prepared, prepared_raw, preparation_runtime, preparation_peak = (
                    _read_worker_message(
                        process,
                        poll_seconds=poll_seconds,
                        timeout_seconds=timeout_seconds,
                    )
                )
                if prepared.get("event") != "prepared" or prepared.get("repetition") != index:
                    raise RuntimeError(
                        "Persistent benchmark worker returned an unexpected prepared event."
                    )
                request = (
                    json.dumps(
                        {"event": "run", "repetition": index},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                process.stdin.write(request)
                process.stdin.flush()
                message, raw, runtime, peak_memory = _read_worker_message(
                    process,
                    poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds,
                )
                if message.get("event") != "result" or message.get("repetition") != index:
                    raise RuntimeError(
                        "Persistent benchmark worker returned an unexpected result event."
                    )
                records.append(
                    {
                        "return_code": 0,
                        "runtime_seconds": runtime,
                        "peak_memory_bytes": peak_memory,
                        "stdout_sha256": _sha256_bytes(raw),
                        "stdout_bytes": len(raw),
                        "repetition": index,
                        "temperature": "cold" if index == 0 else "warm",
                        "preparation_runtime_seconds": preparation_runtime,
                        "preparation_peak_memory_bytes": preparation_peak,
                        "preparation_response_sha256": _sha256_bytes(prepared_raw),
                        "worker_result": message.get("result"),
                    }
                )

            process.stdin.write(b'{"event":"close"}\n')
            process.stdin.flush()
            closed, _, _, _ = _read_worker_message(
                process,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
            if closed.get("event") != "closed":
                raise RuntimeError("Persistent benchmark worker did not acknowledge close.")
            process.stdin.close()
            process.wait(timeout=timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Persistent benchmark worker exited with return code {process.returncode}."
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            stderr_handle.seek(0)
            stderr = stderr_handle.read()
        worker_stream = {
            "stderr_sha256": _sha256_bytes(stderr),
            "stderr_bytes": len(stderr),
        }
    return startup, records, worker_stream


def run_benchmark(
    *,
    command: list[str],
    repetitions: int,
    threads: int,
    method: str,
    implementation: str,
    dataset: str,
    output: Path,
    measurement_component: str = "command",
    protocol_id: str | None = None,
    poll_seconds: float = 0.01,
    persistent_jsonl: bool = False,
    timeout_seconds: float = 3_600.0,
) -> dict[str, Any]:
    """Benchmark *command* without shell interpolation and persist every run.

    With ``persistent_jsonl=True``, *command* must implement the documented
    JSON-lines worker protocol. Process startup is recorded separately, the
    fixture is prepared in a separately recorded phase before each call, the
    first in-process call is cold, and ``repetitions`` subsequent calls are
    genuinely warm in the same process. This is the only mode eligible for a
    native-default promotion assessment.

    The default whole-process mode remains useful for installation and CLI
    benchmarks. Every repetition is honestly labeled ``fresh_process``; those
    records must not be presented as warmed algorithmic/JIT measurements.
    """
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain one or more non-empty arguments.")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two.")
    if threads < 1:
        raise ValueError("threads must be positive.")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if implementation not in {"native", "upstream", "external"}:
        raise ValueError("implementation must be native, upstream, or external.")
    if not isinstance(measurement_component, str) or not measurement_component.strip():
        raise ValueError("measurement_component must be a non-empty string.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable benchmark record: {output}")

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

    startup = None
    worker_stream = None
    if persistent_jsonl:
        try:
            startup, records, worker_stream = _run_persistent(
                command,
                environment=environment,
                repetitions=repetitions,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            failure_payload = {
                "schema_version": 2,
                "protocol_id": protocol_id,
                "method": method,
                "implementation": implementation,
                "dataset": dataset,
                "measurement_component": measurement_component.strip(),
                "command": command,
                "threads": threads,
                "measurement_scope": "in_process_call",
                "warmup_semantics": "first_call_then_same_process",
                "promotion_eligible": False,
                "warm_repetitions": 0,
                "fresh_process_repetitions": 0,
                "repetitions": [],
                "platform": _platform_payload(),
                "failure": {
                    "phase": "persistent_worker_protocol",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }
            _write_hashed_payload(failure_payload, output)
            raise RuntimeError(
                "Persistent benchmark worker failed; content-addressed evidence "
                f"was written to {output}."
            ) from error
    else:
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
                    "temperature": "fresh_process",
                }
            )
            records.append(record)

    payload = {
        "schema_version": 2,
        "protocol_id": protocol_id,
        "method": method,
        "implementation": implementation,
        "dataset": dataset,
        "measurement_component": measurement_component.strip(),
        "command": command,
        "threads": threads,
        "measurement_scope": "in_process_call" if persistent_jsonl else "whole_process",
        "warmup_semantics": (
            "first_call_then_same_process" if persistent_jsonl else "none_fresh_processes"
        ),
        "promotion_eligible": persistent_jsonl,
        "warm_repetitions": repetitions if persistent_jsonl else 0,
        "fresh_process_repetitions": 0 if persistent_jsonl else repetitions + 1,
        "repetitions": records,
        "platform": _platform_payload(),
    }
    if startup is not None:
        payload["startup"] = startup
    if worker_stream is not None:
        payload["worker_stream"] = worker_stream
    _write_hashed_payload(payload, output)

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
    parser.add_argument("--measurement-component", default="command")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--protocol-id")
    parser.add_argument(
        "--persistent-jsonl",
        action="store_true",
        help="Use one initialized JSON-lines worker for cold and warm calls.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
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
        measurement_component=args.measurement_component,
        output=args.output,
        protocol_id=args.protocol_id,
        persistent_jsonl=args.persistent_jsonl,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
