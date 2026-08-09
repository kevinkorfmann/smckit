"""Tests for the publication resource benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "workflow" / "publication" / "scripts" / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("smckit_publication_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_whole_process_benchmark_is_not_mislabeled_as_warm(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    payload = MODULE.run_benchmark(
        command=[sys.executable, "-c", "print('smckit benchmark')"],
        repetitions=3,
        threads=1,
        method="psmc",
        implementation="native",
        dataset="tiny",
        output=output,
        protocol_id="sha256:protocol",
        poll_seconds=0.001,
    )
    assert output.is_file()
    assert payload["protocol_id"] == "sha256:protocol"
    assert payload["schema_version"] == 2
    assert payload["record_sha256"]
    assert [record["temperature"] for record in payload["repetitions"]] == [
        "fresh_process",
        "fresh_process",
        "fresh_process",
        "fresh_process",
    ]
    assert payload["measurement_scope"] == "whole_process"
    assert payload["promotion_eligible"] is False
    assert payload["warm_repetitions"] == 0
    assert payload["fresh_process_repetitions"] == 4
    assert all(record["runtime_seconds"] > 0 for record in payload["repetitions"])
    assert all(record["peak_memory_bytes"] > 0 for record in payload["repetitions"])
    assert len({record["stdout_sha256"] for record in payload["repetitions"]}) == 1


def test_persistent_worker_separates_startup_cold_and_warm_calls(tmp_path) -> None:
    worker = """
import json
import sys
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["event"] == "close":
        print(json.dumps({"event": "closed"}), flush=True)
        break
    if request["event"] == "prepare":
        print(json.dumps({
            "event": "prepared",
            "repetition": request["repetition"],
        }), flush=True)
        continue
    print(json.dumps({
        "event": "result",
        "repetition": request["repetition"],
        "result": {"value": 42},
    }), flush=True)
"""
    output = tmp_path / "persistent.json"
    payload = MODULE.run_benchmark(
        command=[sys.executable, "-u", "-c", worker],
        repetitions=3,
        threads=1,
        method="smcpp",
        implementation="native",
        dataset="split-control-v1",
        measurement_component="inference_api_excluding_input_preparation",
        output=output,
        protocol_id="sha256:protocol",
        poll_seconds=0.001,
        persistent_jsonl=True,
        timeout_seconds=10.0,
    )

    assert payload["measurement_scope"] == "in_process_call"
    assert payload["warmup_semantics"] == "first_call_then_same_process"
    assert payload["promotion_eligible"] is True
    assert payload["measurement_component"] == "inference_api_excluding_input_preparation"
    assert payload["startup"]["runtime_seconds"] > 0
    assert payload["startup"]["peak_memory_bytes"] > 0
    assert [record["temperature"] for record in payload["repetitions"]] == [
        "cold",
        "warm",
        "warm",
        "warm",
    ]
    assert all(record["worker_result"] == {"value": 42} for record in payload["repetitions"])
    assert all(record["preparation_runtime_seconds"] > 0 for record in payload["repetitions"])


def test_benchmark_persists_failure_evidence(tmp_path) -> None:
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="failed"):
        MODULE.run_benchmark(
            command=[sys.executable, "-c", "raise SystemExit(7)"],
            repetitions=2,
            threads=1,
            method="psmc",
            implementation="upstream",
            dataset="tiny",
            output=output,
            poll_seconds=0.001,
        )
    assert output.is_file()


def test_persistent_benchmark_persists_protocol_failure_evidence(tmp_path) -> None:
    worker = """
import json
import sys
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["event"] == "prepare":
        raise SystemExit(7)
"""
    output = tmp_path / "failed-persistent.json"
    with pytest.raises(RuntimeError, match="content-addressed evidence"):
        MODULE.run_benchmark(
            command=[sys.executable, "-u", "-c", worker],
            repetitions=2,
            threads=1,
            method="smcpp",
            implementation="native",
            dataset="split-control-v1",
            output=output,
            persistent_jsonl=True,
            poll_seconds=0.001,
            timeout_seconds=10.0,
        )

    payload = json.loads(output.read_text())
    assert payload["record_sha256"]
    assert payload["promotion_eligible"] is False
    assert payload["failure"]["phase"] == "persistent_worker_protocol"
    assert payload["failure"]["error_type"] == "RuntimeError"


def test_benchmark_refuses_to_overwrite_immutable_record(tmp_path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("original\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.run_benchmark(
            command=[sys.executable, "-c", "print('should not run')"],
            repetitions=2,
            threads=1,
            method="psmc",
            implementation="native",
            dataset="tiny",
            output=output,
        )

    assert output.read_text() == "original\n"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": [], "repetitions": 2, "threads": 1},
        {"command": ["ok"], "repetitions": 1, "threads": 1},
        {"command": ["ok"], "repetitions": 2, "threads": 0},
        {
            "command": ["ok"],
            "repetitions": 2,
            "threads": 1,
            "timeout_seconds": 0,
        },
        {
            "command": ["ok"],
            "repetitions": 2,
            "threads": 1,
            "measurement_component": "",
        },
    ],
)
def test_benchmark_rejects_invalid_contract(kwargs, tmp_path) -> None:
    with pytest.raises(ValueError):
        MODULE.run_benchmark(
            **kwargs,
            method="psmc",
            implementation="native",
            dataset="tiny",
            output=tmp_path / "benchmark.json",
        )
