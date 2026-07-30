"""Tests for the publication resource benchmark harness."""

from __future__ import annotations

import importlib.util
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


def test_benchmark_records_cold_warm_resources_and_hashes(tmp_path) -> None:
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
    assert payload["record_sha256"]
    assert [record["temperature"] for record in payload["repetitions"]] == [
        "cold",
        "warm",
        "warm",
        "warm",
    ]
    assert payload["warm_repetitions"] == 3
    assert all(record["runtime_seconds"] > 0 for record in payload["repetitions"])
    assert all(record["peak_memory_bytes"] > 0 for record in payload["repetitions"])
    assert len({record["stdout_sha256"] for record in payload["repetitions"]}) == 1


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": [], "repetitions": 2, "threads": 1},
        {"command": ["ok"], "repetitions": 1, "threads": 1},
        {"command": ["ok"], "repetitions": 2, "threads": 0},
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
