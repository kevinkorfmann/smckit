"""Tests for immutable publication aggregation and Figure 1 rendering."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _load_script(name: str):
    path = ROOT / "workflow" / "publication" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"smckit_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AGGREGATE = _load_script("aggregate_results")
PLOT = _load_script("plot_figure1")


def _hash(payload: dict) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "record_sha256": hashlib.sha256(canonical).hexdigest()}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _benchmark(implementation: str, runtime: float) -> dict:
    repetitions = [
        {
            "return_code": 0,
            "runtime_seconds": runtime * 1.5,
            "peak_memory_bytes": 120,
            "temperature": "cold",
        }
    ]
    repetitions.extend(
        {
            "return_code": 0,
            "runtime_seconds": runtime * multiplier,
            "peak_memory_bytes": 100 if implementation == "upstream" else 110,
            "temperature": "warm",
        }
        for multiplier in (0.98, 0.99, 1.0, 1.01, 1.02)
    )
    return _hash(
        {
            "schema_version": 1,
            "protocol_id": "sha256:protocol",
            "method": "psmc",
            "implementation": implementation,
            "dataset": "constant-1",
            "command": ["smckit"],
            "threads": 1,
            "warm_repetitions": 5,
            "repetitions": repetitions,
            "platform": {
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
                "processor": "test-cpu",
            },
        }
    )


def _accuracy(kind: str, replicate: int, metric: str, value: float) -> dict:
    return _hash(
        {
            "schema_version": 1,
            "protocol_id": "sha256:protocol",
            "method": "psmc",
            "implementation": "native",
            "dataset": f"{kind}-{replicate}",
            "evaluation_kind": kind,
            "replicate": replicate,
            "metrics": {metric: value},
        }
    )


def test_aggregate_and_render_primary_figure(tmp_path) -> None:
    native = _write(tmp_path / "native.json", _benchmark("native", 1.0))
    upstream = _write(tmp_path / "upstream.json", _benchmark("upstream", 2.0))
    accuracy = [
        _write(tmp_path / "parity.json", _accuracy("parity", 1, "parity_error", 1e-5)),
        _write(
            tmp_path / "simulation.json",
            _accuracy("simulation", 1, "trajectory_error", 0.12),
        ),
        _write(
            tmp_path / "empirical.json",
            _accuracy("empirical", 1, "trajectory_error", 0.18),
        ),
    ]
    aggregate_path = tmp_path / "aggregate.json"
    payload = AGGREGATE.aggregate_publication_results(
        benchmark_paths=[native, upstream],
        accuracy_paths=accuracy,
        output=aggregate_path,
        bootstrap_resamples=500,
    )
    assert payload["performance_comparisons"][0]["promotable"] is True
    assert payload["performance_comparisons"][0]["warm_repetitions"] == 5
    assert payload["aggregate_sha256"]

    outputs = PLOT.plot_figure1(aggregate_path, tmp_path / "figure1")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".tiff"}
    assert all(path.stat().st_size > 1_000 for path in outputs)


def test_aggregate_rejects_tampering_and_insufficient_replication(tmp_path) -> None:
    payload = _benchmark("native", 1.0)
    payload["dataset"] = "tampered"
    tampered = _write(tmp_path / "tampered.json", payload)
    with pytest.raises(ValueError, match="integrity"):
        AGGREGATE.aggregate_publication_results(
            benchmark_paths=[tampered],
            accuracy_paths=[],
            output=tmp_path / "aggregate.json",
        )

    short = _benchmark("native", 1.0)
    short["repetitions"] = short["repetitions"][:-1]
    short.pop("record_sha256")
    short = _hash(short)
    with pytest.raises(ValueError, match="warmed"):
        AGGREGATE.aggregate_publication_results(
            benchmark_paths=[_write(tmp_path / "short.json", short)],
            accuracy_paths=[],
            output=tmp_path / "aggregate.json",
        )


def test_figure_refuses_missing_evidence(tmp_path) -> None:
    payload = {
        "schema_version": 1,
        "protocol_id": "sha256:protocol",
        "required_warm_repetitions": 5,
        "benchmark_records": [],
        "performance_comparisons": [],
        "accuracy_records": [],
        "input_record_sha256": [],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["aggregate_sha256"] = hashlib.sha256(canonical).hexdigest()
    aggregate = _write(tmp_path / "aggregate.json", payload)
    with pytest.raises(ValueError, match="performance"):
        PLOT.plot_figure1(aggregate, tmp_path / "figure1")
