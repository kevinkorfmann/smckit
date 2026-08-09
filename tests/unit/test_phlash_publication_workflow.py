"""Tests for frozen PHLASH accuracy, coverage, and aggregation evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from smckit import SmcData
from smckit._provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "workflow" / "publication" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"smckit_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script("run_phlash_accuracy")
AGGREGATE = _load_script("aggregate_phlash_validation")
SIMULATE = _load_script("simulate")


def test_step_truth_and_metrics_preserve_demographic_changes() -> None:
    truth = {
        "kind": "bottleneck",
        "population_size_epochs": [
            {"time_generations": 0, "population_size": 10_000},
            {"time_generations": 500, "population_size": 1_000},
            {"time_generations": 1_000, "population_size": 10_000},
        ],
    }
    times = np.asarray([100, 499, 500, 999, 1_000, 10_000], dtype=float)
    expected = np.asarray([10_000, 10_000, 1_000, 1_000, 10_000, 10_000])
    assert np.array_equal(RUNNER.evaluate_truth(truth, times), expected)

    result = {
        "time": times,
        "ne": expected,
        "credible_interval": {
            "lower": expected * 0.9,
            "upper": expected * 1.1,
        },
    }
    metrics = RUNNER.trajectory_metrics(
        truth,
        result,
        evaluation_min=100,
        evaluation_max=10_000,
    )
    assert metrics["log_integrated_trajectory_error"] == pytest.approx(0)
    assert metrics["log_root_mean_squared_error"] == pytest.approx(0)
    assert metrics["posterior_coverage"] == pytest.approx(1)
    assert metrics["log_time_weighted_posterior_coverage"] == pytest.approx(1)


def test_runner_uses_independent_holdout_and_writes_evidence(monkeypatch, tmp_path) -> None:
    tree_path = tmp_path / "constant.trees"
    holdout_path = tmp_path / "constant.holdout.trees"
    truth_path = tmp_path / "constant.truth.json"
    truth = SIMULATE.simulate_scenario(
        scenario_name="constant",
        replicate=3,
        seed=400,
        sequence_length=20_000,
        recombination_rate=1e-8,
        mutation_rate=1.25e-8,
        tree_output=tree_path,
        holdout_tree_output=holdout_path,
        truth_output=truth_path,
    )
    config = yaml.safe_load((ROOT / "workflow/publication/config.yaml").read_text())
    protocol = {
        "protocol_id": "sha256:test",
        "source": {"sha256": "config-test"},
        "config": config,
    }
    calls = []

    def fake_phlash(inputs, **options):
        calls.append((list(inputs), options))
        prefix = Path(options["output_prefix"])
        result_path = Path(f"{prefix}.phlash.json")
        posterior_path = Path(f"{prefix}.phlash.posterior.npz")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n", encoding="utf-8")
        np.savez_compressed(posterior_path, posterior_ne=np.full((4, 4), 10_000.0))
        times = np.asarray([100, 1_000, 10_000, 1_000_000], dtype=float)
        sizes = np.full(4, 10_000.0)
        return SmcData(
            results={
                "phlash": {
                    "time": times,
                    "ne": sizes,
                    "posterior_ne": np.full((4, 4), 10_000.0),
                    "credible_interval": {
                        "level": 0.95,
                        "lower": sizes * 0.9,
                        "upper": sizes * 1.1,
                    },
                    "n_posterior_samples": 4,
                    "implementation": "upstream",
                    "upstream": {"version": "1.0.6"},
                    "provenance": {
                        "arguments": {"input_kind": "tree_sequence"},
                        "runtime_seconds": 2.5,
                        "warnings": [],
                        "artifacts": [
                            {"path": str(result_path), "sha256": sha256_file(result_path)},
                            {
                                "path": str(posterior_path),
                                "sha256": sha256_file(posterior_path),
                            },
                        ],
                    },
                }
            }
        )

    monkeypatch.setattr(RUNNER.smckit.tl, "phlash", fake_phlash)
    output = tmp_path / "constant.validation.json"
    prefix = tmp_path / "constant"
    record = RUNNER.run_phlash_accuracy(
        protocol=protocol,
        truth_payload=truth,
        truth_path=truth_path,
        tree_path=tree_path,
        holdout_tree_path=holdout_path,
        artifact_prefix=prefix,
        output_path=output,
    )

    assert calls[0][0] == [holdout_path, tree_path]
    assert calls[0][1]["hold_out"] is True
    assert calls[0][1]["mutation_rate"] == pytest.approx(1.25e-8)
    assert calls[0][1]["niter"] == 1000
    assert calls[0][1]["num_particles"] == 500
    assert record["metrics"]["posterior_coverage"] == pytest.approx(1)
    assert record["metrics"]["log_integrated_trajectory_error"] == pytest.approx(0)
    assert output.is_file()
    assert Path(f"{prefix}.phlash.json").is_file()
    assert Path(f"{prefix}.phlash.posterior.npz").is_file()


def _record(scenario: str, replicate: int, error: float, coverage: float) -> dict:
    return {
        "method": "phlash",
        "protocol_id": "sha256:test",
        "protocol_source_sha256": "config-test",
        "scenario": scenario,
        "replicate": replicate,
        "protocol_expectations": {
            "replicates_per_scenario": 2,
            "scenarios": ["constant", "bottleneck"],
        },
        "inference": {"phlash_version": "1.0.6", "runtime_seconds": 2.0 + replicate},
        "posterior": {"credible_level": 0.95},
        "metrics": {
            "log_integrated_trajectory_error": error,
            "log_root_mean_squared_error": error * 2,
            "log_median_bias": -error,
            "posterior_coverage": coverage,
            "log_time_weighted_posterior_coverage": coverage,
            "mean_log_credible_interval_width": 0.5,
        },
    }


def test_aggregate_groups_scenarios_and_rejects_duplicates() -> None:
    records = [
        _record("constant", 1, 0.1, 1.0),
        _record("constant", 2, 0.2, 0.9),
        _record("bottleneck", 1, 0.3, 0.8),
        _record("bottleneck", 2, 0.4, 0.9),
    ]
    result = AGGREGATE.aggregate_records(records)
    assert result["records"] == 4
    assert result["scenarios"]["constant"]["replicates"] == 2
    assert result["overall"]["metrics"]["posterior_coverage"]["mean"] == pytest.approx(0.9)

    with pytest.raises(ValueError, match="Duplicate"):
        AGGREGATE.aggregate_records([records[0], deepcopy(records[0])])
    with pytest.raises(ValueError, match="complete frozen replicate matrix"):
        AGGREGATE.aggregate_records(records[:-1])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("time", np.asarray([100.0, 100.0]), "strictly increasing"),
        ("ne", np.asarray([10_000.0, np.nan]), "finite"),
        ("ne", np.asarray([10_000.0, 0.0]), "positive"),
    ],
)
def test_trajectory_metrics_reject_invalid_posterior_values(
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    result = {
        "time": np.asarray([100.0, 1_000.0]),
        "ne": np.asarray([10_000.0, 10_000.0]),
        "credible_interval": {
            "lower": np.asarray([9_000.0, 9_000.0]),
            "upper": np.asarray([11_000.0, 11_000.0]),
        },
    }
    result[field] = replacement
    truth = {
        "kind": "constant",
        "population_size_epochs": [{"time_generations": 0, "population_size": 10_000}],
    }
    with pytest.raises(ValueError, match=message):
        RUNNER.trajectory_metrics(
            truth,
            result,
            evaluation_min=100,
            evaluation_max=1_000,
        )


def test_aggregate_rejects_mixed_protocol_sources() -> None:
    records = [
        _record("constant", 1, 0.1, 1.0),
        _record("constant", 2, 0.2, 0.9),
        _record("bottleneck", 1, 0.3, 0.8),
        _record("bottleneck", 2, 0.4, 0.9),
    ]
    records[-1]["protocol_source_sha256"] = "different-config"
    with pytest.raises(ValueError, match="protocol source hash"):
        AGGREGATE.aggregate_records(records)


def test_aggregate_cli_hashes_each_evidence_record(tmp_path: Path) -> None:
    records = [
        _record("constant", 1, 0.1, 1.0),
        _record("constant", 2, 0.2, 0.9),
        _record("bottleneck", 1, 0.3, 0.8),
        _record("bottleneck", 2, 0.4, 0.9),
    ]
    paths = []
    for record in records:
        path = tmp_path / f"{record['scenario']}-{record['replicate']}.json"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        paths.append(path)
    output = tmp_path / "summary.json"

    assert AGGREGATE.main([*(str(path) for path in paths), "--output", str(output)]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert len(summary["input_records"]) == 4
    for evidence in summary["input_records"]:
        path = Path(evidence["path"])
        assert evidence["sha256"] == sha256_file(path)
