#!/usr/bin/env python3
"""Run one frozen PHLASH simulation replicate and record accuracy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import tskit

import smckit
from smckit._provenance import sha256_file
from smckit.validation import log_integrated_trajectory_error, posterior_coverage

SUPPORTED_SCENARIOS = {
    "constant",
    "bottleneck",
    "expansion",
    "selfing_dormancy",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _event_history(truth: dict[str, Any]) -> list[tuple[float, float]]:
    kind = truth.get("kind")
    if kind in {"constant", "bottleneck", "expansion"}:
        raw_events = truth.get("population_size_epochs")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError(f"{kind} truth has no population_size_epochs.")
        events = [
            (float(event["time_generations"]), float(event["population_size"]))
            for event in raw_events
        ]
    elif kind == "esmc2_coalescent_equivalent":
        events = [(0.0, float(truth["effective_population_size"]))]
    else:
        raise ValueError(f"Scenario {kind!r} is not a one-population PHLASH target.")
    events.sort()
    if events[0][0] != 0 or any(time < 0 or size <= 0 for time, size in events):
        raise ValueError("Population-size truth must start at time zero with positive sizes.")
    if any(right[0] <= left[0] for left, right in zip(events, events[1:], strict=False)):
        raise ValueError("Population-size event times must be strictly increasing.")
    return events


def evaluate_truth(truth: dict[str, Any], times: Any) -> np.ndarray:
    """Evaluate the frozen piecewise-constant effective-size truth."""
    query = np.asarray(times, dtype=float)
    if query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query)):
        raise ValueError("Evaluation times must be a non-empty finite vector.")
    if np.any(query <= 0):
        raise ValueError("Evaluation times must be positive.")
    events = _event_history(truth)
    event_times = np.asarray([event[0] for event in events], dtype=float)
    sizes = np.asarray([event[1] for event in events], dtype=float)
    indices = np.searchsorted(event_times, query, side="right") - 1
    return sizes[indices]


def truth_trajectory(
    truth: dict[str, Any],
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return step-preserving knots over a finite positive evaluation interval."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or lower >= upper:
        raise ValueError("Truth trajectory bounds must be finite, positive, and increasing.")
    events = _event_history(truth)
    times = [float(lower)]
    sizes = [float(evaluate_truth(truth, [lower])[0])]
    current = sizes[0]
    for event_time, new_size in events[1:]:
        if not lower < event_time < upper:
            continue
        left = float(np.nextafter(event_time, -np.inf))
        if left > times[-1]:
            times.append(left)
            sizes.append(current)
        times.append(event_time)
        sizes.append(new_size)
        current = new_size
    if upper > times[-1]:
        times.append(float(upper))
        sizes.append(current)
    return np.asarray(times), np.asarray(sizes)


def trajectory_metrics(
    truth: dict[str, Any],
    result: dict[str, Any],
    *,
    evaluation_min: float,
    evaluation_max: float,
) -> dict[str, float]:
    """Calculate log-time-weighted accuracy and interval-coverage metrics."""
    times = np.asarray(result["time"], dtype=float)
    estimate = np.asarray(result["ne"], dtype=float)
    interval = result["credible_interval"]
    lower_interval = np.asarray(interval["lower"], dtype=float)
    upper_interval = np.asarray(interval["upper"], dtype=float)
    if not (
        times.ndim == estimate.ndim == lower_interval.ndim == upper_interval.ndim == 1
        and times.size == estimate.size == lower_interval.size == upper_interval.size
    ):
        raise ValueError("PHLASH trajectory and credible-interval arrays must align.")
    if times.size < 2 or not all(
        np.all(np.isfinite(values)) for values in (times, estimate, lower_interval, upper_interval)
    ):
        raise ValueError(
            "PHLASH trajectory values must be finite and contain at least two points."
        )
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("PHLASH trajectory times must be positive and strictly increasing.")
    if any(np.any(values <= 0) for values in (estimate, lower_interval, upper_interval)):
        raise ValueError("PHLASH population-size estimates and intervals must be positive.")
    if np.any(lower_interval > upper_interval):
        raise ValueError("PHLASH credible-interval lower bounds must not exceed upper bounds.")
    domain_lower = max(float(times[0]), float(evaluation_min))
    domain_upper = min(float(times[-1]), float(evaluation_max))
    selected = (times >= domain_lower) & (times <= domain_upper)
    if np.count_nonzero(selected) < 2:
        raise ValueError("PHLASH result has fewer than two points in the evaluation domain.")
    selected_times = times[selected]
    selected_estimate = estimate[selected]
    selected_lower = lower_interval[selected]
    selected_upper = upper_interval[selected]
    truth_at_estimate = evaluate_truth(truth, selected_times)
    truth_time, truth_size = truth_trajectory(
        truth,
        float(selected_times[0]),
        float(selected_times[-1]),
    )
    log_time = np.log(selected_times)
    log_error = np.log(selected_estimate) - np.log(truth_at_estimate)
    span = log_time[-1] - log_time[0]
    covered = (selected_lower <= truth_at_estimate) & (truth_at_estimate <= selected_upper)
    return {
        "evaluation_min_generations": float(selected_times[0]),
        "evaluation_max_generations": float(selected_times[-1]),
        "log_integrated_trajectory_error": log_integrated_trajectory_error(
            truth_time,
            truth_size,
            selected_times,
            selected_estimate,
        ),
        "log_root_mean_squared_error": float(np.sqrt(np.trapezoid(log_error**2, log_time) / span)),
        "log_median_bias": float(np.median(log_error)),
        "posterior_coverage": posterior_coverage(
            truth_at_estimate,
            selected_lower,
            selected_upper,
        ),
        "log_time_weighted_posterior_coverage": float(
            np.trapezoid(covered.astype(float), log_time) / span
        ),
        "mean_log_credible_interval_width": float(
            np.trapezoid(np.log(selected_upper / selected_lower), log_time) / span
        ),
    }


def _sample_nodes(tree_sequence: tskit.TreeSequence) -> list[tuple[int, int]]:
    nodes = [
        tuple(int(node) for node in individual.nodes) for individual in tree_sequence.individuals()
    ]
    if not nodes or any(len(pair) != 2 for pair in nodes):
        raise ValueError("PHLASH publication inputs require diploid tree-sequence individuals.")
    return nodes


def run_phlash_accuracy(
    *,
    protocol: dict[str, Any],
    truth_payload: dict[str, Any],
    truth_path: Path,
    tree_path: Path,
    holdout_tree_path: Path,
    artifact_prefix: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Execute and persist one PHLASH accuracy record."""
    config = protocol["config"]
    phlash_config = config["phlash"]
    scenario = str(truth_payload["scenario"])
    replicate = int(truth_payload["replicate"])
    if scenario not in SUPPORTED_SCENARIOS or scenario not in phlash_config["scenarios"]:
        raise ValueError(f"Scenario {scenario!r} is not enabled for PHLASH validation.")
    if float(truth_payload["mutation_rate"]) != float(config["mutation_rate"]):
        raise ValueError("Simulation and frozen protocol mutation rates do not match.")
    for path in (truth_path, tree_path, holdout_tree_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_tree_hash = truth_payload["tree_sequence"]["sha256"]
    expected_holdout_hash = truth_payload["holdout_tree_sequence"]["sha256"]
    if sha256_file(tree_path) != expected_tree_hash:
        raise ValueError("Training tree-sequence checksum does not match frozen truth.")
    if sha256_file(holdout_tree_path) != expected_holdout_hash:
        raise ValueError("Holdout tree-sequence checksum does not match frozen truth.")

    training = tskit.load(tree_path)
    holdout = tskit.load(holdout_tree_path)
    nodes = _sample_nodes(training)
    if _sample_nodes(holdout) != nodes:
        raise ValueError("Training and holdout tree sequences have different diploid samples.")
    inference_seed = int(phlash_config["inference_seed"]) + replicate
    data = smckit.tl.phlash(
        [holdout_tree_path, tree_path],
        implementation="upstream",
        input_kind="tree_sequence",
        samples=nodes,
        window_size=int(phlash_config["window_size"]),
        hold_out=bool(phlash_config["hold_out"]),
        grid_size=int(phlash_config["grid_size"]),
        credible_level=float(phlash_config["credible_level"]),
        random_seed=inference_seed,
        output_prefix=artifact_prefix,
        mutation_rate=float(config["mutation_rate"]),
        niter=int(phlash_config["niter"]),
        num_particles=int(phlash_config["num_particles"]),
        num_workers=int(phlash_config["num_workers"]),
        max_samples=int(phlash_config["max_samples"]),
        overlap=int(phlash_config["overlap"]),
        progress=False,
    )
    result = data.results["phlash"]
    metrics = trajectory_metrics(
        truth_payload["truth"],
        result,
        evaluation_min=float(phlash_config["evaluation_min_generations"]),
        evaluation_max=float(phlash_config["evaluation_max_generations"]),
    )
    credible = result["credible_interval"]
    record = {
        "schema_version": 1,
        "method": "phlash",
        "protocol_id": protocol["protocol_id"],
        "protocol_source_sha256": protocol["source"]["sha256"],
        "scenario": scenario,
        "replicate": replicate,
        "protocol_expectations": {
            "replicates_per_scenario": int(config["replicates"]),
            "scenarios": list(phlash_config["scenarios"]),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "inputs": {
            "truth_sha256": sha256_file(truth_path),
            "tree_sequence_sha256": expected_tree_hash,
            "holdout_tree_sequence_sha256": expected_holdout_hash,
            "training_sequence_length": float(training.sequence_length),
            "holdout_sequence_length": float(holdout.sequence_length),
            "diploid_samples": len(nodes),
        },
        "inference": {
            "implementation_requested": "upstream",
            "implementation_used": result["implementation"],
            "phlash_version": result["upstream"]["version"],
            "random_seed": inference_seed,
            "arguments": result["provenance"]["arguments"],
            "runtime_seconds": result["provenance"]["runtime_seconds"],
            "warnings": result["provenance"]["warnings"],
            "artifacts": result["provenance"]["artifacts"],
        },
        "truth": {
            "kind": truth_payload["truth"]["kind"],
            "mutation_rate": float(config["mutation_rate"]),
            "recombination_rate": float(truth_payload["recombination_rate"]),
        },
        "posterior": {
            "credible_level": float(credible["level"]),
            "n_samples": int(result["n_posterior_samples"]),
            "time": np.asarray(result["time"]),
            "median_ne": np.asarray(result["ne"]),
            "credible_lower": np.asarray(credible["lower"]),
            "credible_upper": np.asarray(credible["upper"]),
            "posterior_ne_sha256": _array_sha256(result["posterior_ne"]),
        },
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--holdout-tree", type=Path, required=True)
    parser.add_argument("--artifact-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    truth = _load_json(args.truth)
    run_phlash_accuracy(
        protocol=_load_json(args.protocol),
        truth_payload=truth,
        truth_path=args.truth,
        tree_path=args.tree,
        holdout_tree_path=args.holdout_tree,
        artifact_prefix=args.artifact_prefix,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
