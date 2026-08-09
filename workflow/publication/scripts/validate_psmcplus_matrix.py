#!/usr/bin/env python3
"""Run a deterministic, capability-diverse native/upstream PSMC+ matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msprime
import numpy as np

import smckit

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"
THREAD_ENV = {
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Dataset:
    """One deterministic pairwise simulation and its optional local-rate maps."""

    name: str
    input_path: Path
    mutation_map_path: Path
    recombination_map_path: Path
    ancestry_seed: int
    mutation_seed: int
    demography: str
    sequence_length: int
    heterozygotes: int
    masked_bases: int


@dataclass(frozen=True)
class MatrixCase:
    """One scientifically distinct fit or decoding contract."""

    name: str
    datasets: tuple[str, ...]
    options: dict[str, Any]
    use_mutation_maps: bool = False
    use_recombination_maps: bool = False
    write_marginal: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.view(np.uint8))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _demography(name: str) -> msprime.Demography:
    demography = msprime.Demography()
    if name == "constant":
        demography.add_population(name="pop", initial_size=10_000)
    elif name == "bottleneck":
        demography.add_population(name="pop", initial_size=20_000)
        demography.add_population_parameters_change(
            time=800,
            initial_size=2_000,
            population="pop",
        )
        demography.add_population_parameters_change(
            time=2_400,
            initial_size=15_000,
            population="pop",
        )
    elif name == "expansion":
        demography.add_population(name="pop", initial_size=40_000)
        demography.add_population_parameters_change(
            time=1_500,
            initial_size=5_000,
            population="pop",
        )
    else:  # pragma: no cover - guarded by the fixed scenario table
        raise ValueError(f"Unknown demography: {name}")
    return demography


def _write_multihetsep(
    tree_sequence: msprime.TreeSequence,
    path: Path,
    *,
    chromosome: str,
    missing_stride: int | None,
) -> tuple[int, int]:
    positions: list[int] = []
    for variant in tree_sequence.variants():
        if variant.genotypes[0] != variant.genotypes[1]:
            position = int(variant.site.position) + 1
            if not positions or position > positions[-1]:
                positions.append(position)
    if not positions:
        raise RuntimeError(f"Simulation {chromosome} produced no heterozygous sites.")

    previous = 0
    masked_bases = 0
    lines: list[str] = []
    for index, position in enumerate(positions):
        physical_gap = position - previous
        callable_gap = physical_gap
        if missing_stride is not None and index > 0 and index % missing_stride == 0:
            missing = min(max(physical_gap // 3, 1), physical_gap - 1)
            callable_gap -= missing
            masked_bases += missing
        lines.append(f"{chromosome}\t{position}\t{callable_gap}\t01\n")
        previous = position
    path.write_text("".join(lines), encoding="utf-8")
    return len(positions), masked_bases


def _write_rate_maps(
    *,
    chromosome: str,
    sequence_length: int,
    mutation_path: Path,
    recombination_path: Path,
) -> None:
    breaks = [
        0,
        sequence_length // 5,
        2 * sequence_length // 5,
        3 * sequence_length // 5,
        4 * sequence_length // 5,
    ]
    stops = [*breaks[1:], sequence_length]
    mutation = [0.55, 1.45, 0.75, 1.25, 1.0]
    recombination = [1.8, 0.4, 1.35, 0.45, 1.0]
    mutation_path.write_text(
        "".join(
            f"{chromosome}\t{start}\t{stop}\t{value}\n"
            for start, stop, value in zip(breaks, stops, mutation, strict=True)
        ),
        encoding="utf-8",
    )
    recombination_path.write_text(
        "".join(
            f"{chromosome}\t{start}\t{stop}\t{value}\n"
            for start, stop, value in zip(breaks, stops, recombination, strict=True)
        ),
        encoding="utf-8",
    )


def _simulate_dataset(
    directory: Path,
    *,
    name: str,
    ancestry_seed: int,
    mutation_seed: int,
    missing_stride: int | None,
    sequence_length: int = 400_000,
) -> Dataset:
    chromosome = f"chr_{name}"
    ancestry = msprime.sim_ancestry(
        samples=2,
        ploidy=1,
        sequence_length=sequence_length,
        recombination_rate=1.0e-8,
        demography=_demography(name),
        random_seed=ancestry_seed,
    )
    mutated = msprime.sim_mutations(
        ancestry,
        rate=1.25e-8,
        random_seed=mutation_seed,
        model=msprime.BinaryMutationModel(),
    )
    input_path = directory / f"{name}.multihetsep"
    mutation_map_path = directory / f"{name}.mutation.bed"
    recombination_map_path = directory / f"{name}.recombination.bed"
    heterozygotes, masked_bases = _write_multihetsep(
        mutated,
        input_path,
        chromosome=chromosome,
        missing_stride=missing_stride,
    )
    _write_rate_maps(
        chromosome=chromosome,
        sequence_length=sequence_length,
        mutation_path=mutation_map_path,
        recombination_path=recombination_map_path,
    )
    return Dataset(
        name=name,
        input_path=input_path,
        mutation_map_path=mutation_map_path,
        recombination_map_path=recombination_map_path,
        ancestry_seed=ancestry_seed,
        mutation_seed=mutation_seed,
        demography=name,
        sequence_length=sequence_length,
        heterozygotes=heterozygotes,
        masked_bases=masked_bases,
    )


def _cases() -> tuple[MatrixCase, ...]:
    return (
        MatrixCase(
            "constant_estimated_rho",
            ("constant",),
            {
                "number_time_windows": 4,
                "bin_size": 50,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "bottleneck_grouped_fixed_rho",
            ("bottleneck",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_segments": "2*0,2*2",
                "lambda_initial": [1.6, 0.55, 1.8],
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "two_chromosome_missing",
            ("constant", "expansion"),
            {
                "number_time_windows": 4,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_segments": "1*0,3*1",
                "lambda_initial": [1.25, 0.7, 1.4, 2.0],
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "expansion_midpoint_transitions",
            ("expansion",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "midpoint_transitions": True,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "expansion_midpoint_emissions",
            ("expansion",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "midpoint_emissions": True,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "expansion_nonexponential_recombination",
            ("expansion",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "nonexponential_recombination": True,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "expansion_final_time_factor",
            ("expansion",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "final_time_factor": 3.0,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "expansion_approximation_controls",
            ("expansion",),
            {
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "midpoint_transitions": True,
                "midpoint_emissions": True,
                "nonexponential_recombination": True,
                "final_time_factor": 3.0,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
        ),
        MatrixCase(
            "bottleneck_local_rates",
            ("bottleneck",),
            {
                "number_time_windows": 4,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "rho_fixed": True,
                "recombination_map_downsamples": 2,
                "iterations": 1,
                "likelihood_threshold": 0,
                "cores": 1,
            },
            use_mutation_maps=True,
            use_recombination_maps=True,
        ),
        MatrixCase(
            "constant_decode",
            ("constant",),
            {
                "mode": "decode",
                "number_time_windows": 4,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "lambda_initial": [0.65, 0.9, 1.6, 1.1],
                "decode_downsample": 17,
                "cores": 1,
            },
            write_marginal=True,
        ),
        MatrixCase(
            "expansion_final_time_decode",
            ("expansion",),
            {
                "mode": "decode",
                "number_time_windows": 6,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "lambda_initial": [0.6, 0.8, 1.0, 1.4, 2.0, 0.9],
                "final_time_factor": 3.0,
                "decode_downsample": 23,
                "cores": 1,
            },
            write_marginal=True,
        ),
        MatrixCase(
            "bottleneck_local_rate_decode",
            ("bottleneck",),
            {
                "mode": "decode",
                "number_time_windows": 4,
                "bin_size": 50,
                "scaled_recombination_rate": 4.0e-4,
                "lambda_initial": [1.5, 0.55, 1.8, 1.1],
                "recombination_map_downsamples": 2,
                "decode_downsample": 19,
                "cores": 1,
            },
            use_mutation_maps=True,
            use_recombination_maps=True,
            write_marginal=True,
        ),
    )


def _max_absolute(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _max_relative(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return float("inf")
    denominator = np.maximum(np.abs(left_array), 1e-15)
    return float(np.max(np.abs(left_array - right_array) / denominator, initial=0.0))


def _compare_fit(upstream: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "log_likelihood_absolute_error": abs(
            float(native["log_likelihood"]) - float(upstream["log_likelihood"])
        ),
        "theta_relative_error": _max_relative(upstream["theta"], native["theta"]),
        "rho_relative_error": _max_relative(upstream["rho"], native["rho"]),
        "lambda_relative_error_max": _max_relative(upstream["lambda"], native["lambda"]),
        "left_boundary_absolute_error_max": _max_absolute(
            upstream["scaled_left_time_boundary"],
            native["scaled_left_time_boundary"],
        ),
        "right_boundary_absolute_error_max": _max_absolute(
            upstream["scaled_right_time_boundary"],
            native["scaled_right_time_boundary"],
        ),
    }
    metrics["passed"] = bool(
        metrics["log_likelihood_absolute_error"] <= 1e-6
        and metrics["theta_relative_error"] <= 1e-12
        and metrics["rho_relative_error"] <= 1e-6
        and metrics["lambda_relative_error_max"] <= 1e-6
        and metrics["left_boundary_absolute_error_max"] <= 1e-12
        and metrics["right_boundary_absolute_error_max"] <= 1e-12
    )
    return metrics


def _compare_decode(upstream: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    upstream_marginal = upstream["marginal_recombination"]
    native_marginal = native["marginal_recombination"]
    metrics = {
        "log_likelihood_absolute_error": abs(
            float(native["log_likelihood"]) - float(upstream["log_likelihood"])
        ),
        "posterior_absolute_error_max": _max_absolute(upstream["posterior"], native["posterior"]),
        "posterior_mean_time_absolute_error_max": _max_absolute(
            upstream["posterior_mean_time"], native["posterior_mean_time"]
        ),
        "marginal_recombination_absolute_error_max": _max_absolute(
            upstream_marginal["recombination_probability"],
            native_marginal["recombination_probability"],
        ),
        "position_exact": bool(np.array_equal(upstream["position"], native["position"])),
        "marginal_position_exact": bool(
            np.array_equal(upstream_marginal["position"], native_marginal["position"])
        ),
    }
    metrics["passed"] = bool(
        metrics["log_likelihood_absolute_error"] <= 1e-6
        and metrics["posterior_absolute_error_max"] <= 1e-10
        and metrics["posterior_mean_time_absolute_error_max"] <= 1e-10
        and metrics["marginal_recombination_absolute_error_max"] <= 1e-10
        and metrics["position_exact"]
        and metrics["marginal_position_exact"]
    )
    return metrics


def _result_summary(result: dict[str, Any], mode: str) -> dict[str, Any]:
    summary = {
        "log_likelihood": float(result["log_likelihood"]),
        "theta": float(result["theta"]),
        "rho": float(result["rho"]),
    }
    if mode == "fit":
        summary.update(
            {
                "lambda": np.asarray(result["lambda"], dtype=float).tolist(),
                "scaled_left_time_boundary": np.asarray(
                    result["scaled_left_time_boundary"], dtype=float
                ).tolist(),
                "scaled_right_time_boundary": np.asarray(
                    result["scaled_right_time_boundary"], dtype=float
                ).tolist(),
            }
        )
        return summary

    marginal = result["marginal_recombination"]
    summary["arrays"] = {
        "position": _array_record(result["position"]),
        "posterior": _array_record(result["posterior"]),
        "posterior_mean_time": _array_record(result["posterior_mean_time"]),
        "marginal_position": _array_record(marginal["position"]),
        "marginal_recombination_probability": _array_record(marginal["recombination_probability"]),
    }
    return summary


def _run_case(
    case: MatrixCase,
    datasets: dict[str, Dataset],
    directory: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    selected = [datasets[name] for name in case.datasets]
    inputs = [item.input_path for item in selected]
    mutation_maps = (
        [item.mutation_map_path for item in selected] if case.use_mutation_maps else None
    )
    recombination_maps = (
        [item.recombination_map_path for item in selected] if case.use_recombination_maps else None
    )
    results: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, float] = {}
    for implementation in ("upstream", "native"):
        data = smckit.io.read_multihetsep(inputs)
        marginal_path = (
            directory / f"{case.name}-{implementation}-marginal.txt"
            if case.write_marginal
            else None
        )
        started = time.perf_counter()
        smckit.tl.psmcplus(
            data,
            options=smckit.tl.PSMCPlusOptions(**case.options),
            input_paths=inputs,
            mutation_map_paths=mutation_maps,
            recombination_map_paths=recombination_maps,
            marginal_recombination_path=marginal_path,
            implementation=implementation,
            timeout=timeout,
        )
        runtimes[implementation] = time.perf_counter() - started
        results[implementation] = data.results["psmcplus"]

    mode = case.options.get("mode", "fit")
    comparison = (
        _compare_decode(results["upstream"], results["native"])
        if mode == "decode"
        else _compare_fit(results["upstream"], results["native"])
    )
    return {
        "name": case.name,
        "mode": mode,
        "datasets": list(case.datasets),
        "options": case.options,
        "map_controls": {
            "mutation": case.use_mutation_maps,
            "recombination": case.use_recombination_maps,
        },
        "runtime_seconds": runtimes,
        "results": {
            implementation: _result_summary(result, mode)
            for implementation, result in results.items()
        },
        "comparison": comparison,
    }


def _git_state() -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Persist simulated inputs and engine artifacts in this directory.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only the named case; repeat to select multiple cases.",
    )
    args = parser.parse_args()
    for key, value in THREAD_ENV.items():
        os.environ[key] = value

    available_cases = {case.name: case for case in _cases()}
    selected_names = args.selected_cases or list(available_cases)
    unknown = sorted(set(selected_names) - set(available_cases))
    if unknown:
        parser.error("unknown case(s): " + ", ".join(unknown))

    if args.work_dir is None:
        working_context = tempfile.TemporaryDirectory(prefix="smckit-psmcplus-matrix-")
    else:
        persistent = args.work_dir.expanduser().resolve()
        persistent.mkdir(parents=True, exist_ok=True)
        working_context = nullcontext(str(persistent))

    with working_context as temporary:
        directory = Path(temporary)
        datasets = {
            "constant": _simulate_dataset(
                directory,
                name="constant",
                ancestry_seed=701,
                mutation_seed=1701,
                missing_stride=None,
            ),
            "bottleneck": _simulate_dataset(
                directory,
                name="bottleneck",
                ancestry_seed=709,
                mutation_seed=1709,
                missing_stride=7,
            ),
            "expansion": _simulate_dataset(
                directory,
                name="expansion",
                ancestry_seed=719,
                mutation_seed=1719,
                missing_stride=11,
            ),
        }
        records = [
            _run_case(available_cases[name], datasets, directory, timeout=args.timeout)
            for name in selected_names
        ]
        dataset_records = []
        for dataset in datasets.values():
            record = asdict(dataset)
            for key in ("input_path", "mutation_map_path", "recombination_map_path"):
                path = Path(record[key])
                record[key] = path.name
                record[f"{key}_sha256"] = _sha256(path)
            dataset_records.append(record)

    payload = {
        "schema_version": 2,
        "method": "psmcplus",
        "purpose": "native promotion capability matrix",
        "source": _git_state(),
        "upstream_commit": UPSTREAM_COMMIT,
        "software": {
            "smckit": smckit.__version__,
            "msprime": msprime.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "platform": platform.platform(),
        "thread_environment": THREAD_ENV,
        "datasets": dataset_records,
        "cases": records,
        "passed": all(record["comparison"]["passed"] for record in records),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
