"""Persistent SMC++ clean-split worker for publication-quality benchmarks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from typing import Any

import numpy as np

from smckit import SmcData
from smckit.tl import smcpp


def _model(population: str, eta: list[float]) -> dict[str, Any]:
    return {
        "model": {
            "class": "SMCModel",
            "knots": [0.01, 0.5, 1.5],
            "N0": 10_000.0,
            "spline_class": "Piecewise",
            "y": np.log(eta).tolist(),
            "pid": population,
        },
        "theta": 2.5e-4,
        "rho": 2.0e-4,
    }


def _fixture() -> tuple[SmcData, tuple[dict[str, Any], dict[str, Any]]]:
    observations = []
    for index in range(30):
        observations.extend(
            [
                (4_999, ((0, 0, 3), (0, 0, 2))),
                (
                    1,
                    (
                        (index % 3, (index * 2) % 4, 3),
                        (0, index % 3, 2),
                    ),
                ),
            ]
        )
    data = SmcData(
        uns={
            "n_populations": 2,
            "populations": ["pop-a", "pop-b"],
            "pids": ["pop-a", "pop-b"],
            "joint_observations": observations,
            "n_undist_by_population": [3, 2],
            "n_distinguished_by_population": [2, 0],
            "smcpp_header": {
                "pids": ["pop-a", "pop-b"],
                "dist": [[["distinguished", 0], ["distinguished", 1]], []],
                "undist": [
                    [["pop-a", index] for index in range(3)],
                    [["pop-b", index] for index in range(2)],
                ],
            },
            "total_sites": sum(span for span, _ in observations),
        }
    )
    return data, (_model("pop-a", [1.0, 2.0, 0.8]), _model("pop-b", [1.5, 0.7, 1.2]))


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    model = json.dumps(result["model"], sort_keys=True, separators=(",", ":"))
    return {
        "analysis": result["analysis"],
        "implementation": result["implementation"],
        "split": float(result["split"]),
        "log_scale": float(result["log_scale"]),
        "log_likelihood": float(result["log_likelihood"]),
        "joint_emission_sum": float(result["joint_emission_sum"]),
        "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
    }


def _write(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run_worker(*, implementation: str, max_iterations: int, seed: int) -> int:
    data, models = _fixture()
    prepared: tuple[SmcData, tuple[dict[str, Any], dict[str, Any]]] | None = None
    _write(
        {
            "event": "ready",
            "method": "smcpp",
            "implementation": implementation,
            "dataset": "split-control-v1",
        }
    )
    for raw in sys.stdin:
        request = json.loads(raw)
        if request.get("event") == "close":
            _write({"event": "closed"})
            return 0
        if request.get("event") == "prepare" and isinstance(request.get("repetition"), int):
            prepared = copy.deepcopy(data), copy.deepcopy(models)
            _write({"event": "prepared", "repetition": request["repetition"]})
            continue
        if request.get("event") != "run" or not isinstance(request.get("repetition"), int):
            raise ValueError("Expected a run event with an integer repetition.")
        if prepared is None:
            raise ValueError("Each run event must be preceded by a prepare event.")
        prepared_data, prepared_models = prepared
        prepared = None
        fitted = smcpp(
            prepared_data,
            implementation=implementation,
            split_models=prepared_models,
            max_iterations=max_iterations,
            seed=seed,
        ).results["smcpp"]
        _write(
            {
                "event": "result",
                "repetition": request["repetition"],
                "result": _summary(fitted),
            }
        )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("native", "upstream"), required=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker(
        implementation=args.implementation,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
