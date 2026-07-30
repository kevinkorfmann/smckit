"""Stable SMC++ model serialization and compatibility payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from smckit._core import SmcData

SMCPP_MODEL_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _result_payload(value: SmcData | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, SmcData):
        try:
            return value.results["smcpp"]
        except KeyError as exc:
            raise ValueError("SmcData does not contain an SMC++ result.") from exc
    return value


def smcpp_model_payload(
    value: SmcData | dict[str, Any],
    *,
    population: str = "ALL",
) -> dict[str, Any]:
    """Build a versioned normalized model plus an upstream-readable model block."""
    result = _result_payload(value)
    time = np.asarray(result["time"], dtype=float)
    eta = np.asarray(result["eta"], dtype=float)
    ne = np.asarray(result["ne"], dtype=float)
    boundaries = np.asarray(result["time_boundaries"], dtype=float)
    if time.ndim != 1 or eta.shape != time.shape or ne.shape != time.shape:
        raise ValueError("SMC++ time, eta, and ne must be aligned one-dimensional arrays.")
    if time.size == 0 or np.any(time <= 0) or np.any(np.diff(time) <= 0):
        raise ValueError("SMC++ model times must be positive and strictly increasing.")
    if np.any(eta <= 0) or np.any(ne <= 0):
        raise ValueError("SMC++ eta and population sizes must be positive.")
    n0 = float(result["n0"])
    if n0 <= 0:
        raise ValueError("SMC++ n0 must be positive.")

    upstream_model = {
        "class": "SMCModel",
        "knots": time.tolist(),
        "N0": n0,
        "spline_class": "Piecewise",
        "y": np.log(eta).tolist(),
        "pid": population,
    }
    return {
        "schema_version": SMCPP_MODEL_SCHEMA_VERSION,
        "method": "smcpp",
        "model": upstream_model,
        "theta": float(result["theta"]),
        "rho": float(result["rho"]),
        "hidden_states": {population: [0.0, float("inf")]},
        "smckit": {
            "time": time.tolist(),
            "time_boundaries": boundaries.tolist(),
            "eta": eta.tolist(),
            "ne": ne.tolist(),
            "time_years": np.asarray(result["time_years"], dtype=float).tolist(),
            "n0": n0,
            "n_undist": int(result["n_undist"]),
            "n_distinguished": int(result["n_distinguished"]),
            "regularization": float(result["regularization"]),
        },
    }


def write_smcpp_model(
    value: SmcData | dict[str, Any],
    path: str | Path,
    *,
    population: str = "ALL",
) -> Path:
    """Write a stable SMC++ model JSON document."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = smcpp_model_payload(value, population=population)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return target


def read_smcpp_model(path: str | Path) -> dict[str, Any]:
    """Read and validate normalized or original SMC++ model JSON."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SMC++ model does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed SMC++ model JSON: {source}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("SMC++ model JSON must contain a `model` object.")
    model = payload["model"]
    if not isinstance(model, dict) or model.get("class") != "SMCModel":
        raise ValueError("Only one-population SMCModel JSON is currently supported.")
    knots = np.asarray(model.get("knots", []), dtype=float)
    values = np.asarray(model.get("y", []), dtype=float)
    expected_values = len(knots) + 2 if model.get("spline_class") == "BSpline" else len(knots)
    if knots.ndim != 1 or values.shape != (expected_values,) or knots.size == 0:
        raise ValueError("SMC++ model has an invalid knot or spline-parameter count.")
    if np.any(knots <= 0) or np.any(np.diff(knots) <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("SMC++ model knots must increase and model values must be finite.")
    n0 = float(model.get("N0", 0.0))
    if n0 <= 0:
        raise ValueError("SMC++ model N0 must be positive.")
    return payload


__all__ = [
    "SMCPP_MODEL_SCHEMA_VERSION",
    "read_smcpp_model",
    "smcpp_model_payload",
    "write_smcpp_model",
]
