"""Normalized adapter for the maintained external PHLASH package."""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from smckit._core import SmcData
from smckit.tl._implementation import annotate_result, normalize_implementation


def _load_phlash():
    try:
        return importlib.import_module("phlash")
    except ImportError as exc:
        raise RuntimeError(
            "PHLASH 1.0.6 is not installed. On Python 3.12+, install "
            "`smckit[phlash]`; Python 3.10-3.11 remain supported by smckit but "
            "not by the current PHLASH release."
        ) from exc


def _posterior_payload(models: Sequence[Any], *, grid_size: int) -> dict[str, Any]:
    if not models:
        raise RuntimeError("PHLASH returned no posterior samples.")
    serialized: list[dict[str, Any]] = []
    positive_knots: list[float] = []
    for model in models:
        knots = np.asarray(model.eta.t, dtype=float)
        positive_knots.extend(knots[np.isfinite(knots) & (knots > 0)].tolist())
        eval_knots = knots[np.isfinite(knots)]
        serialized.append(
            {
                "theta": float(model.theta),
                "rho": float(model.rho),
                "time": eval_knots,
                "ne": np.asarray(model.eta(eval_knots, Ne=True), dtype=float),
            }
        )
    if not positive_knots:
        raise RuntimeError("PHLASH posterior has no positive finite time knots.")
    grid = np.geomspace(min(positive_knots), max(positive_knots), grid_size)
    trajectories = np.asarray(
        [model.eta(grid, Ne=True) for model in models],
        dtype=float,
    )
    return {
        "time": grid,
        "ne": np.median(trajectories, axis=0),
        "posterior_ne": trajectories,
        "credible_interval": {
            "level": 0.95,
            "lower": np.quantile(trajectories, 0.025, axis=0),
            "upper": np.quantile(trajectories, 0.975, axis=0),
        },
        "posterior_samples": serialized,
        "n_posterior_samples": len(serialized),
    }


def phlash(
    inputs: Sequence[str | Path | Any],
    *,
    implementation: str = "auto",
    input_kind: str = "auto",
    samples: Sequence[Any] | None = None,
    region: str | None = None,
    grid_size: int = 200,
    **fit_options: Any,
) -> SmcData:
    """Run external PHLASH and return normalized posterior summaries.

    Pass PHLASH contig objects directly, or paths. ``.psmcfa`` paths use
    ``phlash.psmc``; other paths use ``phlash.contig`` and ``phlash.fit``.
    """
    requested = normalize_implementation(implementation)
    if requested == "native":
        raise NotImplementedError(
            "PHLASH is an exact external integration; no smckit-native rewrite is planned."
        )
    package = _load_phlash()
    values = list(inputs)
    if not values:
        raise ValueError("At least one PHLASH input is required.")
    paths = [str(Path(value).resolve()) for value in values if isinstance(value, (str, Path))]
    use_psmc = input_kind == "psmcfa" or (
        input_kind == "auto"
        and len(paths) == len(values)
        and all(path.endswith((".psmcfa", ".psmcfa.gz")) for path in paths)
    )
    started = time.perf_counter()
    if use_psmc:
        models = package.psmc(paths, **fit_options)
    else:
        contigs = [
            package.contig(value, samples=samples, region=region)
            if isinstance(value, (str, Path))
            else value
            for value in values
        ]
        models = package.fit(contigs, **fit_options)
    result = _posterior_payload(models, grid_size=grid_size)
    annotate_result(
        result,
        method_name="phlash",
        implementation_requested=requested,
        implementation_used="upstream",
        upstream_metadata={"tool": "phlash", "version": package.__version__},
        effective_args={
            "input_kind": input_kind,
            "samples": samples,
            "region": region,
            "grid_size": grid_size,
            **fit_options,
        },
        input_paths=paths,
        runtime_seconds=time.perf_counter() - started,
    )
    return SmcData(results={"phlash": result}, uns={"phlash_inputs": values})


__all__ = ["phlash"]
