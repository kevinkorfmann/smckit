"""Normalized, reproducible adapter for the maintained external PHLASH package."""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from smckit._core import SmcData
from smckit._provenance import sha256_file
from smckit.tl._implementation import annotate_result, normalize_implementation

_INPUT_KINDS = {"auto", "psmcfa", "vcf", "tree_sequence", "contig"}
_PSMCFA_SUFFIXES = (".psmcfa", ".psmcfa.gz")
_VCF_SUFFIXES = (".vcf", ".vcf.gz", ".bcf")
_TREE_SEQUENCE_SUFFIXES = (".trees", ".ts", ".tsz", ".tszip")


def _load_phlash():
    try:
        return importlib.import_module("phlash")
    except ImportError as exc:
        raise RuntimeError(
            "PHLASH 1.0.6 is not installed. On Python 3.12+, install "
            "`smckit[phlash]`; Python 3.10-3.11 remain supported by smckit but "
            "not by the current PHLASH release."
        ) from exc


def _path_kind(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(_PSMCFA_SUFFIXES):
        return "psmcfa"
    if lowered.endswith(_VCF_SUFFIXES):
        return "vcf"
    if lowered.endswith(_TREE_SEQUENCE_SUFFIXES):
        return "tree_sequence"
    raise ValueError(
        f"Cannot infer the PHLASH input type from {path!r}; set input_kind explicitly "
        "or use .psmcfa[.gz], .vcf[.gz], .bcf, .trees, .ts, .tsz, or .tszip."
    )


def _resolve_input_kind(values: Sequence[Any], requested: str) -> str:
    if requested not in _INPUT_KINDS:
        choices = ", ".join(sorted(_INPUT_KINDS))
        raise ValueError(f"input_kind must be one of {choices}; got {requested!r}.")
    if requested != "auto":
        return requested
    path_values = [value for value in values if isinstance(value, (str, Path))]
    kinds = {_path_kind(str(value)) for value in path_values}
    if len(kinds) > 1:
        raise ValueError(
            "Automatic PHLASH input detection found mixed path formats; use one "
            "format per call or construct PHLASH Contig objects first."
        )
    if len(kinds) == 1 and len(path_values) == len(values):
        return kinds.pop()
    if kinds:
        raise ValueError(
            "Do not mix path inputs and constructed PHLASH Contig objects in one call."
        )
    return "contig"


def _resolve_paths(values: Sequence[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PHLASH input does not exist: {path}")
        paths.append(str(path))
    return paths


def _seed_key(random_seed: int | None, fit_options: dict[str, Any]) -> None:
    if random_seed is None:
        return
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")
    if "key" in fit_options:
        raise ValueError("Pass either random_seed or the PHLASH `key` option, not both.")
    jax = importlib.import_module("jax")
    fit_options["key"] = jax.random.PRNGKey(random_seed)


def _fit_psmcfa(
    package: Any,
    paths: Sequence[str],
    *,
    window_size: int,
    hold_out: bool,
    fit_options: dict[str, Any],
) -> tuple[Sequence[Any], bool]:
    """Run PHLASH PSMCFA inference, repairing the 1.0.6 matrix-shape regression."""
    raw_contig = getattr(getattr(package, "data", None), "RawContig", None)
    if raw_contig is None:
        return (
            package.psmc(
                list(paths),
                window_size=window_size,
                hold_out=hold_out,
                **fit_options,
            ),
            False,
        )

    contigs = [
        contig
        for path in paths
        for contig in raw_contig.from_psmcfa_iter(path, window_size=window_size)
    ]
    if not contigs:
        raise ValueError("PHLASH found no PSMCFA records.")
    compatibility_shim = False
    repaired = []
    for contig in contigs:
        matrix = np.asarray(contig.het_matrix, dtype=np.int8)
        if matrix.ndim == 2:
            observed = np.full(matrix.shape, window_size, dtype=np.int32)
            observed[matrix < 0] = 0
            heterozygotes = (matrix > 0).astype(np.int32)
            matrix = np.stack([observed, heterozygotes], axis=-1)
            contig = raw_contig(
                het_matrix=matrix,
                afs=contig.afs,
                window_size=contig.window_size,
            )
            compatibility_shim = True
        repaired.append(contig)
    test_data = repaired.pop(0) if hold_out and len(repaired) > 1 else None
    return (
        package.fit(repaired, test_data=test_data, **fit_options),
        compatibility_shim,
    )


def _posterior_payload(
    models: Sequence[Any],
    *,
    grid_size: int,
    credible_level: float,
) -> dict[str, Any]:
    if not models:
        raise RuntimeError("PHLASH returned no posterior samples.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must be strictly between 0 and 1.")

    serialized: list[dict[str, Any]] = []
    positive_knots: list[float] = []
    theta: list[float] = []
    rho: list[float] = []
    for sample_index, model in enumerate(models):
        knots = np.asarray(model.eta.t, dtype=float)
        finite_knots = knots[np.isfinite(knots)]
        if finite_knots.size == 0:
            raise RuntimeError(f"PHLASH posterior sample {sample_index} has no finite knots.")
        positive_knots.extend(finite_knots[finite_knots > 0].tolist())
        sample_theta = float(model.theta)
        sample_rho = float(model.rho)
        sample_ne = np.asarray(model.eta(finite_knots, Ne=True), dtype=float)
        if sample_ne.shape != finite_knots.shape or not np.all(np.isfinite(sample_ne)):
            raise RuntimeError(
                f"PHLASH posterior sample {sample_index} returned an invalid trajectory."
            )
        theta.append(sample_theta)
        rho.append(sample_rho)
        serialized.append(
            {
                "sample": sample_index,
                "theta": sample_theta,
                "rho": sample_rho,
                "time": finite_knots,
                "ne": sample_ne,
            }
        )
    if not positive_knots:
        raise RuntimeError("PHLASH posterior has no positive finite time knots.")

    grid = np.geomspace(min(positive_knots), max(positive_knots), grid_size)
    trajectories = np.asarray([model.eta(grid, Ne=True) for model in models], dtype=float)
    if trajectories.shape != (len(models), grid_size) or not np.all(np.isfinite(trajectories)):
        raise RuntimeError("PHLASH returned invalid posterior trajectories.")
    alpha = (1.0 - credible_level) / 2.0
    theta_values = np.asarray(theta, dtype=float)
    rho_values = np.asarray(rho, dtype=float)
    return {
        "time": grid,
        "ne": np.median(trajectories, axis=0),
        "posterior_ne": trajectories,
        "credible_interval": {
            "level": credible_level,
            "lower": np.quantile(trajectories, alpha, axis=0),
            "upper": np.quantile(trajectories, 1.0 - alpha, axis=0),
        },
        "theta": float(np.median(theta_values)),
        "rho": float(np.median(rho_values)),
        "posterior_theta": theta_values,
        "posterior_rho": rho_values,
        "posterior_samples": serialized,
        "n_posterior_samples": len(serialized),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _output_paths(output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return Path(f"{prefix}.phlash.json"), Path(f"{prefix}.phlash.posterior.npz")


def _write_posterior_archive(result: dict[str, Any], path: Path) -> dict[str, Any]:
    interval = result["credible_interval"]
    np.savez_compressed(
        path,
        time=result["time"],
        ne=result["ne"],
        posterior_ne=result["posterior_ne"],
        credible_lower=interval["lower"],
        credible_upper=interval["upper"],
        credible_level=np.asarray(interval["level"]),
        posterior_theta=result["posterior_theta"],
        posterior_rho=result["posterior_rho"],
    )
    return {
        "kind": "posterior_archive",
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _write_result_json(result: dict[str, Any], path: Path) -> dict[str, Any]:
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "kind": "normalized_result",
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _as_contig(
    package: Any,
    value: Any,
    *,
    samples: Sequence[Any] | None,
    region: str | None,
) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    if samples is None:
        raise ValueError("PHLASH VCF and tree-sequence inputs require `samples`.")
    return package.contig(
        str(Path(value).expanduser().resolve()),
        samples=list(samples),
        region=region,
    )


def phlash(
    inputs: Sequence[str | Path | Any],
    *,
    implementation: str = "auto",
    input_kind: str = "auto",
    samples: Sequence[Any] | None = None,
    region: str | None = None,
    test_input: str | Path | Any | None = None,
    window_size: int = 100,
    hold_out: bool = True,
    grid_size: int = 200,
    credible_level: float = 0.95,
    random_seed: int | None = 1,
    output_prefix: str | Path | None = None,
    **fit_options: Any,
) -> SmcData:
    """Run external PHLASH and return normalized posterior summaries.

    PSMCFA paths are passed to :func:`phlash.psmc`. VCF/BCF and tree-sequence
    paths are converted with :func:`phlash.contig`; constructed PHLASH Contig
    objects may also be supplied directly. The complete original PHLASH Python
    interface remains available by importing :mod:`phlash`.
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
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must be strictly between 0 and 1.")

    resolved_kind = _resolve_input_kind(values, input_kind)
    paths = _resolve_paths(values)
    effective_options = dict(fit_options)
    _seed_key(random_seed, effective_options)
    started = time.perf_counter()
    compatibility_warnings: list[str] = []
    compatibility_shim = False

    if resolved_kind == "psmcfa":
        if len(paths) != len(values):
            raise TypeError("PSMCFA execution accepts paths only.")
        if samples is not None or region is not None or test_input is not None:
            raise ValueError(
                "samples, region, and test_input are not valid for PSMCFA execution; "
                "use hold_out to reserve a PSMCFA contig."
            )
        models, compatibility_shim = _fit_psmcfa(
            package,
            paths,
            window_size=window_size,
            hold_out=hold_out,
            fit_options=effective_options,
        )
        if compatibility_shim:
            compatibility_warnings.append(
                "PHLASH 1.0.6 PSMCFA matrix-shape compatibility shim applied."
            )
    else:
        if resolved_kind == "vcf" and region is None:
            raise ValueError("PHLASH VCF/BCF inputs require a bcftools-style `region`.")
        if resolved_kind == "tree_sequence" and region is not None:
            raise ValueError("PHLASH tree-sequence inputs do not accept `region`.")
        contigs = [_as_contig(package, value, samples=samples, region=region) for value in values]
        test_contig = (
            _as_contig(package, test_input, samples=samples, region=region)
            if test_input is not None
            else None
        )
        if test_contig is None and hold_out and len(contigs) > 1:
            test_contig = contigs.pop(0)
        models = package.fit(
            contigs,
            test_data=test_contig,
            window_size=window_size,
            **effective_options,
        )

    result = _posterior_payload(
        models,
        grid_size=grid_size,
        credible_level=credible_level,
    )
    output_paths = _output_paths(output_prefix) if output_prefix is not None else None
    artifacts = (
        [_write_posterior_archive(result, output_paths[1])] if output_paths is not None else []
    )
    recorded_options = {key: value for key, value in fit_options.items() if key != "key"}
    annotate_result(
        result,
        method_name="phlash",
        implementation_requested=requested,
        implementation_used="upstream",
        upstream_metadata={
            "tool": "phlash",
            "version": package.__version__,
            "interface": "python",
            "psmcfa_compatibility_shim": compatibility_shim,
        },
        effective_args={
            "input_kind": resolved_kind,
            "samples": list(samples) if samples is not None else None,
            "region": region,
            "window_size": window_size,
            "hold_out": hold_out,
            "grid_size": grid_size,
            "credible_level": credible_level,
            **recorded_options,
        },
        input_paths=[
            *paths,
            *(
                [str(Path(test_input).expanduser().resolve())]
                if isinstance(test_input, (str, Path))
                else []
            ),
        ],
        seed=random_seed,
        runtime_seconds=time.perf_counter() - started,
        warning_messages=compatibility_warnings,
        artifacts=artifacts,
    )
    if output_paths is not None:
        result["provenance"]["artifacts"].append(_write_result_json(result, output_paths[0]))
    return SmcData(
        results={"phlash": result},
        uns={
            "phlash_inputs": values,
            "phlash_test_input": test_input,
            "phlash_input_kind": resolved_kind,
        },
    )


__all__ = ["phlash"]
