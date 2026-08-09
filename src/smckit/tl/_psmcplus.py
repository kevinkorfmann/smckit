"""Typed, normalized execution of the preserved PSMC+ implementation."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import smckit.upstream as upstream
from smckit._core import SmcData
from smckit.tl._implementation import (
    annotate_result,
    method_upstream_available,
    normalize_implementation,
    require_upstream_available,
    standard_upstream_metadata,
)

PSMCPlusMode = Literal["fit", "decode"]


@dataclass(frozen=True)
class PSMCPlusOptions:
    """Scientifically meaningful controls of the original PSMC+ inference CLI."""

    mode: PSMCPlusMode = "fit"
    number_time_windows: int = 50
    spread_1: float = 0.1
    spread_2: float = 50.0
    bin_size: int = 100
    scaled_recombination_rate: float | None = None
    scaled_mutation_rate: float | Literal["empirical"] | None = None
    rho_fixed: bool = False
    mutation_recombination_ratio: float = 1.5
    lambda_lower_bound: float = 0.1
    lambda_upper_bound: float = 50.0
    recombination_map_downsamples: int = 200
    iterations: int = 20
    likelihood_threshold: float = 1.0
    lambda_initial: Sequence[float] | str | None = None
    lambda_segments: str | None = None
    parameter_tolerance: float = 1e-4
    objective_tolerance: float = 1e-4
    nonexponential_recombination: bool = False
    midpoint_transitions: bool = False
    midpoint_emissions: bool = False
    final_time_factor: float | None = None
    optimization_method: str = "Powell"
    save_iteration_files: bool = False
    decode_downsample: int = 10
    cores: int | None = None

    def validate(self) -> None:
        """Reject invalid typed options before starting the original process."""
        if self.mode not in {"fit", "decode"}:
            raise ValueError("PSMC+ mode must be 'fit' or 'decode'.")
        if self.number_time_windows < 2:
            raise ValueError("number_time_windows must be at least 2.")
        if self.spread_1 <= 0 or self.spread_2 <= 0:
            raise ValueError("spread_1 and spread_2 must be positive.")
        if self.bin_size < 1:
            raise ValueError("bin_size must be positive.")
        if self.scaled_recombination_rate is not None and self.scaled_recombination_rate <= 0:
            raise ValueError("scaled_recombination_rate must be positive.")
        if isinstance(self.scaled_mutation_rate, str):
            if self.scaled_mutation_rate != "empirical":
                raise ValueError("scaled_mutation_rate string value must be 'empirical'.")
        elif self.scaled_mutation_rate is not None and self.scaled_mutation_rate <= 0:
            raise ValueError("scaled_mutation_rate must be positive.")
        if self.mutation_recombination_ratio <= 0:
            raise ValueError("mutation_recombination_ratio must be positive.")
        if self.lambda_lower_bound <= 0:
            raise ValueError("lambda_lower_bound must be positive.")
        if self.lambda_upper_bound <= self.lambda_lower_bound:
            raise ValueError("lambda_upper_bound must exceed lambda_lower_bound.")
        if self.recombination_map_downsamples < 1:
            raise ValueError("recombination_map_downsamples must be positive.")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if self.likelihood_threshold < 0:
            raise ValueError("likelihood_threshold cannot be negative.")
        if self.parameter_tolerance <= 0 or self.objective_tolerance <= 0:
            raise ValueError("optimizer tolerances must be positive.")
        if self.final_time_factor is not None and self.final_time_factor <= 0:
            raise ValueError("final_time_factor must be positive when supplied.")
        if not self.optimization_method.strip():
            raise ValueError("optimization_method cannot be empty.")
        if self.decode_downsample < 1:
            raise ValueError("decode_downsample must be positive.")
        if self.cores is not None and self.cores < 1:
            raise ValueError("cores must be positive when supplied.")


def _resolve_paths(values: Sequence[str | Path], *, label: str) -> list[Path]:
    resolved: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PSMC+ {label} does not exist: {path}")
        resolved.append(path)
    return resolved


def _input_paths(data: SmcData, supplied: Sequence[str | Path] | None) -> list[Path]:
    values = list(supplied) if supplied is not None else list(data.uns.get("source_paths", []))
    if not values:
        raise ValueError(
            "PSMC+ requires path-backed multihetsep input. Read files with "
            "smckit.io.read_multihetsep(...) or pass input_paths explicitly."
        )
    return _resolve_paths(values, label="multihetsep input")


def _matched_map_paths(
    values: Sequence[str | Path] | None,
    *,
    input_count: int,
    label: str,
) -> list[Path]:
    if values is None:
        return []
    paths = _resolve_paths(values, label=label)
    if len(paths) != input_count:
        raise ValueError(
            f"PSMC+ requires one {label} per multihetsep input; "
            f"received {len(paths)} map(s) for {input_count} input(s)."
        )
    return paths


def _comma_values(values: Sequence[float] | str) -> str:
    if isinstance(values, str):
        return values
    return ",".join(str(float(value)) for value in values)


def _upstream_args(
    options: PSMCPlusOptions,
    *,
    inputs: Sequence[Path],
    mutation_maps: Sequence[Path],
    recombination_maps: Sequence[Path],
    output_path: Path,
    marginal_recombination_path: Path | None,
) -> list[str]:
    args = ["-in", *(str(path) for path in inputs)]
    if mutation_maps:
        args.extend(["-in_M", *(str(path) for path in mutation_maps)])
    if recombination_maps:
        args.extend(["-in_R", *(str(path) for path in recombination_maps)])
    args.extend(
        [
            "-o",
            str(output_path),
            "-D",
            str(options.number_time_windows),
            "-spread_1",
            str(options.spread_1),
            "-spread_2",
            str(options.spread_2),
            "-b",
            str(options.bin_size),
            "-mu_over_rho_ratio",
            str(options.mutation_recombination_ratio),
            "-lambda_lwr",
            str(options.lambda_lower_bound),
            "-lambda_upr",
            str(options.lambda_upper_bound),
            "-number_downsamples_R",
            str(options.recombination_map_downsamples),
            "-its",
            str(options.iterations),
            "-thresh",
            str(options.likelihood_threshold),
            "-xtol",
            str(options.parameter_tolerance),
            "-ftol",
            str(options.objective_tolerance),
            "-midpoint_transitions",
            str(options.midpoint_transitions),
            "-midpoint_emissions",
            str(options.midpoint_emissions),
            "-optimisation_method",
            options.optimization_method,
            "-decode_downsample",
            str(options.decode_downsample),
        ]
    )
    if options.scaled_recombination_rate is not None:
        args.extend(["-rho", str(options.scaled_recombination_rate)])
    if options.scaled_mutation_rate is not None:
        args.extend(["-theta", str(options.scaled_mutation_rate)])
    if options.rho_fixed:
        args.append("-rho_fixed")
    if options.lambda_initial is not None:
        args.extend(["-lambda_A_fg", _comma_values(options.lambda_initial)])
    if options.lambda_segments is not None:
        args.extend(["-lambda_A_segments", options.lambda_segments])
    if options.nonexponential_recombination:
        args.append("-recombnoexp")
    if options.final_time_factor is not None:
        args.extend(["-final_T_factor", str(options.final_time_factor)])
    if options.save_iteration_files:
        args.append("-save_iteration_files")
    if options.mode == "decode":
        args.append("-decode")
    if marginal_recombination_path is not None:
        args.extend(["-o_R", str(marginal_recombination_path)])
    if options.cores is not None:
        args.extend(["-c", str(options.cores)])
    return args


def _comment_payloads(text: str) -> list[str]:
    return [line[1:].strip() for line in text.splitlines() if line.startswith("#")]


def _header_value(text: str, label: str) -> str:
    for payload in _comment_payloads(text):
        if payload.startswith(label):
            return payload[len(label) :].lstrip(" =")
    raise ValueError(f"PSMC+ output is missing the {label!r} header.")


def _scale_time(
    scaled_time: np.ndarray,
    *,
    mutation_rate: float | None,
    generation_time: float,
) -> tuple[np.ndarray, str]:
    if mutation_rate is None:
        return scaled_time, "mutation_scaled"
    return scaled_time / mutation_rate * generation_time, "years"


def _parse_fit_result(
    path: Path,
    *,
    mutation_rate: float | None,
    generation_time: float,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[1] != 3:
        raise ValueError(f"Expected three PSMC+ fit columns, found shape {values.shape}.")
    theta = float(_header_value(text, "theta=4*N_E*mu"))
    rho = float(_header_value(text, "rho=4*N_E*r"))
    scaled_left = values[:, 0]
    scaled_right = values[:, 1]
    scaled_inverse_population_size = values[:, 2]
    if np.any(scaled_inverse_population_size <= 0):
        raise ValueError("PSMC+ returned a non-positive inverse population-size value.")
    left, time_units = _scale_time(
        scaled_left,
        mutation_rate=mutation_rate,
        generation_time=generation_time,
    )
    right, _ = _scale_time(
        scaled_right,
        mutation_rate=mutation_rate,
        generation_time=generation_time,
    )
    ne = 1.0 / scaled_inverse_population_size
    ne_units = "mutation_scaled"
    if mutation_rate is not None:
        ne = ne / mutation_rate
        ne_units = "individuals"
    result: dict[str, Any] = {
        "mode": "fit",
        "backend": "upstream",
        "time": left,
        "left_boundary": left,
        "right_boundary": right,
        "scaled_left_time_boundary": scaled_left,
        "scaled_right_time_boundary": scaled_right,
        "time_units": time_units,
        "ne": ne,
        "ne_units": ne_units,
        "lambda": scaled_inverse_population_size * theta / 4.0,
        "scaled_inverse_population_size": scaled_inverse_population_size,
        "theta": theta,
        "rho": rho,
        "log_likelihood": float(_header_value(text, "final log likelihood")),
        "likelihood_change": float(_header_value(text, "final change in log likelihood")),
        "n_iterations": int(_header_value(text, "number of iterations taken")),
    }
    if time_units == "years":
        result["time_years"] = left
    return result


def _decode_time_boundaries(text: str) -> np.ndarray:
    for payload in reversed(_comment_payloads(text)):
        if "," not in payload:
            continue
        try:
            values = np.asarray([float(value) for value in payload.split(",")], dtype=float)
        except ValueError:
            continue
        if values.size >= 2:
            return values
    raise ValueError("PSMC+ decoding output is missing its time-boundary header.")


def _parse_marginal_recombination(path: Path) -> dict[str, np.ndarray]:
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.shape[0] != 3:
        raise ValueError(
            f"Expected three PSMC+ marginal-recombination rows; found shape {values.shape}."
        )
    return {
        "position": values[0],
        "recombination_probability": values[1],
        "no_recombination_probability": values[2],
    }


def _parse_decode_result(
    path: Path,
    *,
    mutation_rate: float | None,
    generation_time: float,
    marginal_recombination_path: Path | None,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    theta = float(_header_value(text, "theta=4*N_E*mu"))
    raw_boundaries = _decode_time_boundaries(text)
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    expected_rows = raw_boundaries.size
    if values.shape[0] != expected_rows:
        raise ValueError(
            "PSMC+ posterior rows do not match the time grid: "
            f"{values.shape[0] - 1} states for {expected_rows - 1} intervals."
        )
    posterior = values[1:].T
    if np.any(~np.isfinite(posterior)) or np.any(posterior < 0):
        raise ValueError("PSMC+ returned invalid posterior probabilities.")
    if not np.allclose(posterior.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("PSMC+ posterior probabilities do not sum to one.")
    scaled_boundaries = 0.5 * raw_boundaries * theta
    boundaries, time_units = _scale_time(
        scaled_boundaries,
        mutation_rate=mutation_rate,
        generation_time=generation_time,
    )
    state_time = 0.5 * (boundaries[:-1] + boundaries[1:])
    result: dict[str, Any] = {
        "mode": "decode",
        "backend": "upstream",
        "position": values[0],
        "posterior": posterior,
        "time": state_time,
        "time_boundaries": boundaries,
        "coalescent_time_boundaries": raw_boundaries,
        "time_units": time_units,
        "posterior_mean_time": posterior @ state_time,
        "theta": theta,
        "rho": float(_header_value(text, "rho=4*N_E*r")),
        "bin_size": int(float(_header_value(text, "bin_size"))),
        "log_likelihood": float(_header_value(text, "log likelihood is")),
    }
    if time_units == "years":
        result["time_years"] = state_time
    if marginal_recombination_path is not None:
        result["marginal_recombination"] = _parse_marginal_recombination(
            marginal_recombination_path
        )
    return result


def _artifact_kind(relative_path: str, mode: PSMCPlusMode) -> str:
    if relative_path.endswith("final_parameters.txt"):
        return "final_parameters"
    if "params_iteration" in relative_path:
        return "iteration_parameters"
    if mode == "decode" and "marginal_recombination" in relative_path:
        return "marginal_recombination"
    if mode == "decode":
        return "posterior_decoding"
    return "upstream_output"


def _copy_artifacts(
    raw_artifacts: Sequence[dict[str, Any]],
    *,
    workdir: Path,
    mode: PSMCPlusMode,
    internal_output: Path,
    requested_output: Path | None,
    internal_marginal: Path | None,
    requested_marginal: Path | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_artifacts:
        relative = str(raw["path"])
        source = workdir / relative
        destination: Path | None = None
        if mode == "fit" and requested_output is not None:
            internal_prefix = internal_output.name
            if relative.startswith(internal_prefix):
                destination = Path(f"{requested_output}{relative[len(internal_prefix) :]}")
        elif mode == "decode" and relative == internal_output.name:
            destination = requested_output
        elif (
            mode == "decode"
            and internal_marginal is not None
            and relative == internal_marginal.name
        ):
            destination = requested_marginal
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        normalized.append(
            {
                "kind": _artifact_kind(relative, mode),
                "path": str(destination) if destination is not None else relative,
                "sha256": raw["sha256"],
                "size": raw["size"],
                "persisted": destination is not None,
                "upstream_path": relative,
            }
        )
    return normalized


def psmcplus(
    data: SmcData,
    *,
    options: PSMCPlusOptions | None = None,
    input_paths: Sequence[str | Path] | None = None,
    mutation_map_paths: Sequence[str | Path] | None = None,
    recombination_map_paths: Sequence[str | Path] | None = None,
    mutation_rate: float | None = None,
    generation_time: float = 1.0,
    output_prefix: str | Path | None = None,
    marginal_recombination_path: str | Path | None = None,
    implementation: str = "auto",
    timeout: float | None = None,
) -> SmcData:
    """Run preserved PSMC+ with a typed interface and normalized results.

    The complete original CLI remains available through
    ``smckit upstream psmcplus -- ...``. This typed adapter requires original
    path-backed multihetsep input because exact upstream execution must retain
    the source file representation. Native execution is intentionally rejected
    until an independently implemented PSMC+ path passes its parity gates.
    """
    requested = normalize_implementation(implementation)
    if requested == "native":
        raise NotImplementedError(
            "A smckit-native PSMC+ implementation is not available yet; use "
            "implementation='upstream' or 'auto'."
        )
    if not method_upstream_available("psmcplus"):
        require_upstream_available("psmcplus")

    resolved_options = options or PSMCPlusOptions()
    resolved_options.validate()
    inputs = _input_paths(data, input_paths)
    if resolved_options.mode == "decode" and len(inputs) != 1:
        raise ValueError("PSMC+ decoding accepts exactly one multihetsep input.")
    mutation_maps = _matched_map_paths(
        mutation_map_paths,
        input_count=len(inputs),
        label="mutation map",
    )
    recombination_maps = _matched_map_paths(
        recombination_map_paths,
        input_count=len(inputs),
        label="recombination map",
    )
    if mutation_rate is not None and mutation_rate <= 0:
        raise ValueError("mutation_rate must be positive when supplied.")
    if generation_time <= 0:
        raise ValueError("generation_time must be positive.")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive when supplied.")
    if marginal_recombination_path is not None and resolved_options.mode != "decode":
        raise ValueError("marginal_recombination_path is valid only in decode mode.")

    requested_output = (
        Path(output_prefix).expanduser().resolve() if output_prefix is not None else None
    )
    requested_marginal = (
        Path(marginal_recombination_path).expanduser().resolve()
        if marginal_recombination_path is not None
        else None
    )
    all_inputs = [*inputs, *mutation_maps, *recombination_maps]
    with tempfile.TemporaryDirectory(prefix="smckit-psmcplus-typed-") as temporary:
        workdir = Path(temporary)
        internal_output = (
            workdir / "psmcplus_posterior.txt"
            if resolved_options.mode == "decode"
            else workdir / "psmcplus_"
        )
        internal_marginal = (
            workdir / "psmcplus_marginal_recombination.txt"
            if requested_marginal is not None
            else None
        )
        args = _upstream_args(
            resolved_options,
            inputs=inputs,
            mutation_maps=mutation_maps,
            recombination_maps=recombination_maps,
            output_path=internal_output,
            marginal_recombination_path=internal_marginal,
        )
        effective_cores = resolved_options.cores or len(inputs)
        raw = upstream.run(
            "psmcplus",
            args,
            output_dir=workdir,
            timeout=timeout,
            env={
                "MKL_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        if raw.returncode != 0:
            detail = raw.stderr.strip() or raw.stdout.strip() or "no diagnostic output"
            raise RuntimeError(
                f"Original PSMC+ exited with status {raw.returncode}: {detail[-2000:]}"
            )
        if resolved_options.mode == "fit":
            result_path = Path(f"{internal_output}final_parameters.txt")
            if not result_path.is_file():
                raise RuntimeError("Original PSMC+ did not create final_parameters.txt.")
            result = _parse_fit_result(
                result_path,
                mutation_rate=mutation_rate,
                generation_time=generation_time,
            )
        else:
            if not internal_output.is_file():
                raise RuntimeError("Original PSMC+ did not create its posterior output.")
            result = _parse_decode_result(
                internal_output,
                mutation_rate=mutation_rate,
                generation_time=generation_time,
                marginal_recombination_path=internal_marginal,
            )
        artifacts = _copy_artifacts(
            raw.artifacts,
            workdir=workdir,
            mode=resolved_options.mode,
            internal_output=internal_output,
            requested_output=requested_output,
            internal_marginal=internal_marginal,
            requested_marginal=requested_marginal,
        )

    recorded_options = asdict(resolved_options)
    if resolved_options.lambda_initial is not None and not isinstance(
        resolved_options.lambda_initial, str
    ):
        recorded_options["lambda_initial"] = [
            float(value) for value in resolved_options.lambda_initial
        ]
    recorded_options.update(
        {
            "input_paths": [str(path) for path in inputs],
            "mutation_map_paths": [str(path) for path in mutation_maps],
            "recombination_map_paths": [str(path) for path in recombination_maps],
            "mutation_rate": mutation_rate,
            "generation_time": generation_time,
            "output_prefix": None if requested_output is None else str(requested_output),
            "marginal_recombination_path": (
                None if requested_marginal is None else str(requested_marginal)
            ),
            "effective_cores": effective_cores,
            "numeric_library_threads_per_worker": 1,
        }
    )
    upstream_metadata = standard_upstream_metadata(
        "psmcplus",
        effective_args=recorded_options,
        extra={
            "command": raw.command,
            "returncode": raw.returncode,
            "stdout": raw.stdout,
            "stderr": raw.stderr,
            "compatibility_patches": raw.compatibility_patches,
            "raw_artifacts": raw.artifacts,
        },
    )
    result["artifacts"] = artifacts
    annotate_result(
        result,
        method_name="psmcplus",
        implementation_requested=requested,
        implementation_used="upstream",
        upstream_metadata=upstream_metadata,
        effective_args=recorded_options,
        input_paths=[str(path) for path in all_inputs],
        runtime_seconds=raw.runtime_seconds,
        warning_messages=list(raw.compatibility_patches),
        artifacts=artifacts,
    )
    data.results["psmcplus"] = result
    return data


__all__ = ["PSMCPlusOptions", "psmcplus"]
