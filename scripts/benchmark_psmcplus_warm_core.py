#!/usr/bin/env python3
"""Compare warmed native and pinned-upstream PSMC+ inference cores fairly."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from smckit._provenance import sha256_file
from smckit.tl._psmcplus_native import decode_psmcplus_native, fit_psmcplus_native

UPSTREAM_COMMIT = "032168f2ceed3c0e46b7f214f890faf83dff41ae"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _measure(call, repetitions: int) -> tuple[list[float], object]:
    call()
    times: list[float] = []
    value: object = None
    for _ in range(repetitions):
        started = time.perf_counter()
        value = call()
        times.append(time.perf_counter() - started)
    return times, value


def main() -> int:
    args = _parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    if args.bootstrap_replicates < 1:
        raise SystemExit("bootstrap-replicates must be positive")
    input_path = args.input.expanduser().resolve()
    source = Path(__file__).resolve().parents[1] / "vendor/PSMCplus"
    if not source.is_dir():
        raise SystemExit("Initialize the pinned vendor/PSMCplus submodule first.")
    np.math = math  # type: ignore[attr-defined]
    sys.path.insert(0, str(source))

    from BaumWelch import BaumWelch, parse_lambda_fg  # noqa: PLC0415
    from transition_matrix import Transition_Matrix, time_intervals  # noqa: PLC0415
    from utils import (  # noqa: PLC0415
        backward_matmul_scaled_fcn,
        bin_sequence,
        calculate_transition_evidence,
        forward_matmul_scaled_fcn,
        write_emission_masked_probs,
        write_emission_probs,
        write_Q_array_withR,
        write_segments,
    )

    number_states = 4
    bin_size = 100
    spread_1 = 0.1
    spread_2 = 50.0
    input_label = str(input_path)
    with contextlib.redirect_stdout(io.StringIO()):
        sequences_info = [
            bin_sequence(
                input_label,
                bin_size,
                {input_label: "null"},
                {input_label: "null"},
            )
        ]
    theta = sequences_info[0][4] / (sequences_info[0][3] - sequences_info[0][5])
    rho_per_bin = theta / 1.5 * bin_size
    boundaries = time_intervals(number_states, spread_1, spread_2)
    emissions = write_emission_probs(
        number_states,
        bin_size,
        theta,
        sequences_info[0][2],
        boundaries,
        midpoint_end=False,
    )
    masked_emissions = write_emission_masked_probs(
        number_states,
        bin_size,
        theta,
        sequences_info[0][2],
        boundaries,
        midpoint_end=False,
    )
    segments = write_segments(None, number_states)
    initial_lambda = parse_lambda_fg(None, segments)

    def upstream_model() -> BaumWelch:
        return BaumWelch(
            sequences_info=sequences_info,
            D=number_states,
            E=emissions,
            E_masked=masked_emissions,
            lambda_A_values=initial_lambda.copy(),
            lambda_B_values=None,
            gamma_fg=None,
            lambda_A_segs=segments,
            lambda_B_segs=None,
            rho=rho_per_bin,
            theta=theta,
            estimate_rho=True,
            final_T_factor=None,
            T_array=boundaries,
            bin_size=bin_size,
            T_S=None,
            T_E=None,
            j_max=sequences_info[0][2],
            spread_1=spread_1,
            spread_2=spread_2,
            lambda_lwr_bnd=0.1,
            lambda_upr_bnd=50.0,
            gamma_lwr_bnd=None,
            gamma_upr_bnd=None,
            output_path=None,
            cores=1,
            xtol=1e-4,
            ftol=1e-4,
            midpoint_transitions=False,
            midpoint_end=False,
            optimisation_method="Powell",
            save_iteration_files=False,
            lambda_lwr_bnd_struct=None,
            lambda_upr_bnd_struct=None,
            recombnoexp=False,
        )

    def final_upstream_likelihood(model: BaumWelch) -> float:
        sequence, mutation_indices, mutation_values, recombination_indices, rates = (
            model.sequence_fcn(0)
        )
        transition_model = Transition_Matrix(
            D=number_states,
            spread_1=spread_1,
            spread_2=spread_2,
        )
        transition_model.write_tm(
            lambda_A=model.lambda_A_current,
            lambda_B=None,
            T_S_index=None,
            T_E_index=None,
            gamma=None,
            check=True,
            rho=model.rho,
        )
        transitions = write_Q_array_withR(
            transition_model.Q,
            rates,
            model.rho,
            number_states,
            spread_1,
            spread_2,
            model.lambda_A_current,
            False,
        )

        def sequence_function(_index: int):
            return (
                sequence,
                mutation_indices,
                mutation_values,
                recombination_indices,
                rates,
            )

        return float(
            calculate_transition_evidence(
                sequence_function,
                0,
                number_states,
                model.init_dist,
                model.E_masked,
                transitions,
                theta,
                model.rho,
                bin_size,
                model.j_max,
                model.midpoints,
                spread_1,
                spread_2,
                False,
            )[1]
        )

    def upstream_fit():
        with contextlib.redirect_stdout(io.StringIO()):
            model = upstream_model()
            model.BaumWelch(BW_iterations=1, BW_thresh=0)
            likelihood = final_upstream_likelihood(model)
        return model.lambda_A_current.copy(), model.rho / bin_size, likelihood

    def native_fit():
        fit = fit_psmcplus_native(
            [input_path],
            number_states=number_states,
            bin_size=bin_size,
            iterations=1,
            likelihood_threshold=0,
            cores=1,
        )
        return fit.lambda_values, fit.rho, fit.final_log_likelihood

    def upstream_decode():
        with contextlib.redirect_stdout(io.StringIO()):
            model = upstream_model()
        sequence, mutation_indices, mutation_values, recombination_indices, rates = (
            model.sequence_fcn(0)
        )
        transition_model = Transition_Matrix(
            D=number_states,
            spread_1=spread_1,
            spread_2=spread_2,
        )
        transition_model.write_tm(
            lambda_A=model.lambda_A_current,
            lambda_B=None,
            T_S_index=None,
            T_E_index=None,
            gamma=None,
            check=True,
            rho=model.rho,
        )
        transitions = write_Q_array_withR(
            transition_model.Q,
            rates,
            model.rho,
            number_states,
            spread_1,
            spread_2,
            model.lambda_A_current,
            False,
        )
        forward, scales = forward_matmul_scaled_fcn(
            sequence,
            number_states,
            model.init_dist,
            model.E_masked,
            transitions,
            bin_size,
            theta,
            model.midpoints,
            mutation_indices,
            mutation_values,
            recombination_indices,
        )
        backward = backward_matmul_scaled_fcn(
            sequence,
            number_states,
            transitions,
            bin_size,
            theta,
            model.midpoints,
            model.E_masked,
            scales,
            mutation_indices,
            mutation_values,
            recombination_indices,
        )
        posterior = (forward * backward).T
        posterior /= posterior.sum(axis=1, keepdims=True)
        return posterior, float(np.log(scales).sum())

    def native_decode():
        decode = decode_psmcplus_native(
            input_path,
            number_states=number_states,
            bin_size=bin_size,
            lambda_initial=[1, 1, 1, 1],
            downsample=1,
        )
        return decode.posterior, decode.log_likelihood

    upstream_fit_times, upstream_fit_value = _measure(upstream_fit, args.repetitions)
    native_fit_times, native_fit_value = _measure(native_fit, args.repetitions)
    upstream_decode_times, upstream_decode_value = _measure(upstream_decode, args.repetitions)
    native_decode_times, native_decode_value = _measure(native_decode, args.repetitions)

    np.testing.assert_allclose(native_fit_value[0], upstream_fit_value[0], rtol=1e-8)
    np.testing.assert_allclose(native_fit_value[1:], upstream_fit_value[1:], rtol=1e-8)
    np.testing.assert_allclose(native_decode_value[0], upstream_decode_value[0], atol=4e-14)
    np.testing.assert_allclose(native_decode_value[1], upstream_decode_value[1], atol=1e-12)

    def summary(native: list[float], upstream: list[float], *, seed: int) -> dict[str, object]:
        native_median = statistics.median(native)
        upstream_median = statistics.median(upstream)
        native_array = np.asarray(native, dtype=np.float64)
        upstream_array = np.asarray(upstream, dtype=np.float64)
        rng = np.random.default_rng(seed)
        sampled_native = rng.choice(
            native_array,
            size=(args.bootstrap_replicates, native_array.size),
            replace=True,
        )
        sampled_upstream = rng.choice(
            upstream_array,
            size=(args.bootstrap_replicates, upstream_array.size),
            replace=True,
        )
        speedups = np.median(sampled_upstream, axis=1) / np.median(sampled_native, axis=1)
        interval = np.quantile(speedups, [0.025, 0.975])
        return {
            "native_seconds": native,
            "upstream_seconds": upstream,
            "native_median_seconds": native_median,
            "upstream_median_seconds": upstream_median,
            "speedup": upstream_median / native_median,
            "speedup_confidence_interval": [float(interval[0]), float(interval[1])],
            "confidence": 0.95,
            "faster_with_confidence": bool(interval[0] > 1.0),
        }

    repository = Path(__file__).resolve().parents[1]
    try:
        input_label = str(input_path.relative_to(repository))
    except ValueError:
        input_label = str(input_path)
    payload = {
        "schema_version": 1,
        "method": "psmcplus",
        "upstream_commit": UPSTREAM_COMMIT,
        "input": input_label,
        "input_sha256": sha256_file(input_path),
        "threads": 1,
        "repetitions": args.repetitions,
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "measurement_component": "warmed inference core; upstream preprocessing excluded",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "fit": summary(native_fit_times, upstream_fit_times, seed=args.seed),
        "decode": summary(native_decode_times, upstream_decode_times, seed=args.seed + 1),
    }
    payload["runtime_gate_passed"] = bool(
        payload["fit"]["faster_with_confidence"] and payload["decode"]["faster_with_confidence"]
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
