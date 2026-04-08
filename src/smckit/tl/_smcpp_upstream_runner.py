"""Run upstream SMC++ in a controlled side environment.

This script is executed by the Python interpreter from the dedicated upstream
SMC++ environment, not by the main smckit interpreter.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


def _patch_parallel_filters() -> None:
    import smcpp.data_filter as data_filter
    import smcpp.estimation_tools as estimation_tools

    def serial_load_data(files):
        return [estimation_tools._load_data_helper(f) for f in files]

    estimation_tools.load_data = serial_load_data
    data_filter.ProcessParallelFilter.Pool = data_filter.DummyPool
    data_filter.ThreadParallelFilter.Pool = data_filter.DummyPool


def _to_stepwise_ne(model) -> list[float]:
    return (
        np.asarray(model.stepwise_values(), dtype=float) * 2.0 * float(model.N0)
    ).tolist()


def _to_model_vector(model) -> list[float]:
    return np.asarray(model[:], dtype=float).tolist()


def _build_trace_plugin(enabled: bool):
    if not enabled:
        return None

    from smcpp.optimize.plugins.optimizer_plugin import OptimizerPlugin, targets

    class TracePlugin(OptimizerPlugin):
        DISABLED = False

        def __init__(self):
            self.rows = []

        def _append(self, phase: str, **kwargs) -> None:
            analysis = kwargs["analysis"]
            model = analysis.model
            self.rows.append({
                "phase": phase,
                "iteration": int(kwargs["i"]),
                "log_likelihood": float(analysis.loglik()),
                "q": float(analysis.Q()),
                "log_eta": _to_model_vector(model),
                "stepwise_ne": _to_stepwise_ne(model),
            })

        def update(self, message, *args, **kwargs):
            if message == "post E-step":
                self._append("post_e", **kwargs)
            elif message == "post M-step":
                self._append("post_m", **kwargs)

    return TracePlugin()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: _smcpp_upstream_runner.py <payload.json>")

    payload = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    os.makedirs(Path(payload["output_json"]).parent, exist_ok=True)

    _patch_parallel_filters()

    from smcpp.analysis.analysis import Analysis

    if payload["seed"] is not None:
        np.random.seed(int(payload["seed"]))

    args = Namespace(
        verbose=0,
        seed=0 if payload["seed"] is None else int(payload["seed"]),
        cores=1,
        outdir=str(Path(payload["output_json"]).parent / "analysis"),
        base="model",
        timepoints=None,
        length_cutoff=None,
        nonseg_cutoff=None,
        thinning=None,
        w=100,
        no_initialize=False,
        em_iterations=int(payload["max_iterations"]),
        algorithm="L-BFGS-B",
        multi=False,
        ftol=1e-4,
        xtol=0.1,
        Nmax=1e3,
        Nmin=1e-3,
        regularization_penalty=6,
        lambda_=float(payload["regularization"]),
        unfold=False,
        polarization_error=0.5,
        knots=int(payload["n_intervals"]),
        spline="piecewise",
        mu=float(payload["mu"]),
        r=float(payload["recombination_rate"]),
    )
    os.makedirs(args.outdir, exist_ok=True)

    analysis = Analysis(payload["input_paths"], args)
    trace_plugin = _build_trace_plugin(bool(payload.get("trace")))
    if trace_plugin is not None:
        analysis._optimizer.register_plugin(trace_plugin)
    analysis.run()

    model = analysis.model
    time = np.asarray(model.knots, dtype=float)
    ne = np.asarray(model(time), dtype=float) * 2.0 * float(model.N0)
    result = {
        "time": time.tolist(),
        "time_boundaries": np.r_[0.0, time].tolist(),
        "time_years": (time * 2.0 * float(model.N0) * float(payload["generation_time"])).tolist(),
        "theta": float(analysis._theta),
        "rho": float(analysis._rho),
        "alpha": float(analysis.alpha),
        "n0": float(model.N0),
        "ne": ne.tolist(),
        "log_likelihood": float(analysis.loglik()),
        "n_undist": int(sum(analysis.contigs[0].n)),
        "n_distinguished": int(sum(analysis.contigs[0].a)),
        "n_intervals": int(len(time)),
        "regularization": float(analysis._penalty),
        "optimization": {
            "success": True,
            "algorithm": args.algorithm,
            "n_iterations": int(payload["max_iterations"]),
            "history": [] if trace_plugin is None else trace_plugin.rows,
        },
        "model": model.to_dict(),
        "stepwise_ne": _to_stepwise_ne(model),
        "hidden_states": {k: list(v) for k, v in analysis.hidden_states.items()},
    }
    Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
