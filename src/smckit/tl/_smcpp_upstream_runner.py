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


def _to_float_array(values) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    return arr.tolist()


def _convert_gamma_sums(gamma_sums: dict) -> dict[str, list[float]]:
    out = {}
    for key, values in gamma_sums.items():
        out[str(tuple(key))] = _to_float_array(values)
    return out


def _convert_keyed_arrays(values: dict) -> dict[str, list[float]]:
    out = {}
    for key, arr in values.items():
        out[str(tuple(key))] = _to_float_array(arr)
    return out


def _fixed_model_stats(analysis, model_vector: list[float], alpha: float) -> dict:
    analysis.model[:] = np.asarray(model_vector, dtype=float)
    analysis.alpha = float(alpha)
    im = list(analysis._ims.values())[0]
    im.save_gamma = True
    analysis.E_step()
    gammas = np.asarray(im.gammas[0], dtype=float)
    return {
        "im_hidden_states": _to_float_array(im.hidden_states),
        "analysis_hidden_states": {k: list(v) for k, v in analysis.hidden_states.items()},
        "pi": _to_float_array(im.pi),
        "gammas": gammas.tolist(),
        "gamma0": _to_float_array(gammas[:, 0]),
        "gamma_sums": _convert_gamma_sums(im.gamma_sums[0]),
        "xisum": np.asarray(im.xisums, dtype=float).tolist(),
        "transition": np.asarray(im.transition, dtype=float).tolist(),
        "emission_probs": _convert_keyed_arrays(im.emission_probs),
        "q": float(analysis.Q()),
        "log_likelihood": float(analysis.loglik(reg=False)),
    }


def _fixed_model_stats_from_dict(analysis, model_dict: dict, alpha: float) -> dict:
    from smcpp.model import SMCModel

    analysis.model = SMCModel.from_dict(model_dict)
    analysis.alpha = float(alpha)
    im = list(analysis._ims.values())[0]
    im.save_gamma = True
    analysis.E_step()
    gammas = np.asarray(im.gammas[0], dtype=float)
    return {
        "im_hidden_states": _to_float_array(im.hidden_states),
        "analysis_hidden_states": {k: list(v) for k, v in analysis.hidden_states.items()},
        "pi": _to_float_array(im.pi),
        "gammas": gammas.tolist(),
        "gamma0": _to_float_array(gammas[:, 0]),
        "gamma_sums": _convert_gamma_sums(im.gamma_sums[0]),
        "xisum": np.asarray(im.xisums, dtype=float).tolist(),
        "transition": np.asarray(im.transition, dtype=float).tolist(),
        "emission_probs": _convert_keyed_arrays(im.emission_probs),
        "q": float(analysis.Q()),
        "log_likelihood": float(analysis.loglik(reg=False)),
    }


def _fixed_stats_q_compare(
    analysis,
    start_model_vector: list[float],
    candidate_models: dict[str, list[float]],
    alpha: float,
) -> dict:
    analysis.model[:] = np.asarray(start_model_vector, dtype=float)
    analysis.alpha = float(alpha)
    analysis.E_step()
    out = {}
    for name, model_vector in candidate_models.items():
        analysis.model[:] = np.asarray(model_vector, dtype=float)
        out[name] = float(analysis.Q())
    return out


def _fixed_stats_one_mstep(
    analysis,
    start_model_vector: list[float],
    alpha: float,
) -> dict:
    analysis.model[:] = np.asarray(start_model_vector, dtype=float)
    analysis.alpha = float(alpha)
    analysis.E_step()

    optimizer = analysis._optimizer
    kwargs = {"i": 0, "niter": 1}
    scale_trace = []
    for plugin in optimizer._plugins:
        if type(plugin).__name__ != "ScaleOptimizer":
            continue
        orig_f = plugin._f

        def _wrapped_f(alpha_shift, x0, analysis_obj, _orig_f=orig_f):
            val = _orig_f(alpha_shift, x0, analysis_obj)
            scale_trace.append({
                "alpha": float(alpha_shift),
                "objective": float(val),
                "model_vector": _to_model_vector(analysis_obj.model),
            })
            return val

        plugin._f = _wrapped_f
        break
    pre_m_model = _to_model_vector(analysis.model)
    pre_m_q = float(analysis.Q())
    optimizer.update_observers("pre M-step", **kwargs)
    post_pre_m_model = _to_model_vector(analysis.model)
    post_pre_m_q = float(analysis.Q())
    coord_list = optimizer._coordinates()
    mini_steps = []
    for coords in coord_list:
        optimizer.update_observers("M step", coords=coords, **kwargs)
        x0 = optimizer[coords]
        optimizer._bounds = np.transpose(
            [
                np.maximum(x0 - 3.0, np.log(1e-4)),
                np.minimum(x0 + 3.0, np.log(1e4)),
            ]
        )
        res = optimizer._minimize(x0, coords)
        optimizer.update_observers("post minimize", coords=coords, res=res, **kwargs)
        optimizer[coords] = res.x
        optimizer.update_observers("post mini M-step", coords=coords, res=res, **kwargs)
        mini_steps.append({
            "coords": [int(c) for c in coords],
            "x": np.asarray(res.x, dtype=float).tolist(),
            "fun": float(res.fun),
        })
    optimizer.update_observers("post M-step", **kwargs)
    return {
        "plugins": [type(p).__name__ for p in optimizer._plugins],
        "pre_m_model_vector": pre_m_model,
        "pre_m_q": pre_m_q,
        "post_pre_m_model_vector": post_pre_m_model,
        "post_pre_m_q": post_pre_m_q,
        "scale_trace": scale_trace,
        "final_model_vector": _to_model_vector(analysis.model),
        "mini_steps": mini_steps,
        "q": float(analysis.Q()),
    }


def _onepop_initialization_summary(args, input_paths: list[str]) -> dict:
    from smcpp import data_filter, estimation_tools, spline
    from smcpp.analysis import analysis as analysis_mod
    from smcpp.analysis import base as analysis_base
    from smcpp.model import SMCModel

    analysis = analysis_mod.Analysis.__new__(analysis_mod.Analysis)
    analysis_base.BaseAnalysis.__init__(analysis, input_paths, args)

    if analysis.npop != 1:
        raise RuntimeError("one-pop initialization summary requires npop == 1")

    NeN0 = analysis._pipeline["watterson"].theta_hat / (2.0 * args.mu * analysis._N0)
    m = SMCModel([1.0], analysis._N0, spline.Piecewise, None)
    m[:] = np.log(NeN0)
    hs = estimation_tools.balance_hidden_states(m, 2 + args.knots)
    hs /= 2.0 * analysis._N0
    analysis.hidden_states = hs
    analysis._init_knots(hs, None, None)
    prefit_knots = np.asarray(analysis._knots, dtype=float)

    analysis._init_model(args.spline)
    analysis.hidden_states = [0.0, np.inf]
    analysis._init_inference_manager(args.polarization_error, analysis.hidden_states)
    analysis.alpha = 1
    analysis._model[:] = np.log(NeN0)
    analysis._model.randomize()
    prefit_randomized = _to_model_vector(analysis.model)
    analysis._init_optimizer(
        args.outdir,
        args.base,
        args.algorithm,
        args.xtol,
        args.ftol,
        learn_rho=False,
        single=False,
    )
    analysis._init_regularization(args)
    analysis.run(1)
    prefit_after = _to_model_vector(analysis.model)

    pipe = analysis._pipeline
    pipe.add_filter(data_filter.Thin(thinning=args.thinning))
    pipe.add_filter(data_filter.BinObservations(w=args.w))
    pipe.add_filter(data_filter.RecodeMonomorphic())
    pipe.add_filter(data_filter.Compress())
    pipe.add_filter(data_filter.Validate())
    pipe.add_filter(data_filter.DropUninformativeContigs())
    pipe.add_filter(data_filter.Summarize())
    try:
        analysis._empirical_tmrca(2 * args.knots)
        hs = np.r_[0.0, analysis._etmrca_quantiles, np.inf]
    except Exception:
        hs = estimation_tools.balance_hidden_states(m, 2 * args.knots) / 2.0 / analysis._N0
    analysis.hidden_states = hs
    analysis._init_knots(hs, None, None)
    final_knots = np.asarray(analysis._knots, dtype=float)
    final_hidden_states = np.asarray(analysis.hidden_states[analysis.populations[0]], dtype=float)
    m = analysis._model
    analysis._init_model(args.spline)
    analysis._model[:] = np.log(m(analysis._knots))

    return {
        "watterson_ne_n0": float(NeN0),
        "prefit_knots": _to_float_array(prefit_knots),
        "prefit_randomized_model_vector": prefit_randomized,
        "prefit_model_vector": prefit_after,
        "final_knots": _to_float_array(final_knots),
        "final_hidden_states": _to_float_array(final_hidden_states),
        "main_init_model_vector": _to_model_vector(analysis.model),
    }


def _build_onepop_prefit_analysis(args, input_paths: list[str]):
    from smcpp import estimation_tools, spline
    from smcpp.analysis import analysis as analysis_mod
    from smcpp.analysis import base as analysis_base
    from smcpp.model import SMCModel

    analysis = analysis_mod.Analysis.__new__(analysis_mod.Analysis)
    analysis_base.BaseAnalysis.__init__(analysis, input_paths, args)

    if analysis.npop != 1:
        raise RuntimeError("one-pop prefit analysis requires npop == 1")

    NeN0 = analysis._pipeline["watterson"].theta_hat / (2.0 * args.mu * analysis._N0)
    m = SMCModel([1.0], analysis._N0, spline.Piecewise, None)
    m[:] = np.log(NeN0)
    hs = estimation_tools.balance_hidden_states(m, 2 + args.knots)
    hs /= 2.0 * analysis._N0
    analysis.hidden_states = hs
    analysis._init_knots(hs, None, None)
    analysis._init_model(args.spline)
    analysis.hidden_states = [0.0, np.inf]
    analysis._init_inference_manager(args.polarization_error, analysis.hidden_states)
    analysis.alpha = 1
    analysis._model[:] = np.log(NeN0)
    analysis._init_optimizer(
        args.outdir,
        args.base,
        args.algorithm,
        args.xtol,
        args.ftol,
        learn_rho=False,
        single=False,
    )
    analysis._init_regularization(args)
    return analysis


def _onepop_prefit_fixed_stats_q_compare(
    args,
    input_paths: list[str],
    start_model_vector: list[float],
    candidate_models: dict[str, list[float]],
) -> dict:
    analysis = _build_onepop_prefit_analysis(args, input_paths)
    analysis.model[:] = np.asarray(start_model_vector, dtype=float)
    analysis.E_step()
    out = {}
    for name, model_vector in candidate_models.items():
        analysis.model[:] = np.asarray(model_vector, dtype=float)
        out[name] = float(analysis.Q())
    return out


def _onepop_prefit_fixed_model_stats(
    args,
    input_paths: list[str],
    model_vector: list[float],
) -> dict:
    analysis = _build_onepop_prefit_analysis(args, input_paths)
    analysis.model[:] = np.asarray(model_vector, dtype=float)
    im = list(analysis._ims.values())[0]
    im.save_gamma = True
    analysis.E_step()
    gammas = np.asarray(im.gammas[0], dtype=float)
    return {
        "prefit_knots": _to_float_array(analysis.model.knots),
        "hidden_states": _to_float_array(im.hidden_states),
        "gamma0": _to_float_array(gammas[:, 0]),
        "gamma_sums": _convert_gamma_sums(im.gamma_sums[0]),
        "xisum": np.asarray(im.xisums, dtype=float).tolist(),
        "transition": np.asarray(im.transition, dtype=float).tolist(),
        "emission_probs": _convert_keyed_arrays(im.emission_probs),
        "q": float(analysis.Q()),
        "log_likelihood": float(analysis.loglik(reg=False)),
    }


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
    if payload.get("mode") == "fixed_model_stats":
        if "model_dict" in payload:
            result = _fixed_model_stats_from_dict(
                analysis,
                model_dict=payload["model_dict"],
                alpha=float(payload["alpha"]),
            )
        else:
            result = _fixed_model_stats(
                analysis,
                model_vector=payload["model_vector"],
                alpha=float(payload["alpha"]),
            )
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    if payload.get("mode") == "fixed_stats_q_compare":
        result = _fixed_stats_q_compare(
            analysis,
            start_model_vector=payload["start_model_vector"],
            candidate_models=payload["candidate_models"],
            alpha=float(payload["alpha"]),
        )
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    if payload.get("mode") == "fixed_stats_one_mstep":
        result = _fixed_stats_one_mstep(
            analysis,
            start_model_vector=payload["start_model_vector"],
            alpha=float(payload["alpha"]),
        )
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    if payload.get("mode") == "onepop_initialization_summary":
        result = _onepop_initialization_summary(args, payload["input_paths"])
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    if payload.get("mode") == "onepop_prefit_fixed_stats_q_compare":
        result = _onepop_prefit_fixed_stats_q_compare(
            args,
            payload["input_paths"],
            start_model_vector=payload["start_model_vector"],
            candidate_models=payload["candidate_models"],
        )
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0
    if payload.get("mode") == "onepop_prefit_fixed_model_stats":
        result = _onepop_prefit_fixed_model_stats(
            args,
            payload["input_paths"],
            model_vector=payload["model_vector"],
        )
        Path(payload["output_json"]).write_text(json.dumps(result), encoding="utf-8")
        return 0

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
