"""Generate paired native/reference gallery figures with matched axes.

Each method is rendered twice on the same inference task:

- one panel for the native implementation
- one panel for the upstream/reference implementation

The paired figures use the same axis ranges so left/right comparison is direct.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smckit
from smckit.backends._numba_esmc2 import (
    esmc2_build_time_boundaries,
    esmc2_build_transition_matrix,
    esmc2_equilibrium_probs,
    esmc2_expected_times,
)
from smckit._core import SmcData
from smckit.ext.ssm import PsmcSSM
from smckit.io import read_msmc_im_output, read_psmc_output, read_smcpp_input
from smckit.tl import asmc, dical2, esmc2, msmc_im, msmc2, psmc, smcpp

from compare_dical2_oracle import EXAMPLES, _run_oracle, _run_smckit
from generate_method_gallery import (
    COLORS,
    MU,
    OUTDIR,
    R,
    _load_msmc_validation_module,
    apply_style,
    savefig,
    simulate_zigzag_ts,
)

os.environ.setdefault("SMCKIT_ESMC2_RSCRIPT", "/opt/homebrew/bin/Rscript")
PATTERN = "4+5*3+4"
GEN = 25.0


def _add_stats(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(
        0.03,
        0.04,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )


def _positive_limits(*arrays: np.ndarray, log_pad: float = 0.06) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.extend(a[np.isfinite(a) & (a > 0)])
    if not vals:
        return (1e-3, 1.0)
    vals_arr = np.asarray(vals, dtype=float)
    lo = float(vals_arr.min())
    hi = float(vals_arr.max())
    if lo == hi:
        lo *= 0.8
        hi *= 1.2
    return (lo * (10 ** (-log_pad)), hi * (10**log_pad))


def _linear_limits(*arrays: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.extend(a[np.isfinite(a)])
    if not vals:
        return (0.0, 1.0)
    vals_arr = np.asarray(vals, dtype=float)
    lo = float(vals_arr.min())
    hi = float(vals_arr.max())
    if lo == hi:
        delta = 1.0 if hi == 0 else abs(hi) * 0.2
        return (lo - delta, hi + delta)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _step_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    ax.step(np.asarray(x, dtype=float), np.asarray(y, dtype=float), where="post", **kwargs)


def _history_panel(
    *,
    output_name: str,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str = "Generations before present",
    ylabel: str = "Effective population size",
    stats: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    _step_plot(ax, x, y, color=color, linewidth=2.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    if stats:
        _add_stats(ax, stats)
    savefig(fig, output_name)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_curve_panel(
    *,
    output_name: str,
    title: str,
    x: np.ndarray,
    top_curves: list[tuple[np.ndarray, str, str]],
    bottom_curves: list[tuple[np.ndarray, str, str]],
    xlim: tuple[float, float],
    top_ylim: tuple[float, float],
    bottom_ylim: tuple[float, float],
    top_logy: bool = True,
    bottom_logy: bool = False,
    top_ylabel: str = "",
    bottom_ylabel: str = "",
    xlabel: str = "Generations before present",
    stats: list[str] | None = None,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.1, 7.1), constrained_layout=True)
    for arr, label, color in top_curves:
        _step_plot(ax_top, x, arr, color=color, linewidth=2.2, label=label)
    for arr, label, color in bottom_curves:
        _step_plot(ax_bottom, x, arr, color=color, linewidth=2.2, label=label)
    ax_top.set_title(title)
    ax_top.set_xscale("log")
    ax_bottom.set_xscale("log")
    if top_logy:
        ax_top.set_yscale("log")
    if bottom_logy:
        ax_bottom.set_yscale("log")
    ax_top.set_xlim(*xlim)
    ax_bottom.set_xlim(*xlim)
    ax_top.set_ylim(*top_ylim)
    ax_bottom.set_ylim(*bottom_ylim)
    ax_top.set_ylabel(top_ylabel)
    ax_bottom.set_ylabel(bottom_ylabel)
    ax_bottom.set_xlabel(xlabel)
    ax_top.legend(loc="upper right")
    ax_bottom.legend(loc="upper right")
    if stats:
        _add_stats(ax_bottom, stats)
    savefig(fig, output_name)


def _asmc_panel(
    *,
    output_name: str,
    title: str,
    x_mb: np.ndarray,
    posterior: np.ndarray,
    maps: np.ndarray,
    xlim: tuple[float, float],
    posterior_ylim: tuple[float, float],
    map_ylim: tuple[float, float],
    color: str,
    label: str,
    stats: list[str] | None = None,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.2, 7.0), constrained_layout=True)
    ax_top.plot(x_mb, posterior, color=color, linewidth=2.0, label=label)
    ax_top.set_yscale("log")
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(*posterior_ylim)
    ax_top.set_ylabel("Posterior mean TMRCA")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")

    ax_bottom.plot(x_mb, maps, color=color, linewidth=1.3, label=label)
    ax_bottom.set_xlim(*xlim)
    ax_bottom.set_ylim(*map_ylim)
    ax_bottom.set_xlabel("Physical position (Mb)")
    ax_bottom.set_ylabel("MAP state index")
    ax_bottom.legend(loc="upper right")
    if stats:
        _add_stats(ax_bottom, stats)
    savefig(fig, output_name)


def _esmc2_panel(
    *,
    output_name: str,
    title: str,
    tc: np.ndarray,
    xi: np.ndarray,
    tc_xlim: tuple[float, float],
    tc_ylim: tuple[float, float],
    xi_xlim: tuple[float, float],
    xi_ylim: tuple[float, float],
    color: str,
    label: str,
    stats: list[str] | None = None,
) -> None:
    states = np.arange(len(xi))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.8, 7.0), constrained_layout=True)
    ax_top.plot(states, xi, marker="o", color=color, linewidth=2.0, label=label)
    ax_top.set_xlim(*xi_xlim)
    ax_top.set_ylim(*xi_ylim)
    ax_top.set_ylabel("Xi (relative Ne)")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")

    ax_bottom.plot(states, tc, marker="o", color=color, linewidth=2.0, label=label)
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlim(*tc_xlim)
    ax_bottom.set_ylim(*tc_ylim)
    ax_bottom.set_xlabel("Hidden state index")
    ax_bottom.set_ylabel("Tc (time boundary)")
    ax_bottom.legend(loc="upper right")
    if stats:
        _add_stats(ax_bottom, stats)
    savefig(fig, output_name)


def fig_psmc_dual() -> None:
    data = smckit.io.read_psmcfa(ROOT / "tests" / "data" / "NA12878_chr22.psmcfa")
    native = psmc(
        data,
        pattern=PATTERN,
        n_iterations=10,
        max_t=15.0,
        tr_ratio=5.0,
        mu=MU,
        generation_time=GEN,
        seed=42,
        implementation="native",
    ).results["psmc"]
    ref = read_psmc_output(ROOT / "tests" / "data" / "NA12878_chr22.psmc")[-1]
    n0_ref = ref["theta"] / (4.0 * MU * data.window_size)
    ref_time = ref["time"] * 2.0 * n0_ref * GEN
    ref_ne = ref["lambda"] * n0_ref
    xlim = _positive_limits(native["time_years"], ref_time)
    ylim = _positive_limits(native["ne"], ref_ne)
    corr = float(np.corrcoef(native["lambda"], ref["lambda"])[0, 1])

    _history_panel(
        output_name="psmc_native.png",
        title="PSMC on NA12878 chr22",
        x=np.asarray(native["time_years"]),
        y=np.asarray(native["ne"]),
        color=COLORS["smckit"],
        label="native",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"theta={float(native['theta']):.3e}", f"rho={float(native['rho']):.3e}", f"corr={corr:.6f}"],
    )
    _history_panel(
        output_name="psmc_upstream.png",
        title="PSMC on NA12878 chr22",
        x=np.asarray(ref_time),
        y=np.asarray(ref_ne),
        color=COLORS["oracle"],
        label="upstream",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"theta={float(ref['theta']):.3e}", f"rho={float(ref['rho']):.3e}", "source=vendored .psmc"],
    )


def fig_msmc2_dual() -> None:
    validation = _load_msmc_validation_module()
    data = smckit.io.read_multihetsep(ROOT / "data" / "msmc2_test.multihetsep")
    native = msmc2(data, n_iterations=2, mu=MU, generation_time=25.0, implementation="native").results["msmc2"]
    ref_lambda = validation.EXPECTED_LAMBDA_ALL_2
    ref_left = validation.EXPECTED_LEFT_ALL_1
    ref_time = ref_left / MU * 25.0
    ref_ne = 1.0 / (2.0 * MU * ref_lambda)
    xlim = _positive_limits(native["time_years"], ref_time)
    ylim = _positive_limits(native["ne"], ref_ne)
    corr = float(np.corrcoef(native["lambda"], ref_lambda)[0, 1])

    _history_panel(
        output_name="msmc2_native.png",
        title="MSMC2 on the bundled multihetsep fixture",
        x=np.asarray(native["time_years"]),
        y=np.asarray(native["ne"]),
        color=COLORS["smckit"],
        label="native",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"segments={len(native['lambda'])}", f"loglik={float(native['log_likelihood']):.3f}", f"corr={corr:.6f}"],
    )
    _history_panel(
        output_name="msmc2_upstream.png",
        title="MSMC2 on the bundled multihetsep fixture",
        x=np.asarray(ref_time),
        y=np.asarray(ref_ne),
        color=COLORS["oracle"],
        label="upstream fixture",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"segments={len(ref_lambda)}", "source=vendored fixture"],
    )


def fig_smcpp_dual() -> None:
    tracked_fixture = ROOT / "tests" / "data" / "smcpp_onepop_larger.smc"
    data = read_smcpp_input(tracked_fixture)
    params = {
        "n_intervals": 4,
        "max_iterations": 2,
        "generation_time": 1.0,
        "regularization": 10.0,
        "seed": 42,
    }
    native = smcpp(copy.deepcopy(data), implementation="native", **params).results["smcpp"]
    upstream = smcpp(copy.deepcopy(data), implementation="upstream", **params).results["smcpp"]
    grid = np.geomspace(
        max(
            float(np.min(np.asarray(native["time"], dtype=float)[np.asarray(native["time"], dtype=float) > 0])),
            float(np.min(np.asarray(upstream["time"], dtype=float)[np.asarray(upstream["time"], dtype=float) > 0])),
        ),
        min(float(np.max(native["time"])), float(np.max(upstream["time"]))),
        200,
    )

    def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, query, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    native_ne_grid = _step_eval(np.asarray(native["time"], dtype=float), np.asarray(native["ne"], dtype=float), grid)
    upstream_ne_grid = _step_eval(np.asarray(upstream["time"], dtype=float), np.asarray(upstream["ne"], dtype=float), grid)
    log_corr = float(np.corrcoef(np.log(native_ne_grid), np.log(upstream_ne_grid))[0, 1])
    scale_ratio = float(np.median(native_ne_grid) / np.median(upstream_ne_grid))
    median_rel = float(
        np.median(np.abs(native_ne_grid - upstream_ne_grid) / np.maximum(np.abs(upstream_ne_grid), 1e-12))
    )
    xlim = _positive_limits(native["time_years"], upstream["time_years"])
    ylim = _positive_limits(native["ne"], upstream["ne"])

    _history_panel(
        output_name="smcpp_native.png",
        title="SMC++ on the tracked larger one-pop fixture",
        x=np.asarray(native["time_years"]),
        y=np.asarray(native["ne"]),
        color=COLORS["smckit"],
        label="native",
        xlim=xlim,
        ylim=ylim,
        xlabel="Generations before present",
        stats=[
            f"theta={float(native['theta']):.3e}",
            f"rho={float(native['rho']):.3e}",
            f"log corr={log_corr:.6f}",
            f"scale ratio={scale_ratio:.6f}",
            f"median rel err={median_rel:.6f}",
        ],
    )
    _history_panel(
        output_name="smcpp_upstream.png",
        title="SMC++ on the tracked larger one-pop fixture",
        x=np.asarray(upstream["time_years"]),
        y=np.asarray(upstream["ne"]),
        color=COLORS["oracle"],
        label="upstream",
        xlim=xlim,
        ylim=ylim,
        xlabel="Generations before present",
        stats=[
            f"theta={float(upstream['theta']):.3e}",
            f"rho={float(upstream['rho']):.3e}",
            f"log corr={log_corr:.6f}",
            f"scale ratio={scale_ratio:.6f}",
            f"median rel err={median_rel:.6f}",
        ],
    )


def fig_dical2_dual() -> None:
    example = EXAMPLES["exp"]
    native = _run_smckit(example)
    oracle_loglik, oracle_params = _run_oracle(example)
    if oracle_params is None:
        raise RuntimeError("diCal2 oracle did not return parameter values")

    native_round = native["rounds"][-1]
    native_boundaries = np.asarray(native_round["epoch_boundaries"], dtype=float)
    native_sizes = np.asarray(native_round["pop_sizes"], dtype=float)
    native_growth = float(np.asarray(native_round["growth_rates"], dtype=float)[0])

    oracle_arr = np.asarray(oracle_params, dtype=float)
    oracle_growth = float(oracle_arr[0])
    oracle_boundaries = oracle_arr[1:3]
    oracle_sizes = oracle_arr[3:5]

    time_ylim = _linear_limits(native_boundaries, oracle_boundaries)
    size_ylim = _linear_limits(native_sizes, oracle_sizes)
    growth_ylim = _linear_limits(np.array([native_growth]), np.array([oracle_growth]))

    def _panel(output_name: str, title: str, boundaries: np.ndarray, sizes: np.ndarray, growth: float, ll: float | None, color: str) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.0), constrained_layout=True)
        axes[0].bar(["t1", "t2"], boundaries, color=color, alpha=0.9)
        axes[0].set_ylim(*time_ylim)
        axes[0].set_ylabel("Epoch boundary")
        axes[0].set_title(title)

        axes[1].bar(["N_shared", "N_anc"], sizes, color=color, alpha=0.9)
        axes[1].set_ylim(*size_ylim)
        axes[1].set_ylabel("Relative size")

        axes[2].bar(["growth_0"], [growth], color=color, alpha=0.9)
        axes[2].set_ylim(*growth_ylim)
        axes[2].set_ylabel("Growth")
        _add_stats(axes[2], [f"loglik={'n/a' if ll is None else f'{ll:.3f}'}", "example=README exp"])
        savefig(fig, output_name)

    _panel("dical2_native.png", "diCal2 README example", native_boundaries, native_sizes, native_growth, float(native["log_likelihood"]), COLORS["smckit"])
    _panel("dical2_upstream.png", "diCal2 README example", oracle_boundaries, oracle_sizes, oracle_growth, oracle_loglik, COLORS["oracle"])


def fig_asmc_dual() -> None:
    data_dir = ROOT / "vendor" / "ASMC" / "ASMC_data"
    example_root = data_dir / "examples" / "asmc" / "exampleFile.n300.array"
    dq_file = data_dir / "decoding_quantities" / "30-100-2000_CEU.decodingQuantities.gz"
    ref_means = np.loadtxt(data_dir / "testing" / "asmc" / "regression" / "regression.perPairPosteriorMeans.gz")
    ref_maps = np.loadtxt(data_dir / "testing" / "asmc" / "regression" / "regression.perPairMAP.gz").astype(np.int32)

    data = smckit.io.read_asmc(str(example_root), str(dq_file))
    native = asmc(
        data,
        pairs=[(1, 2), (2, 3), (3, 4)],
        mode="array",
        fold_data=True,
        store_per_pair_posterior_mean=True,
        store_per_pair_map=True,
        implementation="native",
    ).results["asmc"]

    x_mb = np.asarray(data.uns["physical_positions"], dtype=float) / 1e6
    native_mean = np.asarray(native["per_pair_posterior_means"][0], dtype=float)
    native_map = np.asarray(native["per_pair_maps"][0], dtype=float)
    upstream_mean = np.asarray(ref_means[0], dtype=float)
    upstream_map = np.asarray(ref_maps[0], dtype=float)
    xlim = _linear_limits(x_mb)
    mean_ylim = _positive_limits(native_mean, upstream_mean)
    map_ylim = _linear_limits(native_map, upstream_map)
    mean_corr = float(np.corrcoef(native_mean, upstream_mean)[0, 1])
    map_agree = float(np.mean(native_map == upstream_map))

    _asmc_panel(
        output_name="asmc_native.png",
        title="ASMC on the vendored n300 regression example",
        x_mb=x_mb,
        posterior=native_mean,
        maps=native_map,
        xlim=xlim,
        posterior_ylim=mean_ylim,
        map_ylim=map_ylim,
        color=COLORS["smckit"],
        label="native",
        stats=[f"posterior corr={mean_corr:.6f}", f"MAP agree={map_agree:.4f}"],
    )
    _asmc_panel(
        output_name="asmc_upstream.png",
        title="ASMC on the vendored n300 regression example",
        x_mb=x_mb,
        posterior=upstream_mean,
        maps=upstream_map,
        xlim=xlim,
        posterior_ylim=mean_ylim,
        map_ylim=map_ylim,
        color=COLORS["oracle"],
        label="upstream",
        stats=["source=regression fixture", f"MAP states={int(np.max(upstream_map) + 1)}"],
    )


def fig_msmc_im_dual() -> None:
    input_path = ROOT / "vendor" / "MSMC-IM" / "example" / "Yoruba_French.8haps.combined.msmc2.final.txt"
    ref_path = ROOT / "vendor" / "MSMC-IM" / "example" / "MSMC_IM_output" / "repro.b1_1e-08.b2_1e-06.MSMC_IM.estimates.txt"
    native = msmc_im(input_path, beta=(1e-8, 1e-6), implementation="native").results["msmc_im"]
    upstream = read_msmc_im_output(ref_path)

    xlim = _positive_limits(native["left_boundary"], upstream["left_boundary"])
    ne_ylim = _positive_limits(native["N1"], native["N2"], upstream["N1"], upstream["N2"])
    mig_ylim = _positive_limits(native["M"], upstream["M"])

    _pair_curve_panel(
        output_name="msmc_im_native.png",
        title="MSMC-IM on the Yoruba-French example",
        x=np.asarray(native["left_boundary"]),
        top_curves=[
            (np.asarray(native["N1"]), "N1 native", COLORS["smckit"]),
            (np.asarray(native["N2"]), "N2 native", COLORS["green"]),
        ],
        bottom_curves=[(np.asarray(native["M"]), "M native", COLORS["smckit"])],
        xlim=xlim,
        top_ylim=ne_ylim,
        bottom_ylim=mig_ylim,
        top_logy=True,
        bottom_logy=True,
        top_ylabel="Effective population size",
        bottom_ylabel="Cumulative migration",
        stats=[f"N1 corr={float(np.corrcoef(native['N1'], upstream['N1'])[0,1]):.6f}", f"N2 corr={float(np.corrcoef(native['N2'], upstream['N2'])[0,1]):.6f}"],
    )
    _pair_curve_panel(
        output_name="msmc_im_upstream.png",
        title="MSMC-IM on the Yoruba-French example",
        x=np.asarray(upstream["left_boundary"]),
        top_curves=[
            (np.asarray(upstream["N1"]), "N1 upstream", COLORS["oracle"]),
            (np.asarray(upstream["N2"]), "N2 upstream", COLORS["accent"]),
        ],
        bottom_curves=[(np.asarray(upstream["M"]), "M upstream", COLORS["oracle"])],
        xlim=xlim,
        top_ylim=ne_ylim,
        bottom_ylim=mig_ylim,
        top_logy=True,
        bottom_logy=True,
        top_ylabel="Effective population size",
        bottom_ylabel="Cumulative migration",
        stats=["source=MSMC_IM.py output"],
    )


def _esmc2_sequence(length: int = 2000, het_rate: float = 0.02, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=length, p=[1 - het_rate, het_rate]).astype(np.int8)


def _esmc2_data(seq: np.ndarray) -> SmcData:
    return SmcData(
        uns={
            "records": [{"codes": seq}],
            "sum_L": int((seq < 2).sum()),
            "sum_n": int((seq == 1).sum()),
        }
    )


def fig_esmc2_dual() -> None:
    validation = _load_module(ROOT / "tests" / "integration" / "test_esmc2_validation.py", "test_esmc2_validation")
    n = 10
    beta = 0.6
    sigma = 0.4
    rho = 3.0
    mu = 5e-4
    mu_b = 0.7
    L = 80_000
    xi = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.8, 1.0, 0.5], dtype=float)

    native_tc = esmc2_build_time_boundaries(n, beta, sigma)
    native_t = esmc2_expected_times(native_tc, xi, beta, sigma)
    native_q = esmc2_equilibrium_probs(native_tc, xi, beta, sigma)
    native_Q = esmc2_build_transition_matrix(native_tc, xi, native_t, beta, sigma, rho, L)

    ref_tc = validation.r_build_Tc(n, beta, sigma)
    ref_t = validation.r_expected_times(ref_tc, xi, beta, sigma)
    ref_q = validation.r_equilibrium_probs(ref_tc, xi, beta, sigma)
    ref_Q = validation.r_build_transition_matrix(ref_tc, xi, ref_t, beta, sigma, rho, L)

    xlim = (-0.1, n - 0.9)
    xi_ylim = _linear_limits(xi)
    tc_ylim = _positive_limits(native_tc[1:], ref_tc[1:])

    _esmc2_panel(
        output_name="esmc2_native.png",
        title="eSMC2 on a matched dormancy/selfing test case",
        tc=np.asarray(native_tc, dtype=float),
        xi=np.asarray(xi, dtype=float),
        tc_xlim=xlim,
        tc_ylim=tc_ylim,
        xi_xlim=xlim,
        xi_ylim=xi_ylim,
        color=COLORS["smckit"],
        label="native",
        stats=[f"q sum={float(np.sum(native_q)):.6f}", f"Q err={float(np.max(np.abs(native_Q - ref_Q))):.2e}"],
    )
    _esmc2_panel(
        output_name="esmc2_upstream.png",
        title="eSMC2 on a matched dormancy/selfing test case",
        tc=np.asarray(ref_tc, dtype=float),
        xi=np.asarray(xi, dtype=float),
        tc_xlim=xlim,
        tc_ylim=tc_ylim,
        xi_xlim=xlim,
        xi_ylim=xi_ylim,
        color=COLORS["oracle"],
        label="reference formulas",
        stats=[f"q sum={float(np.sum(ref_q)):.6f}", f"emit mu={mu:.1e}, mu_b={mu_b:.1f}"],
    )


def _run_ssm(window_size: int) -> dict:
    data = smckit.io.read_psmcfa(ROOT / "tests" / "data" / "NA12878_chr22.psmcfa")
    observations = [rec["codes"] for rec in data.uns["records"]]
    model = PsmcSSM(pattern=PATTERN)
    params = model.make_initial_params(
        data.uns["sum_L"],
        data.uns["sum_n"],
        max_t=15.0,
        tr_ratio=5.0,
        seed=42,
    )
    result = model.fit(observations, params, method="em", n_iterations=10)
    phys = model.to_physical_units(result.params, mu=MU, generation_time=GEN, window_size=window_size)
    return {
        "time_years": phys["time_years"],
        "ne": phys["ne"],
        "lambda": phys["lambda_k"],
        "log_likelihood": result.log_likelihood,
    }


def fig_ssm_dual() -> None:
    data = smckit.io.read_psmcfa(ROOT / "tests" / "data" / "NA12878_chr22.psmcfa")
    psmc_res = psmc(
        data,
        pattern=PATTERN,
        n_iterations=10,
        max_t=15.0,
        tr_ratio=5.0,
        mu=MU,
        generation_time=GEN,
        seed=42,
        implementation="native",
    ).results["psmc"]
    ssm_res = _run_ssm(data.window_size)
    xlim = _positive_limits(psmc_res["time_years"], ssm_res["time_years"])
    ylim = _positive_limits(psmc_res["ne"], ssm_res["ne"])

    _history_panel(
        output_name="ssm_native.png",
        title="PSMC-SSM on NA12878 chr22",
        x=np.asarray(ssm_res["time_years"]),
        y=np.asarray(ssm_res["ne"]),
        color=COLORS["green"],
        label="SSM native",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"loglik={float(ssm_res['log_likelihood']):.3f}", "task=NA12878 chr22"],
    )
    _history_panel(
        output_name="ssm_reference.png",
        title="PSMC-SSM on NA12878 chr22",
        x=np.asarray(psmc_res["time_years"]),
        y=np.asarray(psmc_res["ne"]),
        color=COLORS["smckit"],
        label="PSMC baseline",
        xlim=xlim,
        ylim=ylim,
        xlabel="Years before present",
        stats=[f"corr={float(np.corrcoef(ssm_res['lambda'], psmc_res['lambda'])[0,1]):.6f}", "reference=tl.psmc"],
    )


def main() -> None:
    apply_style()
    fig_psmc_dual()
    fig_asmc_dual()
    fig_msmc2_dual()
    fig_msmc_im_dual()
    fig_esmc2_dual()
    fig_smcpp_dual()
    fig_dical2_dual()
    fig_ssm_dual()


if __name__ == "__main__":
    main()
