"""Generate realistic documentation figures for the main smckit methods.

This script writes:

- docs/gallery/psmc_realistic.png
- docs/gallery/msmc2_realistic.png
- docs/gallery/esmc2_realistic.png
- docs/gallery/smcpp_realistic.png
- docs/gallery/asmc_realistic.png
- docs/gallery/msmc_im_realistic.png

The figures intentionally mix two sources of realism:

1. Fixture-backed oracle comparisons where the repository already ships a
   trusted reference output.
2. A shared msprime zigzag demography for the single-population methods so the
   docs also show how the tools behave on one coherent scenario.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import msprime
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import smckit
from smckit._core import SmcData
from smckit.backends._numba_esmc2 import (
    esmc2_build_time_boundaries,
    esmc2_build_transition_matrix,
    esmc2_expected_times,
)

OUTDIR = ROOT / "docs" / "gallery"
OUTDIR.mkdir(parents=True, exist_ok=True)

MU = 1.25e-8
R = 1.0e-8

COLORS = {
    "truth": "#111827",
    "smckit": "#13678A",
    "oracle": "#D95D39",
    "accent": "#F2A541",
    "green": "#4F772D",
    "purple": "#6B7280",
    "grid": "#D1D5DB",
    "mismatch": "#B91C1C",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFB",
            "axes.edgecolor": "#D1D5DB",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUTDIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step_plot(ax, times: np.ndarray, values: np.ndarray, **kwargs) -> None:
    ax.step(times, values, where="post", **kwargs)


def set_history_axes(ax, title: str, y_label: str = "Effective population size") -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generations before present")
    ax.set_ylabel(y_label)
    ax.set_title(title)


def simulate_zigzag_ts(
    sequence_length: int = 20_000_000,
    n_diploids: int = 4,
    seed: int = 20260406,
):
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=12000)
    demography.add_population_parameters_change(time=600, initial_size=5000, population="pop")
    demography.add_population_parameters_change(time=2200, initial_size=26000, population="pop")
    demography.add_population_parameters_change(time=9000, initial_size=7000, population="pop")
    demography.add_population_parameters_change(time=38000, initial_size=18000, population="pop")

    ts = msprime.sim_ancestry(
        samples={"pop": n_diploids},
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=R,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)

    true_times = np.array([1.0, 600.0, 2200.0, 9000.0, 38000.0, 200000.0])
    true_ne = np.array([12000.0, 5000.0, 26000.0, 7000.0, 18000.0, 18000.0])
    return ts, true_times, true_ne


def ts_to_psmcfa_data(ts, hap_pair: tuple[int, int] = (0, 1), window: int = 100) -> SmcData:
    het_positions = []
    for variant in ts.variants():
        geno = variant.genotypes
        if geno[hap_pair[0]] != geno[hap_pair[1]]:
            het_positions.append(int(variant.site.position))

    n_windows = int(ts.sequence_length) // window
    het_windows = np.zeros(n_windows, dtype=bool)
    if het_positions:
        idx = np.asarray(het_positions, dtype=np.int64) // window
        idx = idx[idx < n_windows]
        het_windows[idx] = True

    with tempfile.NamedTemporaryFile("w", suffix=".psmcfa", delete=False) as fh:
        fh.write(">sim\n")
        seq = np.where(het_windows, "K", "T")
        for start in range(0, len(seq), 60):
            fh.write("".join(seq[start : start + 60]) + "\n")
        path = fh.name

    try:
        return smckit.io.read_psmcfa(path)
    finally:
        os.unlink(path)


def ts_to_multihetsep_data(ts, n_haplotypes: int = 4) -> SmcData:
    with tempfile.NamedTemporaryFile("w", suffix=".multihetsep", delete=False) as fh:
        prev_pos = 0
        for variant in ts.variants():
            pos = int(variant.site.position) + 1
            geno = "".join(str(int(x)) for x in variant.genotypes[:n_haplotypes])
            fh.write(f"chr1\t{pos}\t{pos - prev_pos}\t{geno}\n")
            prev_pos = pos
        path = fh.name

    try:
        return smckit.io.read_multihetsep(path)
    finally:
        os.unlink(path)


def ts_to_smcpp_data(ts, n_haplotypes: int = 8) -> SmcData:
    observations = []
    prev_pos = 0

    for variant in ts.variants():
        geno = variant.genotypes[:n_haplotypes]
        if np.any(geno > 1):
            continue
        pos = int(variant.site.position)
        distinguished = int(geno[0])
        undist_derived = int(np.sum(geno[1:]))
        total = distinguished + undist_derived
        if total == 0 or total == n_haplotypes:
            continue

        gap = max(pos - prev_pos, 0)
        if gap > 0:
            observations.append((gap, 0, 0))
        observations.append((1, distinguished, undist_derived))
        prev_pos = pos + 1

    total_sites = int(ts.sequence_length)
    if prev_pos < total_sites:
        observations.append((total_sites - prev_pos, 0, 0))

    return SmcData(
        uns={
            "records": [{"name": "sim_chr1", "observations": observations}],
            "n_undist": n_haplotypes - 1,
        }
    )


def run_esmc2_demo():
    ts, true_times, true_ne = simulate_zigzag_ts(sequence_length=8_000_000)
    data = ts_to_psmcfa_data(ts)
    data = smckit.tl.esmc2(
        data,
        n_states=18,
        n_iterations=6,
        mu=MU,
        generation_time=1.0,
        rho_over_theta=R / MU,
    )
    return (true_times, true_ne), data.results["esmc2"]


def run_smcpp_demo():
    ts, true_times, true_ne = simulate_zigzag_ts(sequence_length=8_000_000)
    data = ts_to_smcpp_data(ts, n_haplotypes=8)
    data = smckit.tl.smcpp(
        data,
        n_intervals=8,
        max_iterations=10,
        mu=MU,
        recombination_rate=R,
        generation_time=1.0,
        regularization=5.0,
        seed=7,
        backend="native",
    )
    return (true_times, true_ne), data.results["smcpp"]


def fig_psmc_realistic() -> None:
    data = smckit.io.read_psmcfa(ROOT / "tests" / "data" / "NA12878_chr22.psmcfa")
    data = smckit.tl.psmc(
        data,
        pattern="4+5*3+4",
        n_iterations=10,
        tr_ratio=5.0,
        mu=MU,
        generation_time=25.0,
        seed=42,
    )
    ours = data.results["psmc"]

    ref_round = smckit.io.read_psmc_output(ROOT / "tests" / "data" / "NA12878_chr22.psmc")[-1]
    n0_ref = ref_round["theta"] / (4.0 * MU * data.window_size)
    ref_time = ref_round["time"] * 2.0 * n0_ref * 25.0
    ref_ne = ref_round["lambda"] * n0_ref

    fig, (ax_hist, ax_err) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        constrained_layout=True,
    )

    step_plot(
        ax_hist,
        ours["time_years"],
        ours["ne"],
        color=COLORS["smckit"],
        linewidth=2.4,
        label="smckit",
    )
    step_plot(
        ax_hist,
        ref_time,
        ref_ne,
        color=COLORS["oracle"],
        linewidth=2.0,
        linestyle="--",
        label="C reference",
    )
    set_history_axes(ax_hist, "PSMC on NA12878 chr22")
    ax_hist.legend(loc="upper right")

    rel_err = (ours["lambda"] - ref_round["lambda"]) / ref_round["lambda"]
    ax_err.axhline(0.0, color=COLORS["truth"], linewidth=1.0)
    ax_err.bar(np.arange(len(rel_err)), rel_err * 100.0, color=COLORS["accent"], width=0.85)
    ax_err.set_xlabel("Hidden state")
    ax_err.set_ylabel("Relative error (%)")
    ax_err.set_title("Per-state lambda deviation from the original PSMC output")

    savefig(fig, "psmc_realistic.png")


def _load_msmc_validation_module():
    path = ROOT / "tests" / "integration" / "test_msmc_validation.py"
    return _load_module(path, "test_msmc_validation")


def fig_msmc2_realistic() -> None:
    validation = _load_msmc_validation_module()
    data = smckit.io.read_multihetsep(ROOT / "data" / "msmc2_test.multihetsep")
    data = smckit.tl.msmc2(
        data,
        n_iterations=2,
        mu=MU,
        generation_time=25.0,
    )
    ours = data.results["msmc2"]

    ref_left = validation.EXPECTED_LEFT_ALL_1
    ref_lambda = validation.EXPECTED_LAMBDA_ALL_2
    ref_time = ref_left / MU * 25.0
    ref_ne = 1.0 / (2.0 * MU * ref_lambda)

    fig, (ax_hist, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        constrained_layout=True,
    )

    step_plot(
        ax_hist,
        ours["time_years"],
        ours["ne"],
        color=COLORS["smckit"],
        linewidth=2.3,
        label="smckit",
    )
    step_plot(
        ax_hist,
        ref_time,
        ref_ne,
        color=COLORS["oracle"],
        linewidth=2.0,
        linestyle="--",
        label="upstream fixture",
    )
    set_history_axes(ax_hist, "MSMC2 on the bundled multihetsep fixture")
    ax_hist.legend(loc="upper right")

    delta = ours["lambda"] - ref_lambda
    ax_delta.axhline(0.0, color=COLORS["truth"], linewidth=1.0)
    ax_delta.plot(np.arange(len(delta)), delta, color=COLORS["accent"], linewidth=1.8)
    ax_delta.fill_between(np.arange(len(delta)), 0.0, delta, color=COLORS["accent"], alpha=0.2)
    ax_delta.set_xlabel("Time segment")
    ax_delta.set_ylabel("Lambda delta")
    ax_delta.set_title("Residual coalescence-rate difference")

    savefig(fig, "msmc2_realistic.png")


def fig_esmc2_realistic() -> None:
    (true_times, true_ne), res = run_esmc2_demo()

    n_states = 28
    xi = np.ones(n_states, dtype=np.float64)
    rho = 3500.0
    length = 5_000_000
    configs = [
        (1.0, 0.0, "Standard", None),
        (0.45, 0.65, "Seed bank + selfing", None),
    ]
    mats = []
    for beta, sigma, _, _ in configs:
        tc = esmc2_build_time_boundaries(n_states, beta, sigma)
        t = esmc2_expected_times(tc, xi, beta, sigma)
        mats.append(esmc2_build_transition_matrix(tc, xi, t, beta, sigma, rho, length))

    fig = plt.figure(figsize=(12.0, 9.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0])

    ax_hist = fig.add_subplot(gs[0, 0])
    step_plot(ax_hist, true_times, true_ne, color=COLORS["truth"], linewidth=2.2, label="truth")
    step_plot(
        ax_hist,
        res["time_years"],
        res["ne"],
        color=COLORS["smckit"],
        linewidth=2.2,
        label="eSMC2",
    )
    set_history_axes(ax_hist, "eSMC2 on a shared zigzag demography")
    ax_hist.legend(loc="upper right")

    vmax = max(np.nanpercentile(np.log10(np.clip(m[np.triu_indices_from(m, 1)], 1e-15, None)), 99) for m in mats)
    vmin = min(np.nanpercentile(np.log10(np.clip(m[np.triu_indices_from(m, 1)], 1e-15, None)), 1) for m in mats)

    heat_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    last_im = None
    for ax, mat, (beta, sigma, title, _) in zip(heat_axes, mats, configs):
        q = mat[:24, :24].copy()
        np.fill_diagonal(q, np.nan)
        q = np.log10(np.clip(q, 1e-15, None))
        last_im = ax.imshow(q, cmap="cividis", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} transition landscape")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.grid(False)
        ax.text(
            0.03,
            0.97,
            f"beta={beta:.2f}, sigma={sigma:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax_escape = fig.add_subplot(gs[1, 1])
    for mat, (beta, sigma, title, _) in zip(mats, configs):
        escape = 1.0 - np.diag(mat)
        ax_escape.plot(
            np.arange(len(escape)),
            escape,
            linewidth=2.0,
            label=f"{title} (beta={beta:.2f}, sigma={sigma:.2f})",
        )
    ax_escape.set_yscale("log")
    ax_escape.set_xlabel("Hidden state")
    ax_escape.set_ylabel("State-change probability")
    ax_escape.set_title("Escape from the diagonal by time state")
    ax_escape.legend(loc="upper right", fontsize=9)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=heat_axes, shrink=0.82, pad=0.02)
        cbar.set_label("log10 transition probability")

    savefig(fig, "esmc2_realistic.png")


def fig_smcpp_realistic() -> None:
    (true_times, true_ne), res = run_smcpp_demo()

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    step_plot(ax, true_times, true_ne, color=COLORS["truth"], linewidth=2.2, label="truth")
    step_plot(ax, res["time_years"], res["ne"], color=COLORS["smckit"], linewidth=2.3, label="SMC++")
    set_history_axes(ax, "SMC++ on the shared zigzag demography")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.04,
        (
            f"n_undist={res['n_undist']}, theta={res['theta']:.3e}, "
            f"rho={res['rho']:.3e}, iterations={res['optimization']['n_iterations']}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )

    savefig(fig, "smcpp_realistic.png")


def fig_asmc_realistic() -> None:
    data_dir = ROOT / "vendor" / "ASMC" / "ASMC_data"
    example_root = data_dir / "examples" / "asmc" / "exampleFile.n300.array"
    dq_file = data_dir / "decoding_quantities" / "30-100-2000_CEU.decodingQuantities.gz"
    ref_means = np.loadtxt(data_dir / "testing" / "asmc" / "regression" / "regression.perPairPosteriorMeans.gz")
    ref_maps = np.loadtxt(data_dir / "testing" / "asmc" / "regression" / "regression.perPairMAP.gz").astype(np.int32)

    data = smckit.io.read_asmc(str(example_root), str(dq_file))
    data = smckit.tl.asmc(
        data,
        pairs=[(1, 2), (2, 3), (3, 4)],
        mode="array",
        fold_data=True,
        store_per_pair_posterior_mean=True,
        store_per_pair_map=True,
    )
    res = data.results["asmc"]
    x_mb = data.uns["physical_positions"] / 1e6

    fig = plt.figure(figsize=(11.5, 8.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.6, 1.2])

    ax_mean = fig.add_subplot(gs[0, 0])
    ax_mean.plot(
        x_mb,
        ref_means[0],
        color=COLORS["oracle"],
        linewidth=1.8,
        linestyle="--",
        label="upstream posterior mean",
    )
    ax_mean.plot(
        x_mb,
        np.asarray(res["per_pair_posterior_means"][0]),
        color=COLORS["smckit"],
        linewidth=1.6,
        label="smckit posterior mean",
    )
    ax_mean.set_yscale("log")
    ax_mean.set_xlabel("Physical position (Mb)")
    ax_mean.set_ylabel("Posterior mean TMRCA")
    ax_mean.set_title("ASMC on the vendored n300 array regression example")
    ax_mean.legend(loc="upper right")

    ax_map = fig.add_subplot(gs[1, 0])
    mismatch = (np.asarray(res["per_pair_maps"]) != ref_maps).astype(float)
    im = ax_map.imshow(
        mismatch,
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
        extent=[x_mb.min(), x_mb.max(), 3.5, 0.5],
        vmin=0.0,
        vmax=1.0,
    )
    ax_map.set_xlabel("Physical position (Mb)")
    ax_map.set_ylabel("Decoded pair")
    ax_map.set_yticks([1, 2, 3])
    ax_map.set_title("MAP state-index disagreements with the upstream regression output")
    ax_map.grid(False)
    cbar = fig.colorbar(im, ax=ax_map, shrink=0.88, pad=0.02)
    cbar.set_label("Mismatch")

    savefig(fig, "asmc_realistic.png")


def fig_msmc_im_realistic() -> None:
    input_path = ROOT / "vendor" / "MSMC-IM" / "example" / "Yoruba_French.8haps.combined.msmc2.final.txt"
    ref_path = ROOT / "vendor" / "MSMC-IM" / "example" / "MSMC_IM_output" / "repro.b1_1e-08.b2_1e-06.MSMC_IM.estimates.txt"

    ours = smckit.tl.msmc_im(input_path, beta=(1e-8, 1e-6)).results["msmc_im"]
    ref = smckit.io.read_msmc_im_output(ref_path)

    fig, (ax_ne, ax_mig) = plt.subplots(2, 1, figsize=(10.8, 8.2), constrained_layout=True)

    for arr, label, color in [
        (ref["N1"], "Pop 1 upstream", COLORS["oracle"]),
        (ref["N2"], "Pop 2 upstream", COLORS["accent"]),
    ]:
        step_plot(ax_ne, ref["left_boundary"], arr, color=color, linewidth=2.0, linestyle="--", label=label)
    step_plot(ax_ne, ours["left_boundary"], ours["N1"], color=COLORS["smckit"], linewidth=2.1, label="Pop 1 smckit")
    step_plot(ax_ne, ours["left_boundary"], ours["N2"], color=COLORS["green"], linewidth=2.1, label="Pop 2 smckit")
    ax_ne.set_xscale("log")
    ax_ne.set_yscale("log")
    ax_ne.set_xlabel("Generations before present")
    ax_ne.set_ylabel("Effective population size")
    ax_ne.set_title("MSMC-IM on the Yoruba-French 8-haplotype example")
    ax_ne.legend(loc="upper right", ncol=2)

    step_plot(ax_mig, ref["left_boundary"], ref["M"], color=COLORS["oracle"], linewidth=2.0, linestyle="--", label="Upstream M(t)")
    step_plot(ax_mig, ours["left_boundary"], ours["M"], color=COLORS["smckit"], linewidth=2.2, label="smckit M(t)")
    ax_mig.set_xscale("log")
    ax_mig.set_xlabel("Generations before present")
    ax_mig.set_ylabel("Cumulative migration")
    ax_mig.set_title("Integrated migration history")
    ax_mig.legend(loc="upper left")

    savefig(fig, "msmc_im_realistic.png")


def main() -> None:
    apply_style()
    fig_psmc_realistic()
    fig_msmc2_realistic()
    fig_esmc2_realistic()
    fig_smcpp_realistic()
    fig_asmc_realistic()
    fig_msmc_im_realistic()


if __name__ == "__main__":
    main()
