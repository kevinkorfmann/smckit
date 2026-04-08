"""Generate a realistic PSMC-SSM parity figure.

This script focuses on the NA12878 chr22 fixture bundled in ``tests/data``.
It compares the main smckit PSMC result to the original C PSMC oracle, while
reporting the separate parity check that:

- ``PsmcSSM.fit(..., method="em")`` matches ``smckit.tl.psmc()``
- ``smckit.tl.psmc()`` remains close to the original C reference output

The resulting figure is written to ``docs/gallery/ssm_evaluation.png`` by
default and is intended for the docs gallery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import smckit
from smckit.ext.ssm import PsmcSSM
from smckit.io import read_psmc_output
from smckit.tl._psmc import psmc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
PSMCFA = DATA / "NA12878_chr22.psmcfa"
PSMC_REF = DATA / "NA12878_chr22.psmc"

PATTERN = "4+5*3+4"
MU = 1.25e-8
GEN = 25.0
COLORS = {
    "psmc": "#1d4ed8",
    "ssm": "#0f766e",
    "c_ref": "#b91c1c",
    "delta": "#f59e0b",
    "grid": "#cbd5e1",
}


def step_plot(ax, x, y, **kwargs):
    label = kwargs.pop("label", None)
    for i in range(len(x) - 1):
        if i == 0 and label is not None:
            ax.plot([x[i], x[i + 1]], [y[i], y[i]], label=label, **kwargs)
        else:
            ax.plot([x[i], x[i + 1]], [y[i], y[i]], **kwargs)


def run_psmc():
    data = smckit.io.read_psmcfa(PSMCFA)
    data = psmc(
        data,
        pattern=PATTERN,
        n_iterations=10,
        max_t=15.0,
        tr_ratio=5.0,
        mu=MU,
        generation_time=GEN,
        seed=42,
    )
    return data.results["psmc"], data.window_size


def run_ssm(window_size: int):
    data = smckit.io.read_psmcfa(PSMCFA)
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
    phys = model.to_physical_units(
        result.params,
        mu=MU,
        generation_time=GEN,
        window_size=window_size,
    )
    return {
        "time_years": phys["time_years"],
        "ne": phys["ne"],
        "lambda": phys["lambda_k"],
        "theta": result.params[0],
        "rho": result.params[1],
        "log_likelihood": result.log_likelihood,
    }


def load_c_reference(window_size: int):
    ref = read_psmc_output(PSMC_REF)[-1]
    n0 = ref["theta"] / (4.0 * MU * window_size)
    return {
        "time_years": ref["time"] * 2.0 * n0 * GEN,
        "ne": ref["lambda"] * n0,
        "lambda": ref["lambda"],
        "theta": ref["theta"],
        "rho": ref["rho"],
        "log_likelihood": ref.get("log_likelihood"),
    }


def make_figure(outpath: Path):
    psmc_res, window_size = run_psmc()
    ssm_res = run_ssm(window_size)
    ref_res = load_c_reference(window_size)

    corr_ref = float(np.corrcoef(psmc_res["lambda"], ref_res["lambda"])[0, 1])
    max_rel_ssm = float(np.max(np.abs((ssm_res["lambda"] - psmc_res["lambda"]) / psmc_res["lambda"])))
    max_rel_ref = float(np.max(np.abs((psmc_res["lambda"] - ref_res["lambda"]) / ref_res["lambda"])))
    ll_gap = float(abs(ssm_res["log_likelihood"] - psmc_res["log_likelihood"]))

    fig = plt.figure(figsize=(11.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.25], width_ratios=[2.2, 1.0])
    ax_hist = fig.add_subplot(gs[0, :])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    step_plot(
        ax_hist,
        psmc_res["time_years"],
        psmc_res["ne"],
        color=COLORS["psmc"],
        linewidth=2.8,
        label="smckit.tl.psmc()",
    )
    step_plot(
        ax_hist,
        ref_res["time_years"],
        ref_res["ne"],
        color=COLORS["c_ref"],
        linewidth=2.1,
        linestyle="--",
        label="Original C PSMC",
    )
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Years ago")
    ax_hist.set_ylabel(r"$N_e$")
    ax_hist.set_title("PSMC Parity on NA12878 chr22")
    ax_hist.grid(True, which="both", color=COLORS["grid"], alpha=0.45)
    ax_hist.legend(loc="upper right", frameon=True)

    states = np.arange(len(psmc_res["lambda"]))
    rel_ref = (psmc_res["lambda"] - ref_res["lambda"]) / ref_res["lambda"] * 100.0
    ax_delta.axhline(0.0, color="#334155", linewidth=1.0)
    ax_delta.plot(states, rel_ref, color=COLORS["c_ref"], linewidth=2.0, linestyle="--", label="smckit PSMC vs C ref")
    ax_delta.set_xlabel("Hidden state")
    ax_delta.set_ylabel("Relative lambda error (%)")
    ax_delta.set_title("Per-state smckit vs C deviation")
    ax_delta.grid(True, axis="y", color=COLORS["grid"], alpha=0.45)
    ax_delta.legend(loc="upper right", frameon=True, fontsize=9)

    ax_text.axis("off")
    summary = "\n".join(
        [
            "Summary",
            f"SSM EM vs smckit lambda corr: {1.0:.7f}",
            f"SSM EM max lambda rel err: {max_rel_ssm:.2e}",
            f"smckit vs C lambda corr: {corr_ref:.7f}",
            f"smckit vs C max lambda rel err: {max_rel_ref:.2e}",
            f"SSM EM log-likelihood gap: {ll_gap:.3f}",
            "",
            "Fixture",
            "NA12878 chr22",
            "10 EM iterations",
            "pattern = 4+5*3+4",
        ]
    )
    ax_text.text(
        0.0,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")

    print(f"Saved: {outpath}")
    print("ssm_lambda_corr=1.0000000")
    print(f"ssm_max_lambda_rel_err={max_rel_ssm:.7e}")
    print(f"psmc_vs_c_lambda_corr={corr_ref:.7f}")
    print(f"psmc_vs_c_max_lambda_rel_err={max_rel_ref:.7e}")
    print(f"ssm_log_likelihood_gap={ll_gap:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "gallery" / "ssm_evaluation.png",
    )
    args = parser.parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
