#!/usr/bin/env python3
"""Generate scoped eSMC2 parity figures for the documentation gallery.

The raw native/upstream gallery pair was misleading because it showed a single
formula-style case instead of the oracle-backed fit matrix that now drives
parity decisions. This script replaces that with:

1. ``docs/gallery/esmc2_parity_demography.png``:
   Demographic overlays for the tracked native/upstream fit cases.
2. ``docs/gallery/esmc2_parity_transition.png``:
   Transition-matrix agreement for representative challenging fit branches.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".smckit-cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smckit._core import SmcData  # noqa: E402
from smckit.backends._numba_esmc2 import esmc2_build_hmm  # noqa: E402
from smckit.tl._esmc2 import esmc2  # noqa: E402

OUTDIR = ROOT / "docs" / "gallery"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 800
HET_RATE = 0.01
SEED = 7
MU_PER_GEN = 1e-8
GENERATION_TIME = 1.0
MU_B = 1.0

COLORS = {
    "native": "#13678A",
    "upstream": "#D95D39",
    "accent": "#F2A541",
    "grid": "#D1D5DB",
    "text": "#111827",
}

CASES = [
    {
        "slug": "fixed_rho",
        "title": "Fixed rho",
        "subtitle": "1 iteration",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 1,
            "estimate_rho": False,
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
    {
        "slug": "redo_rho",
        "title": "Rho Redo",
        "subtitle": "2 iterations",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 2,
            "estimate_rho": True,
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
    {
        "slug": "beta",
        "title": "Beta Fit",
        "subtitle": "fixed rho",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 1,
            "estimate_beta": True,
            "estimate_sigma": False,
            "estimate_rho": False,
            "beta": 0.3,
            "sigma": 0.0,
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
    {
        "slug": "sigma",
        "title": "Sigma Fit",
        "subtitle": "fixed rho",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 1,
            "estimate_beta": False,
            "estimate_sigma": True,
            "estimate_rho": False,
            "beta": 0.5,
            "sigma": 0.2,
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
    {
        "slug": "beta_sigma",
        "title": "Beta + Sigma",
        "subtitle": "fixed rho",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 1,
            "estimate_beta": True,
            "estimate_sigma": True,
            "estimate_rho": False,
            "beta": 0.3,
            "sigma": 0.2,
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
    {
        "slug": "grouped_beta",
        "title": "Grouped Xi",
        "subtitle": "pop_vect=[3,3]",
        "kwargs": {
            "n_states": 6,
            "n_iterations": 1,
            "estimate_beta": True,
            "estimate_sigma": False,
            "estimate_rho": False,
            "beta": 0.3,
            "sigma": 0.0,
            "pop_vect": [3, 3],
            "mu": MU_PER_GEN,
            "generation_time": GENERATION_TIME,
        },
    },
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFB",
            "axes.edgecolor": COLORS["grid"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _sequence() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([0, 1], size=SEQUENCE_LENGTH, p=[1.0 - HET_RATE, HET_RATE]).astype(np.int8)


def _smcdata_from_sequence(seq: np.ndarray) -> SmcData:
    return SmcData(
        uns={
            "records": [{"codes": seq}],
            "sum_L": int((seq < 2).sum()),
            "sum_n": int((seq == 1).sum()),
        }
    )


def _max_rel_error(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left_arr - right_arr) / np.maximum(np.abs(right_arr), 1e-12)))


def _history_x(times: np.ndarray) -> np.ndarray:
    x = np.asarray(times, dtype=np.float64).copy()
    positive = x[x > 0.0]
    if positive.size == 0:
        return x
    x[x <= 0.0] = positive.min() * 0.5
    return x


def _run_case(seq: np.ndarray, case: dict) -> dict[str, object]:
    data = _smcdata_from_sequence(seq)
    native = esmc2(
        copy.deepcopy(data),
        implementation="native",
        **case["kwargs"],
    ).results["esmc2"]
    upstream = esmc2(
        copy.deepcopy(data),
        implementation="upstream",
        upstream_options={"capture_final_sufficient_statistics": True},
        **case["kwargs"],
    ).results["esmc2"]
    return {
        "case": case,
        "native": native,
        "upstream": upstream,
        "tc_rel_pct": 100.0 * _max_rel_error(native["Tc"], upstream["Tc"]),
        "xi_rel_pct": 100.0 * _max_rel_error(native["Xi"], upstream["Xi"]),
        "final_ll_rel_pct": 100.0
        * (
            abs(
                float(native["log_likelihood"])
                - float(upstream["upstream"]["final_sufficient_statistics"]["log_likelihood"])
            )
            / max(abs(float(native["log_likelihood"])), 1e-12)
        ),
    }


def _build_q(result: dict) -> np.ndarray:
    Q, _, _, _, _ = esmc2_build_hmm(
        len(result["Xi"]),
        np.asarray(result["Xi"], dtype=np.float64),
        float(result["beta"]),
        float(result["sigma"]),
        float(result["rho_per_sequence"]),
        float(result["mu"]),
        MU_B,
        SEQUENCE_LENGTH,
    )
    return np.asarray(Q, dtype=np.float64)


def _summary_lines(result: dict[str, object]) -> list[str]:
    native = result["native"]
    upstream = result["upstream"]
    lines = [
        f"max Tc err {result['tc_rel_pct']:.3f}%",
        f"max Xi err {result['xi_rel_pct']:.3f}%",
        f"final LL err {result['final_ll_rel_pct']:.4f}%",
    ]
    beta_delta = abs(float(native["beta"]) - float(upstream["beta"]))
    sigma_delta = abs(float(native["sigma"]) - float(upstream["sigma"]))
    if beta_delta > 0.0:
        lines.append(f"Δbeta {beta_delta * 100:.3f} pp")
    if sigma_delta > 0.0:
        lines.append(f"Δsigma {sigma_delta * 100:.3f} pp")
    return lines


def plot_demography(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, result in zip(axes.flat, results, strict=True):
        case = result["case"]
        native = result["native"]
        upstream = result["upstream"]

        x_native = _history_x(native["time_years"])
        x_upstream = _history_x(upstream["time_years"])
        y_native = np.asarray(native["ne"], dtype=np.float64)
        y_upstream = np.asarray(upstream["ne"], dtype=np.float64)

        ax.step(
            x_upstream,
            y_upstream,
            where="post",
            color=COLORS["upstream"],
            linewidth=2.2,
            label="upstream",
        )
        ax.step(
            x_native,
            y_native,
            where="post",
            color=COLORS["native"],
            linewidth=1.8,
            linestyle="--",
            label="native",
        )
        ax.scatter(x_upstream, y_upstream, s=16, color=COLORS["upstream"], zorder=3)
        ax.scatter(x_native, y_native, s=12, color=COLORS["native"], zorder=3)

        positive_x = np.concatenate([x_native[x_native > 0.0], x_upstream[x_upstream > 0.0]])
        positive_y = np.concatenate([y_native[y_native > 0.0], y_upstream[y_upstream > 0.0]])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(positive_x.min() * 0.8, positive_x.max() * 1.25)
        ax.set_ylim(positive_y.min() * 0.8, positive_y.max() * 1.25)
        ax.set_title(f"{case['title']}\n{case['subtitle']}")
        ax.set_xlabel("Generations before present")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Effective population size")
        ax.text(
            0.03,
            0.03,
            "\n".join(_summary_lines(result)),
            transform=ax.transAxes,
            fontsize=8,
            color=COLORS["text"],
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.9},
        )

    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "eSMC2 Tracked Fit Parity on the Clean 800 bp Oracle Fixture",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        (
            "Six enforced branches: fixed rho, rho redo, beta, sigma, "
            "beta+sigma, and grouped Xi with pop_vect=[3,3]."
        ),
        ha="center",
        fontsize=9,
        color=COLORS["text"],
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = OUTDIR / "esmc2_parity_demography.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def plot_transition_agreement(results_by_slug: dict[str, dict[str, object]]) -> None:
    transition_cases = [
        ("beta_sigma", "Beta + Sigma"),
        ("grouped_beta", "Grouped Xi [3,3]"),
    ]
    q_pairs = []
    log_vals: list[np.ndarray] = []
    max_abs_bp = 0.0
    for slug, _ in transition_cases:
        native_Q = _build_q(results_by_slug[slug]["native"])
        upstream_Q = _build_q(results_by_slug[slug]["upstream"])
        q_pairs.append((slug, native_Q, upstream_Q))
        log_vals.append(np.log10(np.maximum(native_Q, 1e-15)))
        log_vals.append(np.log10(np.maximum(upstream_Q, 1e-15)))
        max_abs_bp = max(max_abs_bp, float(np.max(np.abs((native_Q - upstream_Q) * 1e4))))

    vmin = min(float(np.min(arr)) for arr in log_vals)
    vmax = max(float(np.max(arr)) for arr in log_vals)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for row, (slug, label) in enumerate(transition_cases):
        native_Q = q_pairs[row][1]
        upstream_Q = q_pairs[row][2]
        diff_bp = (native_Q - upstream_Q) * 1e4
        stable = np.abs(upstream_Q) > 1e-10
        max_rel_pct = 100.0 * float(
            np.max(
                np.abs((native_Q[stable] - upstream_Q[stable]) / upstream_Q[stable])
            )
        )

        for col, (matrix, title) in enumerate(
            (
                (np.log10(np.maximum(native_Q, 1e-15)), "native log10(Q)"),
                (np.log10(np.maximum(upstream_Q, 1e-15)), "upstream log10(Q)"),
                (diff_bp, "delta (native-upstream), bp"),
            )
        ):
            ax = axes[row, col]
            if col < 2:
                ax.imshow(
                    matrix,
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                ax.imshow(
                    matrix,
                    origin="lower",
                    aspect="equal",
                    cmap="coolwarm",
                    vmin=-max_abs_bp,
                    vmax=max_abs_bp,
                )
                ax.text(
                    0.03,
                    0.03,
                    (
                        f"max |ΔQ| {float(np.max(np.abs(diff_bp))):.2f} bp\n"
                        f"max rel {max_rel_pct:.3f}%"
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    color=COLORS["text"],
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.9},
                )
            ax.set_title(f"{label}\n{title}")
            ax.set_xlabel("Current state")
            if col == 0:
                ax.set_ylabel("Next state")

    q_bar = fig.colorbar(
        axes[0, 0].images[0],
        ax=axes[:, :2],
        shrink=0.9,
        label="log10(Q)",
    )
    q_bar.outline.set_edgecolor(COLORS["grid"])
    diff_bar = fig.colorbar(
        axes[0, 2].images[0],
        ax=axes[:, 2],
        shrink=0.9,
        label="difference (bp)",
    )
    diff_bar.outline.set_edgecolor(COLORS["grid"])

    fig.suptitle(
        "eSMC2 Transition-Matrix Agreement on Representative Challenging Branches",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.92,
        bottom=0.08,
        top=0.90,
        wspace=0.30,
        hspace=0.34,
    )
    path = OUTDIR / "esmc2_parity_transition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def main() -> None:
    apply_style()
    if "SMCKIT_ESMC2_RSCRIPT" not in os.environ:
        rscript = shutil.which("Rscript")
        if rscript is not None:
            os.environ["SMCKIT_ESMC2_RSCRIPT"] = rscript

    seq = _sequence()
    results = [_run_case(seq, case) for case in CASES]
    results_by_slug = {result["case"]["slug"]: result for result in results}
    plot_demography(results)
    plot_transition_agreement(results_by_slug)


if __name__ == "__main__":
    main()
