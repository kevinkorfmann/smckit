"""Generate SMC++ gallery figures from the tracked larger one-pop fixture.

This avoids the optional ``msprime`` dependency used by the broader gallery
scripts and keeps the docs panels aligned with the tracked native/upstream
comparison that currently underwrites the SMC++ docs status.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smckit.io import read_smcpp_input
from smckit.tl import smcpp

OUTDIR = ROOT / "docs" / "gallery"
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "native": "#13678A",
    "upstream": "#D95D39",
    "grid": "#D1D5DB",
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


def _tracked_fixture():
    return read_smcpp_input(ROOT / "tests" / "data" / "smcpp_onepop_larger.smc")


def _positive_limits(*arrays: np.ndarray, log_pad: float = 0.06) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.extend(a[np.isfinite(a) & (a > 0)])
    vals_arr = np.asarray(vals, dtype=float)
    lo = float(vals_arr.min())
    hi = float(vals_arr.max())
    return (lo * (10 ** (-log_pad)), hi * (10**log_pad))


def _step_eval(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def _metrics(native: dict, upstream: dict) -> dict[str, float]:
    native_time = np.asarray(native["time"], dtype=float)
    upstream_time = np.asarray(upstream["time"], dtype=float)
    t_min = max(float(np.min(native_time[native_time > 0])), float(np.min(upstream_time[upstream_time > 0])))
    t_max = min(float(np.max(native_time)), float(np.max(upstream_time)))
    grid = np.geomspace(t_min, t_max, 400)
    native_ne = _step_eval(native_time, np.asarray(native["ne"], dtype=float), grid)
    upstream_ne = _step_eval(upstream_time, np.asarray(upstream["ne"], dtype=float), grid)
    rel = np.abs(native_ne - upstream_ne) / np.maximum(np.abs(upstream_ne), 1e-12)
    return {
        "log_corr": float(np.corrcoef(np.log(native_ne), np.log(upstream_ne))[0, 1]),
        "scale_ratio": float(np.median(native_ne) / np.median(upstream_ne)),
        "median_rel": float(np.median(rel)),
        "median_log10_error": float(np.median(np.abs(np.log10(native_ne / upstream_ne)))),
    }


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
    stats: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    ax.step(np.asarray(x, dtype=float), np.asarray(y, dtype=float), where="post", color=color, linewidth=2.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Generations before present")
    ax.set_ylabel("Effective population size")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.04,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )
    savefig(fig, output_name)


def main() -> None:
    apply_style()
    params = {
        "n_intervals": 4,
        "regularization": 10.0,
        "max_iterations": 2,
        "seed": 42,
        "generation_time": 1.0,
    }
    data = _tracked_fixture()
    native = smcpp(copy.deepcopy(data), implementation="native", **params).results["smcpp"]
    upstream = smcpp(copy.deepcopy(data), implementation="upstream", **params).results["smcpp"]
    metrics = _metrics(native, upstream)
    xlim = _positive_limits(np.asarray(native["time_years"]), np.asarray(upstream["time_years"]))
    ylim = _positive_limits(np.asarray(native["ne"]), np.asarray(upstream["ne"]))
    shared = [
        f"log corr={metrics['log_corr']:.6f}",
        f"scale ratio={metrics['scale_ratio']:.6f}",
        f"median rel={metrics['median_rel']:.6f}",
        f"median log10 err={metrics['median_log10_error']:.6f}",
    ]

    _history_panel(
        output_name="smcpp_native.png",
        title="SMC++ on the tracked larger one-pop fixture",
        x=np.asarray(native["time_years"]),
        y=np.asarray(native["ne"]),
        color=COLORS["native"],
        label="native",
        xlim=xlim,
        ylim=ylim,
        stats=shared + [f"loglik={float(native['log_likelihood']):.3f}"],
    )
    _history_panel(
        output_name="smcpp_upstream.png",
        title="SMC++ on the tracked larger one-pop fixture",
        x=np.asarray(upstream["time_years"]),
        y=np.asarray(upstream["ne"]),
        color=COLORS["upstream"],
        label="upstream",
        xlim=xlim,
        ylim=ylim,
        stats=shared + [f"loglik={float(upstream['log_likelihood']):.3f}"],
    )


if __name__ == "__main__":
    main()
