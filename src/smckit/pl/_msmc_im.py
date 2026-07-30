"""Publication-oriented MSMC-IM diagnostic visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from smckit._core import SmcData


def _msmc_im_result(data: SmcData | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, SmcData):
        if "msmc_im" not in data.results:
            raise ValueError("SmcData does not contain an MSMC-IM result.")
        return data.results["msmc_im"]
    return data


def _validated_series(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(result["left_boundary"], dtype=np.float64)
    n1 = np.asarray(result["N1"], dtype=np.float64)
    n2 = np.asarray(result["N2"], dtype=np.float64)
    migration = np.asarray(result.get("m_thresholded", result["m"]), dtype=np.float64)
    cumulative = np.asarray(result["M"], dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("MSMC-IM left boundaries must be a strictly increasing vector.")
    if any(values.shape != time.shape for values in (n1, n2, migration, cumulative)):
        raise ValueError("MSMC-IM diagnostic series must match the time-axis shape.")
    if not all(np.all(np.isfinite(values)) for values in (time, n1, n2, migration, cumulative)):
        raise ValueError("MSMC-IM diagnostic series must be finite.")
    if np.any(time <= 0) or np.any(n1 <= 0) or np.any(n2 <= 0):
        raise ValueError("MSMC-IM time and population sizes must be positive.")
    return time, n1, n2, migration, cumulative


def msmc_im_summary(
    data: SmcData | dict[str, Any],
    *,
    axes: tuple[Axes, Axes, Axes] | None = None,
    population_labels: tuple[str, str] = ("Population 1", "Population 2"),
) -> tuple[Axes, Axes, Axes]:
    """Plot fitted sizes, instantaneous migration, and cumulative migration.

    The three aligned panels keep quantities with incompatible units on
    separate axes. Split-time quantiles are marked on the cumulative panel.
    """
    result = _msmc_im_result(data)
    time, n1, n2, migration, cumulative = _validated_series(result)
    if len(population_labels) != 2:
        raise ValueError("population_labels must contain exactly two labels.")
    if axes is None:
        _, created_axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(7.2, 6.4),
            gridspec_kw={"height_ratios": (1.2, 1.0, 1.0), "hspace": 0.12},
        )
        axes = tuple(created_axes)
    if len(axes) != 3:
        raise ValueError("axes must contain three matplotlib axes.")
    size_ax, migration_ax, cumulative_ax = axes

    size_ax.step(
        time,
        n1,
        where="post",
        color="#0072B2",
        linewidth=1.7,
        label=population_labels[0],
    )
    size_ax.step(
        time,
        n2,
        where="post",
        color="#D55E00",
        linewidth=1.7,
        label=population_labels[1],
    )
    size_ax.set_yscale("log")
    size_ax.set_ylabel(r"$N_e$")
    size_ax.legend(frameon=False, ncol=2, loc="best")

    positive_migration = np.where(migration > 0, migration, np.nan)
    migration_ax.step(
        time,
        positive_migration,
        where="post",
        color="#009E73",
        linewidth=1.7,
    )
    if np.any(np.isfinite(positive_migration)):
        migration_ax.set_yscale("log")
    migration_ax.set_ylabel(r"$m(t)$")

    cumulative_ax.step(
        time,
        cumulative,
        where="post",
        color="#CC79A7",
        linewidth=1.7,
    )
    quantiles = result.get("split_time_quantiles", {})
    markers = {0.25: "o", 0.5: "s", 0.75: "^"}
    for quantile, marker in markers.items():
        split_time = quantiles.get(quantile, quantiles.get(str(quantile)))
        if split_time is None:
            continue
        cumulative_ax.scatter(
            [float(split_time)],
            [quantile],
            marker=marker,
            s=28,
            facecolor="white",
            edgecolor="#332288",
            linewidth=1.1,
            zorder=3,
            label=f"{quantile:g} quantile",
        )
    cumulative_ax.set_ylim(0, max(1.0, float(np.max(cumulative)) * 1.05))
    cumulative_ax.set_ylabel(r"$M(t)$")
    cumulative_ax.set_xlabel("Time (generations)")
    cumulative_ax.set_xscale("log")
    if cumulative_ax.collections:
        cumulative_ax.legend(frameon=False, ncol=3, fontsize=8, loc="best")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.6)
    return axes


def save_msmc_im_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 600,
) -> Path:
    """Export an MSMC-IM figure as vector art or publication-resolution raster."""
    path = Path(path)
    if path.suffix.lower() not in {".pdf", ".svg", ".eps", ".png", ".tif", ".tiff"}:
        raise ValueError("Use PDF, SVG, EPS, PNG, or TIFF for MSMC-IM figures.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


__all__ = ["msmc_im_summary", "save_msmc_im_figure"]
