"""Publication-oriented ASMC posterior visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from smckit._core import SmcData


def _asmc_plot_data(
    data: SmcData | dict[str, Any],
    genotype: str | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if isinstance(data, SmcData):
        if "asmc" not in data.results:
            raise ValueError("SmcData does not contain an ASMC result.")
        result = data.results["asmc"]
        start, stop = result.get(
            "site_slice",
            (0, np.asarray(result["sum_of_posteriors"]).shape[0]),
        )
        positions = np.asarray(
            data.uns.get(
                "physical_positions",
                np.arange(np.asarray(result["sum_of_posteriors"]).shape[0]),
            )
        )[start:stop]
    else:
        result = data
        positions = np.arange(
            np.asarray(result["sum_of_posteriors"]).shape[0],
            dtype=np.float64,
        )
    if genotype is None:
        posterior = np.asarray(result["sum_of_posteriors"], dtype=np.float64)
    else:
        if genotype not in {"00", "01", "11"}:
            raise ValueError("genotype must be one of: 00, 01, 11")
        try:
            posterior = np.asarray(
                result["sum_of_posteriors_major_minor"][genotype],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ValueError("ASMC result does not contain major/minor posterior sums.") from exc
    return result, positions, posterior


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).copy()
    totals = normalized.sum(axis=1, keepdims=True)
    np.divide(normalized, totals, out=normalized, where=totals != 0)
    return normalized


def asmc_posterior_heatmap(
    data: SmcData | dict[str, Any],
    *,
    genotype: str | None = None,
    normalize: bool = True,
    max_time: float | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> Axes:
    """Plot ASMC coalescence density across genomic position and time.

    The heatmap uses a perceptually uniform colormap and a logarithmic time
    axis. Values are normalized within sites by default.
    """
    result, positions, posterior = _asmc_plot_data(data, genotype)
    if normalize:
        posterior = _normalize_rows(posterior)
    expected_times = np.asarray(result["expected_times"], dtype=np.float64)
    if posterior.shape[1] != expected_times.size:
        raise ValueError("ASMC posterior state count does not match expected_times.")
    keep = np.ones(expected_times.size, dtype=bool)
    if max_time is not None:
        if max_time <= 0:
            raise ValueError("max_time must be positive")
        keep = expected_times <= max_time
        if not np.any(keep):
            raise ValueError("max_time excludes every ASMC state.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 2.7))
    mesh = ax.pcolormesh(
        positions,
        expected_times[keep],
        posterior[:, keep].T,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel("Coalescence time (generations)")
    colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Posterior probability" if normalize else "Posterior sum over pairs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def asmc_recent_coalescence_density(
    data: SmcData | dict[str, Any],
    *,
    max_time: float,
    genotype: str | None = None,
    window_sites: int = 100,
    step_sites: int | None = None,
    ax: Axes | None = None,
    color: str = "#0072B2",
) -> Axes:
    """Plot the density of recent coalescence (DRC) in marker windows."""
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    if window_sites < 1:
        raise ValueError("window_sites must be at least 1")
    if step_sites is None:
        step_sites = max(1, window_sites // 2)
    if step_sites < 1:
        raise ValueError("step_sites must be at least 1")

    result, positions, posterior = _asmc_plot_data(data, genotype)
    posterior = _normalize_rows(posterior)
    expected_times = np.asarray(result["expected_times"], dtype=np.float64)
    recent = posterior[:, expected_times <= max_time].sum(axis=1)
    if recent.size < window_sites:
        raise ValueError("window_sites exceeds the number of ASMC sites.")

    starts = np.arange(0, recent.size - window_sites + 1, step_sites)
    centers = np.asarray([np.mean(positions[start : start + window_sites]) for start in starts])
    density = np.asarray([np.mean(recent[start : start + window_sites]) for start in starts])

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(centers, density, color=color, linewidth=1.4)
    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel(f"DRC (≤{max_time:g} generations)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save_asmc_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 600,
) -> Path:
    """Export an ASMC figure as vector art or publication-resolution raster."""
    path = Path(path)
    if path.suffix.lower() not in {".pdf", ".svg", ".eps", ".png", ".tif", ".tiff"}:
        raise ValueError("Use PDF, SVG, EPS, PNG, or TIFF for ASMC figures.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    return path


__all__ = [
    "asmc_posterior_heatmap",
    "asmc_recent_coalescence_density",
    "save_asmc_figure",
]
