"""Publication-oriented plots for normalized PHLASH posterior results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from smckit._core import SmcData


def _payload(data: SmcData | dict[str, Any]) -> dict[str, Any]:
    if hasattr(data, "results"):
        return data.results["phlash"]
    return data


def phlash_demographic_history(
    data: SmcData | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
    color: str = "#0072B2",
    label: str = "PHLASH median",
    show_interval: bool = True,
    posterior_samples: int = 0,
    sample_alpha: float = 0.08,
    log_x: bool = True,
    log_y: bool = True,
) -> matplotlib.axes.Axes:
    """Plot the PHLASH posterior median and credible interval."""
    import matplotlib.pyplot as plt

    result = _payload(data)
    time = np.asarray(result["time"], dtype=float)
    ne = np.asarray(result["ne"], dtype=float)
    posterior = np.asarray(result["posterior_ne"], dtype=float)
    interval = result["credible_interval"]
    lower = np.asarray(interval["lower"], dtype=float)
    upper = np.asarray(interval["upper"], dtype=float)
    if time.ndim != 1 or ne.shape != time.shape:
        raise ValueError(
            "PHLASH time and median trajectory must be aligned one-dimensional arrays."
        )
    if posterior.ndim != 2 or posterior.shape[1] != time.size:
        raise ValueError("PHLASH posterior_ne must have shape (samples, time).")
    if lower.shape != time.shape or upper.shape != time.shape:
        raise ValueError("PHLASH credible interval must align with time.")
    if not 0 <= posterior_samples <= posterior.shape[0]:
        raise ValueError("posterior_samples exceeds the available posterior draws.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if posterior_samples:
        indices = np.linspace(
            0,
            posterior.shape[0] - 1,
            posterior_samples,
            dtype=int,
        )
        for index in indices:
            ax.plot(time, posterior[index], color=color, alpha=sample_alpha, linewidth=0.65)
    if show_interval:
        percentage = 100 * float(interval["level"])
        ax.fill_between(
            time,
            lower,
            upper,
            color=color,
            alpha=0.22,
            linewidth=0,
            label=f"{percentage:g}% credible interval",
        )
    ax.plot(time, ne, color=color, linewidth=2.2, label=label)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Time before present")
    ax.set_ylabel(r"Effective population size ($N_e$)")
    ax.legend(frameon=False)
    return ax


def save_phlash_figure(
    figure: matplotlib.figure.Figure,
    path: str | Path,
    *,
    dpi: int = 600,
) -> Path:
    """Export a PHLASH figure as vector art or publication-resolution raster."""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".pdf", ".svg", ".eps", ".png", ".tif", ".tiff"}:
        raise ValueError("Use PDF, SVG, EPS, PNG, or TIFF for publication figures.")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output


__all__ = ["phlash_demographic_history", "save_phlash_figure"]
