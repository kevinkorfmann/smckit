"""Publication-oriented SMC++ demographic and cross-validation plots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from smckit._core import SmcData

_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_BLACK = "#000000"


def _payload(data: SmcData | dict[str, Any]) -> dict[str, Any]:
    if hasattr(data, "results"):
        return data.results["smcpp"]
    return data


def _despine(ax: matplotlib.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def smcpp_demographic_history(
    data: SmcData | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
    time_unit: str = "years",
    color: str = _BLUE,
    label: str = "SMC++",
    log_x: bool = True,
    log_y: bool = True,
    show_knots: bool = False,
) -> matplotlib.axes.Axes:
    """Plot a normalized SMC++ population-size history."""
    import matplotlib.pyplot as plt

    result = _payload(data)
    if time_unit == "years":
        time = np.asarray(result["time_years"], dtype=float)
        xlabel = "Time before present (years)"
    elif time_unit == "generations":
        generation_time = float(
            getattr(data, "params", {}).get("generation_time", 1.0)
            if hasattr(data, "params")
            else 1.0
        )
        if generation_time <= 0:
            raise ValueError("generation_time must be positive.")
        time = np.asarray(result["time_years"], dtype=float) / generation_time
        xlabel = "Time before present (generations)"
    elif time_unit == "scaled":
        time = np.asarray(result["time"], dtype=float)
        xlabel = "Scaled coalescent time"
    else:
        raise ValueError("time_unit must be 'years', 'generations', or 'scaled'.")
    ne = np.asarray(result["ne"], dtype=float)
    if time.ndim != 1 or ne.shape != time.shape:
        raise ValueError("SMC++ time and ne must be aligned one-dimensional arrays.")
    if np.any(time <= 0) or np.any(ne <= 0):
        raise ValueError("SMC++ demographic values must be positive.")

    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
    ax.step(time, ne, where="post", color=color, linewidth=1.8, label=label)
    if show_knots:
        ax.scatter(
            time,
            ne,
            s=12,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Effective population size ($N_e$)")
    ax.legend(frameon=False)
    _despine(ax)
    return ax


def smcpp_cross_validation_scores(
    data: SmcData | dict[str, Any],
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot fold-level and summed held-out log likelihoods."""
    import matplotlib.pyplot as plt

    result = _payload(data)
    try:
        cross_validation = result["cross_validation"]
    except KeyError as exc:
        raise ValueError("SMC++ result has no cross-validation evidence.") from exc
    candidates = list(cross_validation["candidates"])
    if not candidates:
        raise ValueError("SMC++ cross-validation candidate list is empty.")
    regularization = np.asarray(
        [candidate["regularization"] for candidate in candidates],
        dtype=float,
    )
    totals = np.asarray(
        [candidate["heldout_log_likelihood"] for candidate in candidates],
        dtype=float,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)

    for candidate in candidates:
        x = float(candidate["regularization"])
        fold_scores = np.asarray(
            [fold["heldout_log_likelihood"] for fold in candidate["folds"]],
            dtype=float,
        )
        ax.scatter(
            np.full(fold_scores.size, x),
            fold_scores,
            color=_BLACK,
            alpha=0.35,
            s=14,
            marker="o",
            zorder=2,
        )
    ax.plot(
        regularization,
        totals,
        color=_BLUE,
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Summed held-out log likelihood",
    )
    selected = float(cross_validation["selected_regularization"])
    selected_index = int(np.flatnonzero(regularization == selected)[0])
    ax.scatter(
        [selected],
        [totals[selected_index]],
        color=_ORANGE,
        edgecolor=_BLACK,
        linewidth=0.6,
        s=35,
        zorder=4,
        label="Selected",
    )
    ax.set_xlabel("Regularization penalty")
    ax.set_ylabel("Held-out log likelihood")
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)
    return ax


def save_smcpp_figure(
    figure: matplotlib.figure.Figure,
    path: str | Path,
    *,
    dpi: int = 600,
) -> Path:
    """Export an SMC++ figure as vector art or publication-resolution raster."""
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() not in {".pdf", ".svg", ".eps", ".png", ".tif", ".tiff"}:
        raise ValueError("Use PDF, SVG, EPS, PNG, or TIFF for publication figures.")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output


__all__ = [
    "save_smcpp_figure",
    "smcpp_cross_validation_scores",
    "smcpp_demographic_history",
]
