"""Demographic history plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    from smckit._core import SmcData


def demographic_history(
    data: SmcData,
    method: str = "psmc",
    mu: float | None = None,
    generation_time: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    log_x: bool = True,
    log_y: bool = False,
    label: str | None = None,
    color: str | None = None,
    show_bootstraps: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot effective population size N_e(t) as a step function.

    Parameters
    ----------
    data : SmcData
        Data with results from an SMC method.
    method : str
        Which method results to plot (default ``"psmc"``).
    mu : float, optional
        Override mutation rate for unit conversion.
    generation_time : float, optional
        Override generation time for unit conversion.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    log_x : bool
        Log-scale x-axis (default True).
    log_y : bool
        Log-scale y-axis.
    label : str, optional
        Legend label.
    color : str, optional
        Line color.
    show_bootstraps : bool
        If True, plot bootstrap replicates as thin lines.
    **kwargs
        Passed to ``matplotlib.axes.Axes.step()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    res = data.results[method]

    # Get or override conversion parameters
    _mu = mu or data.params.get("mu", 1.25e-8)
    _gen = generation_time or data.params.get("generation_time", 25.0)

    time_years = res["time_years"] if "time_years" in res else res["time"] * 2 * res["n0"] * _gen
    ne = res["ne"]

    # Build step-plot arrays (PSMC-style: each lambda spans an interval)
    # Duplicate points for step function
    n_states = len(ne)
    x = np.empty(2 * n_states, dtype=np.float64)
    y = np.empty(2 * n_states, dtype=np.float64)

    for k in range(n_states):
        t_start = time_years[k]
        if k + 1 < n_states:
            t_end = time_years[k + 1]
        else:
            # Extend last interval
            t_end = t_start * 2 if t_start > 0 else 1.0
        x[2 * k] = t_start
        x[2 * k + 1] = t_end
        y[2 * k] = ne[k]
        y[2 * k + 1] = ne[k]

    # Filter out zero/negative for log scale
    if log_x:
        mask = x > 0
        x = x[mask]
        y = y[mask]

    plot_kwargs = {"linewidth": 2}
    plot_kwargs.update(kwargs)
    if color is not None:
        plot_kwargs["color"] = color

    ax.plot(x, y, label=label, **plot_kwargs)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Years ago")
    ax.set_ylabel("Effective population size ($N_e$)")
    ax.set_title("Demographic History")

    if label is not None:
        ax.legend()

    return ax
