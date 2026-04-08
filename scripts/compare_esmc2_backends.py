#!/usr/bin/env python3
"""Compare eSMC2 native and upstream backends for parity tracking."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smckit._core import SmcData
from smckit.tl._esmc2 import esmc2


OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "gallery"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTDIR / "esmc2_parity_compare.png"


def _sequence(length: int, het_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1], size=length, p=[1 - het_rate, het_rate]).astype(np.int8)


def _smcdata_from_sequence(seq: np.ndarray) -> SmcData:
    return SmcData(
        uns={
            "records": [{"codes": seq}],
            "sum_L": int((seq < 2).sum()),
            "sum_n": int((seq == 1).sum()),
        }
    )


def _run_backend(seq: np.ndarray, backend: str, **kwargs) -> dict:
    data = _smcdata_from_sequence(seq)
    return esmc2(copy.deepcopy(data), backend=backend, **kwargs).results["esmc2"]


def _rel_diff(native: np.ndarray, upstream: np.ndarray) -> float:
    native = np.asarray(native, dtype=np.float64)
    upstream = np.asarray(upstream, dtype=np.float64)
    denom = np.maximum(np.abs(upstream), 1e-12)
    return float(np.max(np.abs(native - upstream) / denom))


def _plot_xi(native: np.ndarray, upstream: np.ndarray) -> None:
    states = np.arange(len(native))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(states, native, marker="o", linestyle="-", label="Native", color="#0b8bbe")
    ax.plot(states, upstream, marker="s", linestyle="--", label="Upstream", color="#d84b4b")
    ax.set_xlabel("Hidden state index")
    ax.set_ylabel("Xi (relative Ne)")
    ax.set_title("eSMC2: Native vs Upstream Xi(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(states)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the parity comparison and persist the figure."""
    os.environ.setdefault("SMCKIT_ESMC2_RSCRIPT", "/opt/homebrew/bin/Rscript")
    seq = _sequence(length=2000, het_rate=0.02, seed=42)
    params = dict(
        n_states=6,
        n_iterations=20,
        estimate_rho=True,
        estimate_beta=False,
        estimate_sigma=False,
        rho_over_theta=1.0,
        mu=1e-8,
        generation_time=1.0,
        rho_penalty=1.0,
    )
    native = _run_backend(seq, backend="native", **params)
    upstream = _run_backend(seq, backend="upstream", **params)

    metrics = {
        "Tc_rel_diff": _rel_diff(native["Tc"], upstream["Tc"]),
        "Xi_rel_diff": _rel_diff(native["Xi"], upstream["Xi"]),
        "Xi_corr": float(np.corrcoef(native["Xi"], upstream["Xi"])[0, 1]),
        "loglik_diff": abs(native["log_likelihood"] - upstream["log_likelihood"]),
        "native_rho": float(native["rho"]),
        "upstream_rho": float(upstream["rho"]),
    }

    print("eSMC2 parity comparison (native vs upstream)")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6g}")

    plt_kwargs = dict(native=np.asarray(native["Xi"], dtype=np.float64), upstream=np.asarray(upstream["Xi"], dtype=np.float64))
    _plot_xi(**plt_kwargs)


if __name__ == "__main__":
    main()
