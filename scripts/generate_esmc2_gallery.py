"""Generate eSMC2 gallery figures for the smckit documentation.

Produces three figures:
1. esmc2_time_scaling.png — How dormancy and selfing modify HMM structure
2. esmc2_transition_matrix.png — Transition matrices under different ecological regimes
3. esmc2_demographic_inference.png — eSMC2 recovering demography from msprime simulation

Usage:
    python scripts/generate_esmc2_gallery.py
"""

import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import smckit
from smckit.backends._numba_esmc2 import (
    esmc2_build_time_boundaries,
    esmc2_build_transition_matrix,
    esmc2_equilibrium_probs,
    esmc2_expected_times,
)
from smckit.tl._esmc2 import esmc2

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "docs", "gallery")
os.makedirs(OUTDIR, exist_ok=True)

MU = 1e-8
Ne0 = 10000


# ---------------------------------------------------------------------------
# Figure 1: Time scaling with dormancy and selfing
# ---------------------------------------------------------------------------


def fig_time_scaling():
    """Show how beta and sigma modify time boundaries and equilibrium probs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = 20

    configs = [
        (1.0, 0.0, "Standard ($\\beta$=1, $\\sigma$=0)", "#2196F3"),
        (0.5, 0.0, "Dormancy ($\\beta$=0.5)", "#FF9800"),
        (1.0, 0.5, "Selfing ($\\sigma$=0.5)", "#4CAF50"),
        (0.5, 0.5, "Both ($\\beta$=0.5, $\\sigma$=0.5)", "#F44336"),
    ]

    # Left: Time boundaries
    ax = axes[0]
    for beta, sigma, label, color in configs:
        Tc = esmc2_build_time_boundaries(n, beta, sigma)
        ax.plot(range(n), Tc, "o-", label=label, color=color, markersize=4, linewidth=1.5)
    ax.set_xlabel("Hidden state index")
    ax.set_ylabel("Coalescent time ($2N_0$ generations)")
    ax.set_title("Time Discretization")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Equilibrium probabilities
    ax = axes[1]
    Xi = np.ones(n, dtype=np.float64)
    for i, (beta, sigma, label, color) in enumerate(configs):
        Tc = esmc2_build_time_boundaries(n, beta, sigma)
        q = esmc2_equilibrium_probs(Tc, Xi, beta, sigma)
        ax.bar(np.arange(n) + i * 0.2 - 0.3, q, width=0.2, label=label, color=color, alpha=0.8)
    ax.set_xlabel("Hidden state index")
    ax.set_ylabel("Equilibrium probability")
    ax.set_title("Initial State Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("eSMC2: Effect of Dormancy and Selfing on Time Structure", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(OUTDIR, "esmc2_time_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Transition matrix heatmaps (realistic parameters)
# ---------------------------------------------------------------------------


def fig_transition_matrix():
    """Heatmap of transition matrices under different ecological regimes.

    Uses realistic population-genetics parameters matching eSMC2 Tutorial 3:
    Ne=10^4, mu=10^-8, r=10^-8, L=10^7.
    """
    n = 20
    Xi = np.ones(n, dtype=np.float64)
    # rho = 4*Ne*r*L = 4*10^4*10^-8*10^7 = 4000
    rho = 4000.0
    L = 10_000_000

    configs = [
        (1.0, 0.0, "Standard\n($\\beta$=1, $\\sigma$=0)"),
        (0.5, 0.0, "Dormancy\n($\\beta$=0.5)"),
        (1.0, 0.7, "Selfing\n($\\sigma$=0.7)"),
        (0.5, 0.5, "Both\n($\\beta$=0.5, $\\sigma$=0.5)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    vmin, vmax = None, None
    # Pre-compute global min/max for consistent color scale
    all_vals = []
    for beta, sigma, _ in configs:
        Tc = esmc2_build_time_boundaries(n, beta, sigma)
        t = esmc2_expected_times(Tc, Xi, beta, sigma)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)
        off = Q.copy()
        np.fill_diagonal(off, 0)
        pos = off[off > 0]
        if len(pos) > 0:
            all_vals.extend(np.log10(pos).tolist())
    if all_vals:
        vmin = np.percentile(all_vals, 1)
        vmax = np.percentile(all_vals, 99)

    for idx, (beta, sigma, title) in enumerate(configs):
        ax = axes[idx]
        Tc = esmc2_build_time_boundaries(n, beta, sigma)
        t = esmc2_expected_times(Tc, Xi, beta, sigma)
        Q = esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)

        Q_plot = Q.copy()
        np.fill_diagonal(Q_plot, np.nan)
        Q_plot = np.where(Q_plot > 0, np.log10(Q_plot), np.nan)

        im = ax.imshow(
            Q_plot, cmap="viridis", aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("To state")
        if idx == 0:
            ax.set_ylabel("From state")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="log$_{10}$(Q$_{ij}$)")
    fig.suptitle(
        "eSMC2: Transition Matrix (off-diagonal, $N_e$=10$^4$, $L$=10$^7$)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    path = os.path.join(OUTDIR, "esmc2_transition_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Demographic inference on msprime-simulated data
# ---------------------------------------------------------------------------


def _simulate_psmcfa(
    Ne: float,
    mu: float,
    recomb_rate: float,
    length: float,
    demography: msprime.Demography | None = None,
    seed: int = 42,
) -> str:
    """Simulate diploid genome with msprime, write as PSMCFA, return path."""
    if demography is None:
        demography = msprime.Demography()
        demography.add_population(initial_size=Ne)

    ts = msprime.sim_ancestry(
        samples=1,
        demography=demography,
        sequence_length=length,
        recombination_rate=recomb_rate,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)

    # Collect heterozygous positions efficiently
    het_positions = np.array(
        [int(v.site.position) for v in ts.variants() if v.genotypes[0] != v.genotypes[1]]
    )

    # Convert to PSMCFA (window_size=100) using numpy binning
    window = 100
    n_windows = int(length) // window
    if len(het_positions) > 0:
        window_indices = het_positions // window
        window_indices = window_indices[window_indices < n_windows]
        het_windows = np.zeros(n_windows, dtype=bool)
        het_windows[window_indices] = True
    else:
        het_windows = np.zeros(n_windows, dtype=bool)
    seq = ["K" if het_windows[w] else "T" for w in range(n_windows)]

    tmp = tempfile.NamedTemporaryFile(suffix=".psmcfa", delete=False, mode="w")
    tmp.write(">sim\n")
    for i in range(0, len(seq), 60):
        tmp.write("".join(seq[i : i + 60]) + "\n")
    tmp.close()
    return tmp.name


def fig_demographic_inference():
    """Simulate bottleneck demography with msprime, recover with eSMC2."""
    print("Simulating with msprime (bottleneck scenario)...")

    # Bottleneck scenario matching Tutorial_3_A:
    # Ne=10^4, bottleneck to Ne/10 between 1000-10000 gen ago
    demography = msprime.Demography()
    demography.add_population(initial_size=Ne0)
    demography.add_population_parameters_change(time=1000, initial_size=Ne0 / 10)
    demography.add_population_parameters_change(time=10000, initial_size=Ne0)

    L = 5e7  # 50 Mb
    r = 1e-8

    psmcfa_path = _simulate_psmcfa(Ne0, MU, r, L, demography=demography, seed=2024)

    print(f"  Simulated {L/1e6:.0f} Mb, reading...")
    data = smckit.io.read_psmcfa(psmcfa_path)
    os.unlink(psmcfa_path)

    n_hets = data.uns["sum_n"]
    n_sites = data.uns["sum_L"]
    print(f"  {n_sites} callable sites, {n_hets} het windows")

    # Run eSMC2
    print("Running eSMC2 (20 states, 20 iterations)...")
    data = esmc2(
        data,
        n_states=20,
        n_iterations=20,
        mu=MU,
        generation_time=1.0,
        rho_over_theta=r / MU,
    )

    res = data.results["esmc2"]

    # Build the true demography step function
    true_t = np.array([1, 1000, 10000, 1e6])
    true_ne = np.array([Ne0, Ne0 / 10, Ne0, Ne0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Ne(t) recovery
    ax = axes[0]
    ax.step(true_t, true_ne, where="post", color="black", linewidth=2.5, label="True demography")
    ax.plot(res["time_years"], res["ne"], "o-", color="#FF9800", linewidth=2, markersize=4,
            label="eSMC2 (20 states, 20 iter)")
    ax.set_xscale("log")
    ax.set_xlabel("Generations ago")
    ax.set_ylabel("Effective population size ($N_e$)")
    ax.set_title("Demographic History Recovery")
    ax.set_xlim(100, 2e5)
    ax.set_ylim(0, Ne0 * 2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: EM convergence
    ax = axes[1]
    rounds = res["rounds"]
    iters = [rd["round"] for rd in rounds]
    lls = [rd["log_likelihood"] for rd in rounds]
    ax.plot(iters, lls, "o-", color="#FF9800", linewidth=2, markersize=5)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("EM Convergence")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "eSMC2: Bottleneck Recovery (msprime, $L$=50Mb, $N_e$=10$^4$, 10x bottleneck)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(OUTDIR, "esmc2_demographic_inference.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig_time_scaling()
    fig_transition_matrix()
    fig_demographic_inference()
    print("Done.")
