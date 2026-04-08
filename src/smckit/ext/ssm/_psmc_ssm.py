"""PSMC as a State-Space Model.

Hidden states  = coalescent time intervals (discretized)
Observations   = per-window heterozygosity (0=homo, 1=het, 2=missing)
Transition     = coalescent process with recombination
Emission       = mutation process
"""

from __future__ import annotations

import numpy as np

from smckit.backends._numba import (
    compute_hmm_params_jit,
    compute_time_intervals_jit,
)
from smckit.ext.ssm._base import SmcStateSpace
from smckit.tl._psmc import PSMC_N_PARAMS, parse_pattern


class PsmcSSM(SmcStateSpace):
    """PSMC formulated as a state-space model.

    Parameters
    ----------
    pattern : str
        PSMC pattern string (e.g. ``"4+5*3+4"``).
    alpha : float
        Time-interval spacing parameter.
    divergence : bool
        Whether to include a divergence-time parameter.
    """

    def __init__(
        self,
        pattern: str = "4+5*3+4",
        alpha: float = 0.1,
        divergence: bool = False,
    ):
        self.pattern = pattern
        self.par_map, self.n_free, self.n = parse_pattern(pattern)
        self.n_states = self.n + 1
        self.n_obs = 3  # homo, het, missing
        self.alpha = alpha
        self.divergence = divergence

    def _hmm_params(self, params: np.ndarray):
        """Compute HMM matrices from population parameters via Numba backend.

        Returns (a, e, sigma, C_pi, C_sigma).
        """
        t = compute_time_intervals_jit(self.n, float(params[2]), self.alpha)
        return compute_hmm_params_jit(params, self.par_map, self.n, t, self.divergence)

    def transition_matrix(self, params: np.ndarray) -> np.ndarray:
        a, _, _, _, _ = self._hmm_params(params)
        return a

    def emission_matrix(self, params: np.ndarray) -> np.ndarray:
        _, e, _, _, _ = self._hmm_params(params)
        return e

    def initial_distribution(self, params: np.ndarray) -> np.ndarray:
        _, _, sigma, _, _ = self._hmm_params(params)
        return sigma

    def make_initial_params(
        self,
        sum_L: int,
        sum_n: int,
        max_t: float = 15.0,
        tr_ratio: float = 4.0,
        random_init: float = 0.01,
        seed: int | None = None,
    ) -> np.ndarray:
        """Create an initial parameter vector from data statistics.

        Parameters
        ----------
        sum_L : int
            Total number of sites (excluding missing).
        sum_n : int
            Total number of heterozygous sites.
        max_t : float
            Maximum coalescent time (units of 2N0).
        tr_ratio : float
            Initial theta/rho ratio.
        random_init : float
            Amplitude of random perturbation for lambda values.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Parameter vector [theta, rho, max_t, lam_0, ..., lam_{n_free-1}].
        """
        rng = np.random.default_rng(seed)
        n_params = self.n_free + PSMC_N_PARAMS
        params = np.zeros(n_params, dtype=np.float64)

        theta = -np.log(1.0 - sum_n / sum_L) if sum_L > sum_n else sum_n / sum_L
        params[0] = theta
        params[1] = theta / tr_ratio
        params[2] = max_t

        for k in range(PSMC_N_PARAMS, n_params):
            params[k] = 1.0 + (rng.random() * 2.0 - 1.0) * random_init
            if params[k] < 0.1:
                params[k] = 0.1

        return params

    def to_physical_units(
        self,
        params: np.ndarray,
        mu: float = 1.25e-8,
        generation_time: float = 25.0,
        window_size: int = 100,
    ) -> dict:
        """Convert fitted parameters to physical units.

        Parameters
        ----------
        params : np.ndarray
            Fitted parameter vector.
        mu : float
            Per-base per-generation mutation rate.
        generation_time : float
            Generation time in years.
        window_size : int
            Window size used in input data.

        Returns
        -------
        dict
            Keys: ``time_years``, ``ne``, ``lambda_k``, ``n0``, ``theta``, ``rho``.
        """
        theta = float(params[0])
        rho = float(params[1])
        n0 = theta / (4.0 * mu * window_size)

        t = compute_time_intervals_jit(self.n, float(params[2]), self.alpha)
        time_coal = t[: self.n_states]

        lam = params[self.par_map + PSMC_N_PARAMS]
        ne = lam * n0
        time_years = time_coal * 2.0 * n0 * generation_time

        return {
            "time_years": time_years,
            "ne": ne,
            "lambda_k": lam,
            "n0": n0,
            "theta": theta,
            "rho": rho,
            "time": time_coal,
        }
