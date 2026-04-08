"""Abstract base class for State-Space Model formulations of SMC methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    """Result from fitting an SMC state-space model.

    Parameters
    ----------
    params : np.ndarray
        Optimized parameter vector [theta, rho, max_t, lam_0, ..., lam_{n_free-1}].
    log_likelihood : float
        Final log-likelihood value.
    n_iterations : int
        Number of iterations performed.
    converged : bool
        Whether the optimizer converged.
    history : list of dict
        Per-iteration records (loss, params, etc.).
    """

    params: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool
    history: list[dict] = field(default_factory=list)


class SmcStateSpace(ABC):
    """Abstract state-space model for SMC methods.

    Subclasses define the coalescent-to-HMM mapping by implementing three
    methods: :meth:`transition_matrix`, :meth:`emission_matrix`, and
    :meth:`initial_distribution`. The base class provides generic
    :meth:`log_likelihood` and :meth:`fit` using these building blocks.

    Attributes
    ----------
    n_states : int
        Number of hidden states (coalescent time intervals).
    n_obs : int
        Number of observation symbols.
    """

    n_states: int
    n_obs: int

    @abstractmethod
    def transition_matrix(self, params: np.ndarray) -> np.ndarray:
        """Compute the (n_states, n_states) transition matrix.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.

        Returns
        -------
        np.ndarray
            Row-stochastic transition matrix.
        """

    @abstractmethod
    def emission_matrix(self, params: np.ndarray) -> np.ndarray:
        """Compute the (n_obs, n_states) emission matrix.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.

        Returns
        -------
        np.ndarray
            Emission probabilities per symbol per state.
        """

    @abstractmethod
    def initial_distribution(self, params: np.ndarray) -> np.ndarray:
        """Compute the (n_states,) initial state distribution.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.

        Returns
        -------
        np.ndarray
            Probability vector summing to 1.
        """

    def log_likelihood(
        self,
        params: np.ndarray,
        observations: list[np.ndarray],
    ) -> float:
        """Compute log P(observations | params) via the scaled forward algorithm.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        observations : list of np.ndarray
            Each element is an int8 observation sequence (e.g. one chromosome).

        Returns
        -------
        float
            Total log-likelihood summed over all sequences.
        """
        from smckit.backends._numba import forward_jit, log_likelihood_jit

        a = self.transition_matrix(params)
        e = self.emission_matrix(params)
        a0 = self.initial_distribution(params)

        total_ll = 0.0
        for seq in observations:
            f, s = forward_jit(a, e, a0, seq)
            total_ll += log_likelihood_jit(s)
        return total_ll

    def fit(
        self,
        observations: list[np.ndarray],
        params_init: np.ndarray,
        method: str = "gradient",
        n_iterations: int | None = None,
        **kwargs,
    ) -> FitResult:
        """Fit the model to observations.

        Parameters
        ----------
        observations : list of np.ndarray
            Observation sequences (one per chromosome/record).
        params_init : np.ndarray
            Initial parameter vector.
        method : str
            ``"gradient"`` for JAX gradient-based optimization (requires JAX),
            ``"em"`` for EM with Hooke-Jeeves M-step (NumPy/Numba fallback).
        n_iterations : int, optional
            Number of iterations. Defaults to 500 for gradient, 30 for EM.
        **kwargs
            Passed to the backend fitting function.

        Returns
        -------
        FitResult
            Optimization results.
        """
        if method == "gradient":
            from smckit.ext.ssm._jax_backend import fit_gradient

            if n_iterations is None:
                n_iterations = 500
            return fit_gradient(self, observations, params_init,
                                n_iterations=n_iterations, **kwargs)
        elif method == "em":
            from smckit.ext.ssm._numpy_backend import fit_em

            if n_iterations is None:
                n_iterations = 30
            return fit_em(self, observations, params_init,
                          n_iterations=n_iterations, **kwargs)
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'gradient' or 'em'.")
