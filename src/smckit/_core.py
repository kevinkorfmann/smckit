"""Core data container for smckit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SmcData:
    """Central data container for SMC-based demographic inference.

    Inspired by AnnData — carries input data, inferred parameters, and results
    through the full analysis pipeline.

    Parameters
    ----------
    sequences : np.ndarray | None
        Input sequence data. Format depends on the method:
        - PSMC: binary heterozygosity array (n_windows,)
        - MSMC: multi-sample haplotype matrix (n_haplotypes, n_sites)
    window_size : int
        Window size used for binning heterozygosity calls.
    params : dict
        Model parameters (mutation rate, recombination rate, time intervals, etc.).
    results : dict
        Inference results keyed by method name, e.g.
        ``{"psmc": {"ne": ..., "time": ..., "log_likelihood": ...}}``.
    uns : dict
        Unstructured annotations. Free-form storage for metadata, intermediate
        computations, or method-specific data that doesn't fit elsewhere.
    """

    sequences: np.ndarray | None = None
    window_size: int = 100
    params: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sites(self) -> int | None:
        """Number of sites/windows in the input data."""
        if self.sequences is None:
            return None
        return self.sequences.shape[-1]

    def __repr__(self) -> str:
        parts = [f"SmcData (n_sites={self.n_sites})"]
        if self.params:
            parts.append(f"  params: {list(self.params.keys())}")
        if self.results:
            parts.append(f"  results: {list(self.results.keys())}")
        return "\n".join(parts)

    def result(self, method: str) -> dict[str, Any]:
        """Return one method result, raising a precise error when absent."""
        try:
            return self.results[method]
        except KeyError as exc:
            available = ", ".join(sorted(self.results)) or "none"
            raise KeyError(
                f"No result for method {method!r}; available results: {available}."
            ) from exc

    def times(self, method: str) -> np.ndarray:
        """Return the normalized time axis for one result."""
        result = self.result(method)
        for key in ("time_years", "time", "left_boundary", "time_boundaries"):
            if key in result:
                return np.asarray(result[key])
        raise KeyError(f"Result {method!r} has no normalized time field.")

    def effective_population_size(self, method: str) -> np.ndarray:
        """Return the normalized effective-population-size trajectory."""
        result = self.result(method)
        for key in ("ne", "N", "N1"):
            if key in result:
                return np.asarray(result[key])
        raise KeyError(f"Result {method!r} has no effective population size field.")

    def log_likelihood(self, method: str) -> float:
        """Return the final log likelihood for one result."""
        result = self.result(method)
        if "log_likelihood" not in result:
            raise KeyError(f"Result {method!r} has no log_likelihood field.")
        return float(result["log_likelihood"])
