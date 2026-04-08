"""Compute backends: NumPy (reference) and Numba JIT (production)."""

from __future__ import annotations

from smckit.settings import settings


def get_array_module(backend: str | None = None):
    """Return the array module for a named or configured backend."""
    selected = settings.backend if backend is None else backend
    if selected == "numpy":
        import numpy as np

        return np
    if selected in {"cupy", "cuda"}:
        import cupy as cp

        return cp
    raise ValueError(f"Unsupported backend: {selected}")


__all__ = ["get_array_module"]
