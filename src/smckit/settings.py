"""Global settings for smckit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Global configuration for smckit.

    Attributes
    ----------
    verbosity : int
        Logging verbosity. 0 = silent, 1 = progress, 2 = debug.
    n_jobs : int
        Number of parallel jobs for CPU backends. -1 = all cores.
    seed : int | None
        Random seed for reproducibility.
    """

    verbosity: int = 1
    n_jobs: int = 1
    seed: int | None = None


settings = Settings()
