"""Global settings for smckit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_backend() -> str:
    try:
        import cupy  # noqa: F401
    except Exception:
        return "numpy"
    return "cupy"


@dataclass
class Settings:
    """Global configuration for smckit.

    Attributes
    ----------
    verbosity : int
        Logging verbosity. 0 = silent, 1 = progress, 2 = debug.
    backend : str
        Compute backend selector. ``"numpy"`` by default, or ``"cupy"``
        when CuPy is importable.
    n_jobs : int
        Number of parallel jobs for CPU backends. -1 = all cores.
    seed : int | None
        Random seed for reproducibility.
    upstream_cache_dir : Path
        Local cache root for bootstrapped upstream tool artifacts.
    upstream_runtime_overrides : dict[str, str]
        Optional per-tool runtime executable overrides.
    """

    backend: str = field(default_factory=_default_backend)
    verbosity: int = 1
    n_jobs: int = 1
    seed: int | None = None
    upstream_cache_dir: Path = Path(".smckit-cache/upstream")
    upstream_runtime_overrides: dict[str, str] = field(default_factory=dict)

    def use_gpu(self) -> bool:
        """Return whether the selected backend targets GPU execution."""
        return self.backend in {"cupy", "cuda"}


settings = Settings()
