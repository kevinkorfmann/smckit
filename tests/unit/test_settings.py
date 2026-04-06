"""Tests for settings and backend dispatch."""

from smckit.backends import get_array_module
from smckit.settings import Settings


def test_default_settings():
    s = Settings()
    assert s.backend in ("numpy", "cupy")
    assert s.verbosity == 1


def test_numpy_backend():
    import numpy as np

    xp = get_array_module("numpy")
    assert xp is np


def test_use_gpu():
    s = Settings(backend="numpy")
    assert not s.use_gpu()
    s.backend = "cupy"
    assert s.use_gpu()
    s.backend = "cuda"
    assert s.use_gpu()
