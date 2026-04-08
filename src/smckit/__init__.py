"""smckit — Unified framework for Sequentially Markovian Coalescent methods."""

from smckit import io, pl, pp, tl, upstream
from smckit._core import SmcData
from smckit.settings import settings

__version__ = "0.0.1b1"

__all__ = ["SmcData", "io", "pl", "pp", "settings", "tl", "upstream"]
