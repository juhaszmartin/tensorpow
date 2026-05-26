"""Top-level package for the ``tensorpow`` library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core import TensorPowerCalculator, solve_su3_pieri

__all__ = [
    "TensorPowerCalculator",
    "solve_su3_pieri",
]

try:
    __version__ = version("tensorpow")
except PackageNotFoundError:
    __version__ = "0.2.0"
