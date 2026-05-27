"""Load precomputed SL(2) representation matrices bundled with ``tensorpow``."""

from __future__ import annotations

import ast
from typing import Any, Callable, List

import numpy as np
import sympy as sp
from .file_handler import _ensure_file_downloaded

MAX_SHIPPED_SL2_K = 79

_a, _b, _c, _d = sp.symbols("a b c d")
_SL2_SYMBOLS = (_a, _b, _c, _d)

_raw_reps: List[Any] | None = None
_compiled_cache: dict[int, Callable[..., np.ndarray]] = {}


def _load_raw_reps() -> List[Any]:
    global _raw_reps
    if _raw_reps is not None:
        return _raw_reps

    try:
        # Use caching downloader instead of importlib.resources
        local_path = _ensure_file_downloaded("sl2reps.txt")
        text = local_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise FileNotFoundError(
            "SL(2) representation data could not be found locally or downloaded from GitHub. " "Please check your internet connection."
        ) from exc

    _raw_reps = ast.literal_eval(text)

    if len(_raw_reps) - 1 < MAX_SHIPPED_SL2_K:
        raise ValueError(f"sl2reps.txt has {len(_raw_reps)} entries; " f"expected at least {MAX_SHIPPED_SL2_K + 1}.")
    return _raw_reps


def _compile_k(k: int) -> Callable[..., np.ndarray]:
    if k in _compiled_cache:
        return _compiled_cache[k]

    raw = _load_raw_reps()
    if k < 0 or k >= len(raw):
        raise ValueError(f"SL(2) representation index k={k} out of range (file has k=0..{len(raw) - 1}).")

    mat_strings = raw[k]
    sym_mat = sp.Matrix([[sp.sympify(expr) for expr in row] for row in mat_strings])
    func = sp.lambdify(_SL2_SYMBOLS, sym_mat, "numpy")
    _compiled_cache[k] = func
    return func


def eval_sl2_rep_matrix(k: int, M: np.ndarray) -> np.ndarray:
    """Evaluate the k-th symmetric SL(2) representation matrix at 2x2 matrix M."""
    if k == 0:
        return np.array([[1.0]], dtype=complex)

    if k > MAX_SHIPPED_SL2_K:
        raise ValueError(
            f"SL(2) representation index k={k} exceeds supported data (max k={MAX_SHIPPED_SL2_K}). " "Use a smaller 2x2 tensor power n."
        )

    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    compiled = _compile_k(k)
    return np.asarray(compiled(a, b, c, d), dtype=complex)
