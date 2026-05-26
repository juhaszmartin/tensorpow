"""Load precomputed SU(3) representation tensors bundled with ``tensorpow``."""

from __future__ import annotations

from importlib import resources

import numpy as np
from scipy import sparse

# Symmetric representation data shipped in ``tensorpow/_data/`` (degrees 1..N).
MAX_SHIPPED_SYM = 26


def _data_file(name: str) -> resources.abc.Traversable:
    return resources.files("tensorpow") / "_data" / name


def load_compressed(filename: str):
    """Load ``piM_sym_<k>`` sparse tensor and exponent table from package data."""
    stem = filename
    sparse_path = _data_file(f"{stem}_T_sparse.npz")
    exps_path = _data_file(f"{stem}_exps.npz")
    with resources.as_file(sparse_path) as sparse_local:
        T_huge = sparse.load_npz(sparse_local)
    with resources.as_file(exps_path) as exps_local:
        with np.load(exps_local) as data:
            exps = data["exps"]
    return T_huge, exps


def evaluate_compressed_tensor(T_huge, exps, A):
    """Evaluate the sparse tensor at matrix ``A``."""
    n = int(np.sqrt(T_huge.shape[0]))
    vals = np.array(A).reshape(-1)
    monoms = np.prod(np.power(vals, exps), axis=1)
    flat_result = T_huge.dot(monoms)
    return flat_result.reshape(n, n)
