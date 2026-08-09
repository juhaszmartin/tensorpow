"""Load precomputed SU(3) representation tensors bundled with ``tensorpow``."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import sparse
import platformdirs

# Define local cache and the remote Zenodo URL
CACHE_DIR = Path(platformdirs.user_data_dir("tensorpow", appauthor=False))
ZENODO_BASE_URL = "https://zenodo.org/records/21862985/files/"
MAX_ZENODO_POWER = 30


def _ensure_file_downloaded(filename: str, k: int = 0) -> Path:
    """Check if file exists locally; if not, download it from Zenodo or suggest generation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / filename

    if not local_path.exists():
        if k <= MAX_ZENODO_POWER or filename == "sl2reps.txt":
            print(f"tensorpow: Downloading {filename} from Zenodo (first-time only)...")
            remote_url = ZENODO_BASE_URL + filename
            try:
                urllib.request.urlretrieve(remote_url, local_path)
            except Exception as e:
                # Clean up partial downloads if it crashes midway
                if local_path.exists():
                    local_path.unlink()
                raise RuntimeError(f"Failed to download {filename} from Zenodo: {e}")
        else:
            raise ValueError(
                f"Symmetric power k={k} is not pre-calculated on Zenodo. "
                f"Please run `python -m tensorpow.su3_sym_runner --min-k {k} --max-k {k}` to calculate it locally."
            )

    return local_path


def load_compressed(filename: str):
    """Load ``piM_sym_<k>`` sparse tensor and exponent table from local cache or Zenodo."""
    sparse_name = f"{filename}_T_sparse.npz"
    exps_name = f"{filename}_exps.npz"
    
    # Extract k from filename assuming format like "piM_sym_{k}" or "piM_antisym_{k}"
    parts = filename.split('_')
    try:
        k = int(parts[-1])
    except ValueError:
        k = 0 # Fallback, should not happen

    # Download (if needed) and get local file paths
    sparse_path = _ensure_file_downloaded(sparse_name, k)
    exps_path = _ensure_file_downloaded(exps_name, k)

    # Load the data from the local cache
    T_huge = sparse.load_npz(sparse_path)
    with np.load(exps_path) as data:
        exps = data["exps"]

    return T_huge, exps


def evaluate_compressed_tensor(T_huge, exps, A):
    """Evaluate the sparse tensor at matrix ``A``."""
    n = int(np.sqrt(T_huge.shape[0]))
    vals = np.array(A).reshape(-1)
    monoms = np.prod(np.power(vals, exps), axis=1)
    flat_result = T_huge.dot(monoms)
    return flat_result.reshape(n, n)


def gen_exponents(k, n=9):
    """Generate all exponent tuples of length n summing to k."""
    def recurse(pos, rem, cur):
        if pos == n - 1:
            yield tuple(cur + [rem])
            return
        for e in range(rem, -1, -1):
            yield from recurse(pos + 1, rem - e, cur + [e])

    return list(recurse(0, k, []))


def save_pi_matrix_compressed(piM, k, filename):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows, cols = piM.shape
    matrix_vars = sp.symbols("a b c d e f g h i")

    exps_list = gen_exponents(k, len(matrix_vars))
    M_dim = len(exps_list)
    exps_arr = np.array(exps_list, dtype=np.uint8)  # Save space

    sym_to_idx = {}
    one = sp.Integer(1)
    for idx, exp_tuple in enumerate(exps_list):
        term = one
        for var, e in zip(matrix_vars, exp_tuple):
            if e > 0:
                term *= var**e
        sym_to_idx[term] = idx

    all_data = []
    all_rows = []
    all_cols = []

    total = rows * cols

    for r in range(rows):
        for c in range(cols):
            flat_row_idx = r * cols + c
            expr = piM[r, c]
            coeff_dict = expr.as_coefficients_dict()

            for term_key, dict_val in coeff_dict.items():
                if dict_val == 0:
                    continue
                scalar_part, monomial_part = term_key.as_independent(*matrix_vars)
                final_coeff = float(dict_val * scalar_part)

                idx = sym_to_idx.get(monomial_part)
                if idx is None:
                    if monomial_part == 1:
                        idx = sym_to_idx.get(one)

                all_data.append(final_coeff)
                all_rows.append(flat_row_idx)
                all_cols.append(idx)

    T_huge = sparse.coo_matrix((all_data, (all_rows, all_cols)), shape=(total, M_dim), dtype=np.float64).tocsr()

    sparse_path = CACHE_DIR / f"{filename}_T_sparse.npz"
    exps_path = CACHE_DIR / f"{filename}_exps.npz"
    
    sparse.save_npz(sparse_path, T_huge)
    np.savez_compressed(exps_path, exps=exps_arr)
