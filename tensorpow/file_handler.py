"""Load precomputed SU(3) representation tensors bundled with ``tensorpow``."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse

# Symmetric representation data supported (degrees 1..N).
MAX_SHIPPED_SYM = 26

# Define local cache and the remote GitHub raw content URL
CACHE_DIR = Path.home() / ".tensorpow_data"
GITHUB_BASE_URL = "https://raw.githubusercontent.com/juhaszmartin/tensorpow/refs/heads/main/tensorpow/_data/"


def _ensure_file_downloaded(filename: str) -> Path:
    """Check if file exists locally; if not, download it from GitHub."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / filename

    if not local_path.exists():
        print(f"tensorpow: Downloading {filename} from GitHub (first-time only)...")
        remote_url = GITHUB_BASE_URL + filename
        try:
            urllib.request.urlretrieve(remote_url, local_path)
        except Exception as e:
            # Clean up partial downloads if it crashes midway
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Failed to download {filename} from GitHub: {e}")

    return local_path


def load_compressed(filename: str):
    """Load ``piM_sym_<k>`` sparse tensor and exponent table from local cache or GitHub."""
    sparse_name = f"{filename}_T_sparse.npz"
    exps_name = f"{filename}_exps.npz"

    # Download (if needed) and get local file paths
    sparse_path = _ensure_file_downloaded(sparse_name)
    exps_path = _ensure_file_downloaded(exps_name)

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
