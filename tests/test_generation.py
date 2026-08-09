import pytest
import urllib.request
import ast
import numpy as np
from scipy import sparse

from tensorpow.su3_sym_runner import worker_task as su3_worker
from tensorpow.sl2_sym_runner import worker_task as sl2_worker
from tensorpow import file_handler

ZENODO_BASE_URL = "https://zenodo.org/records/21862985/files/"

def test_su3_generation_matches_zenodo(tmp_path, monkeypatch):
    k = 6
    zenodo_dir = tmp_path / "zenodo"
    zenodo_dir.mkdir()
    
    sparse_name = f"piM_sym_{k}_T_sparse.npz"
    exps_name = f"piM_sym_{k}_exps.npz"
    
    urllib.request.urlretrieve(ZENODO_BASE_URL + sparse_name, zenodo_dir / sparse_name)
    urllib.request.urlretrieve(ZENODO_BASE_URL + exps_name, zenodo_dir / exps_name)
    
    zenodo_sparse = sparse.load_npz(zenodo_dir / sparse_name)
    with np.load(zenodo_dir / exps_name) as data:
        zenodo_exps = data["exps"]

    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    monkeypatch.setattr(file_handler, "CACHE_DIR", gen_dir)
    
    su3_worker(k)
    
    gen_sparse = sparse.load_npz(gen_dir / sparse_name)
    with np.load(gen_dir / exps_name) as data:
        gen_exps = data["exps"]
        
    assert np.array_equal(zenodo_exps, gen_exps)
    # Compare sparse matrices by checking difference
    diff = zenodo_sparse - gen_sparse
    # Drop near-zero floating point differences just in case, but they should be exact
    diff.data[np.abs(diff.data) < 1e-12] = 0
    diff.eliminate_zeros()
    assert diff.nnz == 0

def test_sl2_generation_matches_zenodo(tmp_path):
    k = 6
    zenodo_file = tmp_path / "sl2reps.txt"
    urllib.request.urlretrieve(ZENODO_BASE_URL + "sl2reps.txt", zenodo_file)
    
    with open(zenodo_file, "r", encoding="utf-8") as f:
        zenodo_data = ast.literal_eval(f.read())
        
    zenodo_rep_k = zenodo_data[k]
    
    result_k, status, msg, generated_rep = sl2_worker(k)
    
    assert status == "SUCCESS"
    assert result_k == k
    
    assert generated_rep == zenodo_rep_k
