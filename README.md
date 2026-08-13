# tensorpow

`tensorpow` is a small Python library for working with norms and
representations of tensor powers.  Base matrices can be either **2×2**
(using SL(2) representation data) or **3×3** (using SU(3) representation
data).  The **tensor power** itself may be any positive integer – the
library uses precomputed representation data indexed by that power.  A
class-based interface computes block decompositions via precomputed SL(2)/SU(3)
data and evaluates quantities such as Schatten‑p norms without building the
full Kronecker power.

## Installation

Requires **Python 3.11–3.13**.

```bash
pip install tensorpow
```

For development:

```bash
pip install -e ".[test]"
```

Dependencies (`numpy`, `scipy`, `sympy`) are installed automatically.

### Data limits

Precomputed SU(3) symmetric representation data is available for degrees **1 through 30**. Larger 3×3 tensor powers require higher degrees; those cases raise a clear error.

Precomputed SL(2) data (``sl2reps.txt``) supports 2×2 tensor powers **n ≤ 79**.


## GPU Acceleration

The `tensorpow` package supports NVIDIA GPU acceleration for operations involving 3x3 matrices to dramatically speed up large tensor power calculations (e.g. $n \ge 10$).

Because CuPy is heavily dependent on your specific CUDA version, you must manually install the appropriate CuPy wheel for your machine in addition to the optional GPU dependency:

```bash
# Install tensorpow with GPU dependencies
pip install tensorpow[gpu]

# Install the CuPy wheel matching your CUDA version (e.g. CUDA 13.x for Blackwell architectures)
pip install cupy-cuda13x
```

Once installed, you can pass the `backend` flag to `schatten_p_norm_weighted`:
```python
val = calc.schatten_p_norm_weighted([A, B], n=20, p=2, backend="smart_hybrid")
```

**Backend Options:**
- `"cpu"` (Default): Uses standard NumPy and runs entirely on the CPU.
- `"smart_hybrid"`: The recommended GPU implementation. It performs massive matrix exponentiation on the GPU, offloads the scaling block aggregations back to System RAM to gracefully bypass VRAM limits, and attempts to compute the final SVD on the GPU (automatically falling back to CPU SVD for massive dimension blocks).
- `"cupy"`: A pure CuPy implementation that attempts to keep the entire block pipeline strictly in VRAM. This avoids PCIe transfer overhead but will easily exceed VRAM on standard cards for $n \ge 16$.

## Quick start

```python
from tensorpow import TensorPowerCalculator
import numpy as np

# Example 1: 3×3 matrices (SU(3) case)
A = np.eye(3)
B = 2 * np.eye(3)
calc = TensorPowerCalculator()           # no args
# compute Schatten‑2 norm of A⊗A - 1/2 * B⊗B (tensorpower=2) 
norm_3x3 = calc.schatten_p_norm_weighted([A, B], n=2, p=2, coeffs=[1.0, -0.5])
print(f"3×3 result: {norm_3x3}")

# Example 2: 2×2 matrices (SL(2) case)
C = np.eye(2)
D = 2 * np.eye(2)
# compute Schatten‑2 norm of C⊗C - 1/2 * D⊗D (tensorpower=2)
norm_2x2 = calc.schatten_p_norm_weighted([C, D], n=2, p=2, coeffs=[1.0, -0.5])
print(f"2×2 result: {norm_2x2}")

# Example 3: block decomposition of a single 2×2 matrix (tensor power n=5)
M = np.array([[1.0, 0.5], [0.0, 1.0]])
blocks = calc.block_decomposition(M, n=5)
# each entry is (multiplicity, block_matrix)
for mult, block in blocks:
    print(mult, block.shape)
```

`TensorPowerCalculator` exposes two public methods:

- **`block_decomposition`** — irrep blocks of one matrix's n-th tensor power
- **`schatten_p_norm_weighted`** — weighted Schatten-p norm of a linear
  combination of tensor powers

Both 2×2 and 3×3 matrices are supported; the library dispatches to SL(2) or
SU(3) representation data based on matrix dimension.

For **3×3** matrices, `block_decomposition` may return **negative**
multiplicities (a virtual decomposition). Singular values of the full tensor
power are not listed block-by-block; use weighted sums of singular-value powers
over blocks (as `schatten_p_norm_weighted` does internally).

## Generating Representation Data Locally

The package relies on massive symmetric representation tensors. By default, it downloads precomputed tensors (from Zenodo) securely into your OS user data directory (e.g. `~/.local/share/tensorpow`) so they are preserved across updates.

However, you can run the built-in representation generators locally on your own machine to extend the package's capabilities beyond the published limits! These symbolic generator scripts are thoroughly tested in `tests/test_generation.py` to ensure their output exactly matches the Zenodo baseline.

### Extending SL(2) (2×2 Matrices)

The base package supports up to $n \le 79$ for 2x2 matrices (Sym^k for k = 0..79). If you need higher tensor powers, you can easily generate them:

```bash
python -m tensorpow.sl2_sym_runner --max-k 100
```
This multi-processed script will compute the representations and append them locally to your cached `sl2reps.txt` file.

### Extending SU(3) (3×3 Matrices)

The base package supports up to $n \le 30$ for 3x3 matrices using compressed `piM_sym_<deg>_*.npz` files. If you need dimensions beyond degree 30, you can generate the sparse tensor representations locally (note: this is computationally heavy and automatically runs in parallel across all CPU cores):

```bash
python -m tensorpow.su3_sym_runner --min-k 31 --max-k 35 --workers 8
```

Your custom generated representation files are instantly recognized by `tensorpow` in your next run!

### Building on the decomposition

`block_decomposition` exposes the irrep blocks of a single matrix's n-th tensor
power. Methods such as `schatten_p_norm_weighted` combine block singular values
without forming the full Kronecker product; you can use the same blocks to define
other functionals (other norms, traces, eigenvalue statistics, and so on).
Additional calculator methods may be added in future releases; they will build on
this decomposition.

## Package structure

- `tensorpow/core.py` – main implementation and `TensorPowerCalculator`,
  including SL(2) (2×2) and SU(3) (3×3) block decomposition logic
- `tensorpow/*_runner.py` – generator tools to calculate arbitrary dimension representations locally
- `tensorpow/file_handler.py` – loaders for downloaded NPZ tensors
  (compressed SU(3) representation data; not part of the runtime API)
- `tensorpow/sl2_loader.py` – loads downloaded `sl2reps.txt` at runtime

## Testing

Run `pytest tests` to exercise the small test suite.

## License

GNU GPLv3 or later (see `LICENSE`).
