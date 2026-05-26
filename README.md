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

Dependencies (`numpy`, `pulp`, `scipy`, `sympy`) are installed automatically.

### Data limits

Precomputed SU(3) symmetric representation data is bundled for degrees **1 through 26**. Some large 3×3 tensor powers may require higher degrees; those cases raise a clear error.

Precomputed SL(2) data (``sl2reps.txt``) supports 2×2 tensor powers **n ≤ 79**.

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

### Precomputed data

- **2×2 (SL(2)):** bundled as `tensorpow/_data/sl2reps.txt`. Supports tensor
  power **n ≤ 79** (representations Sym^k for k = 0..79). Regenerate with the
  external `tensorprod` tool:

  ```bash
  python -m tensorprod.sl2_sym_runner --max-k 79
  ```

- **3×3 (SU(3)):** bundled as `tensorpow/_data/piM_sym_<deg>_*.npz` for the
  degrees required by the Pieri decomposition at your chosen `n`.

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
- `tensorpow/file_handler.py` – loaders for bundled `_data/` NPZ tensors
  (compressed SU(3) representation data; not part of the runtime API)
- `tensorpow/sl2_loader.py` – loads `_data/sl2reps.txt` at runtime

## Testing

Run `pytest tests` to exercise the small test suite.

## License

GNU GPLv3 or later (see `LICENSE`).
