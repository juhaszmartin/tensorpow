"""Core routines for the ``tensorpow`` package.

SU(3) Pieri decomposition, Young diagram helpers, SL(2)/SU(3) block
decompositions, and ``TensorPowerCalculator`` (``block_decomposition`` and
weighted Schatten-p norms).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from .file_handler import load_compressed, evaluate_compressed_tensor
from .sl2_loader import eval_sl2_rep_matrix

# ---------------------------------------------------------------------------
# combinatorial helpers
# ---------------------------------------------------------------------------


def generate_young_diagrams(n: int, max_rows: int = 3) -> List[Tuple[int, ...]]:
    """All Young diagrams with ``n`` boxes and at most ``max_rows`` rows.

    The result is padded with zeros to ``max_rows`` length.
    """

    def partitions(rem: int, max_len: int, max_val: int) -> List[List[int]]:
        if rem == 0:
            return [[]]
        if max_len == 0:
            return []
        out: List[List[int]] = []
        for i in range(min(rem, max_val), 0, -1):
            for tail in partitions(rem - i, max_len - 1, i):
                out.append([i] + tail)
        return out

    raw = partitions(n, max_rows, n)
    return [tuple(p + [0] * (max_rows - len(p))) for p in raw if len(p) <= max_rows]


def standard_tableaux_count(shape: Tuple[int, ...]) -> int:
    """Hook‑length formula for the number of standard tableaux."""
    n = sum(shape)
    hooks: List[int] = []
    rows = shape
    for i in range(len(rows)):
        for j in range(rows[i]):
            hook_len = rows[i] - j
            for k in range(i + 1, len(rows)):
                if j < rows[k]:
                    hook_len += 1
            hooks.append(hook_len)
    product = 1
    for h in hooks:
        product *= h
    return math.factorial(n) // product


# ---------------------------------------------------------------------------
# block decomposition
# ---------------------------------------------------------------------------


def solve_su3_pieri(n: int, debug: bool = False) -> dict:
    """
    Deterministic O(1) solver for the SU(3) Pieri decomposition.
    Computes the tensor decomposition weights purely using the Jacobi-Trudi
    identity and standard tableaux counts.
    """
    diagrams = generate_young_diagrams(n, max_rows=3)
    weights_map: dict[Tuple[Tuple[int, ...], Tuple[int, ...], int], int] = {}

    for shape in diagrams:
        m = shape[2]
        p = shape[0] - shape[1]
        q = shape[1] - shape[2]
        mult = standard_tableaux_count(shape)

        # Positive term: Sym^{p+q} \otimes Sym^q
        pos_key = ((p + q, 0, 0), (q, 0, 0), m)
        weights_map[pos_key] = weights_map.get(pos_key, 0) + mult

        # Negative term: Sym^{p+q+1} \otimes Sym^{q-1} (only valid if q >= 1)
        if q >= 1:
            neg_key = ((p + q + 1, 0, 0), (q - 1, 0, 0), m)
            weights_map[neg_key] = weights_map.get(neg_key, 0) - mult

    solution: list[dict] = []
    for (a_shape, b_shape, m_val), weight in weights_map.items():
        if weight != 0:
            solution.append(
                {
                    "candidate": (a_shape, b_shape),
                    "m_removed": m_val,
                    "selected": float(weight),
                }
            )

    if debug:
        print(f"Jacobi-Trudi deterministic solver generated {len(solution)} terms.")
        for s in solution:
            print(f"  Weight: {s['selected']:>4.1f} | Pairs: {s['candidate'][0]} x {s['candidate'][1]} | det^{s['m_removed']}")

    return {
        "status": "Optimal",
        "solution": solution,
    }


# ---------------------------------------------------------------------------
# representation utilities
# ---------------------------------------------------------------------------

rep_cache: dict = {}


def label_to_rep(lbl: Union[Tuple[Any, ...], Any]) -> Tuple[Any, Any]:
    if not isinstance(lbl, tuple):
        lbl = tuple(lbl)
    # (1,1,1) is the determinant representation
    if lbl == (1, 1, 1):
        return ("trivial", 1)

    # Remaining labels are symmetric (Jacobi-Trudi)
    if lbl[1] == 0 and lbl[2] == 0:
        return ("sym", lbl[0])

    return lbl


def load_rep(lbl: Tuple[Any, Any]):
    rtype, deg = lbl

    # 0-th power and determinant don't need files
    if rtype == "trivial" or (rtype == "sym" and deg == 0):
        return None

    if lbl in rep_cache:
        return rep_cache[lbl]

    if rtype == "sym":
        fname = f"piM_sym_{deg}"
    else:
        raise ValueError(f"Unknown rep label {lbl}")

    T_huge, exps = load_compressed(fname)
    rep_cache[lbl] = (T_huge, exps)
    return rep_cache[lbl]


def eval_rep_matrix(lbl: Tuple[Any, Any], matrix: np.ndarray) -> np.ndarray:
    # Sym^0 is just the 1x1 scalar 1.0
    if lbl[0] == "sym" and lbl[1] == 0:
        return np.array([[1.0]], dtype=complex)

    if lbl[0] == "trivial":
        det_val = np.linalg.det(matrix)
        return np.array([[det_val]], dtype=complex)

    T_huge, exps = load_rep(lbl)
    return evaluate_compressed_tensor(T_huge, exps, matrix)


# ---------------------------------------------------------------------------
# SL(2) 2x2 specific logic
# ---------------------------------------------------------------------------


def m2_tens_p_block_multipl(n: int, d_dim: int) -> int:
    """Multiplicity of the d_dim irreducible rep of SL(2) in n-fold tensor power."""
    if n < (d_dim - 1) or (n - d_dim + 1) % 2 != 0:
        return 0
    numerator = d_dim * math.comb(n + 1, (n - d_dim + 1) // 2)
    return numerator // (n + 1)


def m2_tens_p_bfm(n: int, M: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Returns diagonal blocks appearing in blockform of M^⊗n for 2x2 M."""
    det_M = np.linalg.det(M)
    s = 2 if n % 2 != 0 else 1

    blockform = []
    for j in range(s, n + 2, 2):
        multiplicity = m2_tens_p_block_multipl(n, j)
        if multiplicity == 0:
            continue

        k_dim = j - 1

        rep_matrix = eval_sl2_rep_matrix(k_dim, M)

        factor = det_M ** ((n - j + 1) / 2)
        block = rep_matrix * factor
        blockform.append((multiplicity, block))

    return blockform


# ---------------------------------------------------------------------------
# public class
# ---------------------------------------------------------------------------


class TensorPowerCalculator:
    """Utility class for computing quantities of tensor powers of matrices.

    Public methods:

    - ``block_decomposition`` — irrep blocks of a single matrix's n-th tensor
      power; shared building block for norms and other functionals.
    - ``schatten_p_norm_weighted`` — weighted Schatten-p norm of a linear
      combination of tensor powers, aggregating block singular values.

    For 3×3 matrices, ``block_decomposition`` may return negative
    multiplicities (virtual decomposition); see that method's docstring.

    The constructor does **not** take any arguments.  All parameters like
    the tensor power ``n`` are provided to the computation methods.  Passing
    anything to ``__init__`` is therefore a user error and will raise a
    helpful ``TypeError``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create an empty calculator.

        Parameters
        ----------
        *args, **kwargs
            Any values passed here are considered incorrect usage.  A
            ``TypeError`` is raised with a clear explanation.  This makes
            accidental calls such as ``TensorPowerCalculator(A)`` easier to
            diagnose.
        """
        if args or kwargs:
            raise TypeError(
                "TensorPowerCalculator() takes no arguments; "
                "pass matrices to schatten_p_norm_weighted() or block_decomposition()"
            )
        self._pieri_cache: dict[int, dict] = {}

    def _get_pieri_result(self, n: int, debug: bool = False) -> dict:
        """Get cached Pieri decomposition result for dimension *n*, computing if needed.

        The block decomposition is only required for tensors of
        degree greater than 1.  For the trivial case ``n==1`` we return an
        empty mapping and callers early-exit before using it.
        """
        # trivial power requires no decomposition
        if n == 1:
            return {}

        if n not in self._pieri_cache:
            if n < 1 or not isinstance(n, int):
                raise ValueError("tensor power n must be a positive integer")
            self._pieri_cache[n] = solve_su3_pieri(n, debug=debug)
        return self._pieri_cache[n]

    def _preload_reps(self, pieri_result: dict) -> None:
        """Ensure all SU(3) representation data needed are loaded."""
        labels = set()
        for entry in pieri_result["solution"]:
            labels.add(entry["candidate"][0])
            labels.add(entry["candidate"][1])
        for lbl in labels:
            rep_lbl = label_to_rep(lbl)
            if rep_lbl[0] != "trivial":
                load_rep(rep_lbl)

    def schatten_p_norm_weighted(
        self,
        matrices: Iterable[Union[np.ndarray, List[float]]],
        n: int = 1,
        p: float = 2.0,
        coeffs: Optional[Iterable[float]] = None,
        debug: bool = False,
    ) -> float:
        """Compute the weighted Schatten-p norm of the n-th tensor power.

        Computes: || c_1 * A^n + c_2 * B^n + ... ||_p

        Parameters
        ----------
        matrices : iterable of ndarray
            Square matrices (either all 2x2 or all 3x3).
        n : int
            Tensor power exponent (positive integer).  ``n=1`` is a special
            case that computes the norm of the weighted sum directly; larger
            values invoke the SU(3) Pieri decomposition for 3x3 case and SL(2)
            for 2x2.
            For 3×3 matrices, precomputed SU(3) data must cover the degrees required
            by the Pieri decomposition. For 2×2 matrices, ``data/sl2reps.txt`` must
            be present; shipped data supports tensor power ``n <= 79``.
        p : float
            Schatten norm exponent (default 2).  ``np.inf`` may be used to
            obtain the operator norm (largest singular value).
        coeffs : iterable of float, optional
            Weights for each matrix (default 1.0 for all).
        debug : bool
            If True, print solver output.

        Returns
        -------
        float
            The Schatten-p norm value.

        Raises
        ------
        ValueError
            If any matrix is not a square array of shape ``(2,2)`` or ``(3,3)``.
        """
        # handle trivial power immediately
        if n == 1:
            # allow a single matrix as input
            if isinstance(matrices, np.ndarray):
                matrices = [matrices]
            mats = [np.array(M, dtype=complex) for M in matrices]
            if not mats:
                raise ValueError("No matrices provided.")
            dim = mats[0].shape[0]
            if dim not in (2, 3):
                raise ValueError(f"Matrix dimension {dim} not supported. Only 2x2 and 3x3 matrices are supported.")
            if coeffs is None:
                coeffs = [1.0] * len(mats)
            coeffs = list(coeffs)
            if len(coeffs) != len(mats):
                raise ValueError(f"Length of coeffs ({len(coeffs)}) must match number of matrices ({len(mats)}).")
            for idx, M in enumerate(mats):
                if M.ndim != 2 or M.shape[0] != M.shape[1]:
                    raise ValueError(f"matrix {idx} is not square: {M.shape}")
                if M.shape[0] != dim:
                    raise ValueError(f"matrix {idx} has shape {M.shape}; expected ({dim},{dim})")
            total_matrix = sum(c * M for c, M in zip(coeffs, mats))
            ss = np.linalg.svd(total_matrix, compute_uv=False)
            if p == np.inf:
                return float(np.max(ss))
            return np.sum(np.power(ss, p)) ** (1.0 / p)

        # Allow a single matrix to be supplied without wrapping in a list.
        if isinstance(matrices, np.ndarray):
            matrices = [matrices]

        # Process matrices
        mats = [np.array(M, dtype=complex) for M in matrices]
        if not mats:
            raise ValueError("No matrices provided.")

        dim = mats[0].shape[0]
        if dim not in (2, 3):
            raise ValueError(f"Matrix dimension {dim} not supported. Only 2x2 and 3x3 matrices are supported.")

        if coeffs is None:
            coeffs = [1.0] * len(mats)
        coeffs = list(coeffs)
        if len(coeffs) != len(mats):
            raise ValueError(f"Length of coeffs ({len(coeffs)}) must match number of matrices ({len(mats)}).")

        # Validate matrix shapes (all must match the detected dimension)
        for idx, M in enumerate(mats):
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"matrix {idx} is not square: {M.shape}")
            if M.shape[0] != dim:
                raise ValueError(f"matrix {idx} has shape {M.shape}; expected ({dim},{dim})")

        # Dispatch to appropriate block decomposition method
        if dim == 2:
            # SL(2) case: 2x2 matrices use their own representation system
            return self._schatten_p_norm_weighted_2x2(mats, n, p, coeffs, debug)
        else:  # dim == 3
            # SU(3) case: load decomposition and representation data only for 3x3 matrices
            pieri_result = self._get_pieri_result(n, debug=debug)
            self._preload_reps(pieri_result)
            return self._schatten_p_norm_weighted_3x3(mats, n, p, coeffs, debug)

    def _schatten_p_norm_weighted_2x2(self, mats: List[np.ndarray], n: int, p: float, coeffs: List[float], debug: bool) -> float:
        # Compute blockform for each matrix and add them weighted
        total_blockform = None
        for i, M_curr in enumerate(mats):
            c_val = coeffs[i]
            if c_val == 0:
                continue

            bfm = m2_tens_p_bfm(n, M_curr)

            # scale the blocks by c_val
            bfm = [(mult, c_val * block) for mult, block in bfm]

            if total_blockform is None:
                total_blockform = bfm
            else:
                total_blockform = [(mult1, b1 + b2) for ((mult1, b1), (_, b2)) in zip(total_blockform, bfm)]

        if total_blockform is None:
            return 0.0

        if p == np.inf:
            max_norm = 0.0
            for mult, block in total_blockform:
                if mult == 0:
                    continue
                s_vals = np.linalg.svd(block, compute_uv=False)
                max_norm = max(max_norm, float(np.max(s_vals)))
            return max_norm
        else:
            total_p_power = 0.0
            for mult, block in total_blockform:
                if mult == 0:
                    continue
                # svd values
                s_vals = np.linalg.svd(block, compute_uv=False)
                block_p_sum = np.sum(np.power(s_vals, p))
                total_p_power += mult * block_p_sum

                if debug:
                    block_norm = np.power(block_p_sum, 1.0 / p) if block_p_sum > 0 else 0.0
                    print(f"Block dim {block.shape[0]}, mult {mult}: ||·||_p = {block_norm:.4f}")

            return np.power(total_p_power, 1.0 / p)

    def _schatten_p_norm_weighted_3x3(self, mats: List[np.ndarray], n: int, p: float, coeffs: List[float], debug: bool) -> float:
        pieri_result = self._get_pieri_result(n, debug=debug)
        self._preload_reps(pieri_result)

        dets = [np.linalg.det(M) for M in mats]
        total_p_power = 0.0

        if debug:
            print(f"{'A_lbl':<10} {'B_lbl':<10} {'k':<6} {'m':<4} {'Dim':<6} {'Block ||·||_p':<15}")
            print("-" * 70)

        max_block_norm = 0.0
        for entry in pieri_result["solution"]:
            a_raw, b_raw = entry["candidate"]
            k = float(entry["selected"])
            m = float(entry["m_removed"])

            a_lbl = label_to_rep(a_raw)
            b_lbl = label_to_rep(b_raw)

            sum_matrix: Optional[np.ndarray] = None

            for i, M_curr in enumerate(mats):
                c_val = coeffs[i]
                if c_val == 0:
                    continue
                pi_a = eval_rep_matrix(a_lbl, M_curr)
                pi_b = eval_rep_matrix(b_lbl, M_curr)
                scalar = c_val * (dets[i] ** m)
                term = scalar * np.kron(pi_a, pi_b)
                if sum_matrix is None:
                    sum_matrix = term
                else:
                    sum_matrix += term

            if sum_matrix is None:
                block_p_sum = 0.0
                sum_dim = 0
            else:
                s_vals = np.linalg.svd(sum_matrix, compute_uv=False)
                if p == np.inf:
                    block_norm = float(np.max(s_vals))
                    block_p_sum = None  # unused
                else:
                    block_p_sum = np.sum(np.power(s_vals, p))
                    block_norm = np.power(block_p_sum, 1.0 / p) if block_p_sum > 0 else 0.0
                sum_dim = sum_matrix.shape[0]

            if p == np.inf:
                # k doesn’t affect the result in the infinity norm
                max_block_norm = max(max_block_norm, block_norm)
            else:
                total_p_power += k * block_p_sum

            if debug:
                if p == np.inf:
                    print(f"{str(a_raw):<10} {str(b_raw):<10} {k:<6.0f} {m:<4.0f} {sum_dim:<6} {block_norm:<15.4f}")
                else:
                    print(f"{str(a_raw):<10} {str(b_raw):<10} {k:<6.0f} {m:<4.0f} {sum_dim:<6} {block_norm:<15.4f}")

        if p == np.inf:
            return max_block_norm
        return np.power(total_p_power, 1.0 / p)

    def block_decomposition(
        self,
        matrix: Union[np.ndarray, List[List[float]]],
        n: int = 1,
        debug: bool = False,
    ) -> List[Tuple[float, np.ndarray]]:
        """Compute the block-diagonal decomposition of the n-th tensor power.

        Returns a list of tuples containing the multiplicity and the
        corresponding block matrix for the given tensor power.  For 3×3
        matrices, multiplicities may be negative (virtual decomposition).

        Parameters
        ----------
        matrix : ndarray or list of lists
            A single square matrix (either 2x2 or 3x3).
        n : int
            Tensor power exponent (positive integer).
        debug : bool
            If True, print solver output.

        Returns
        -------
        list of tuples
            A list where each element is `(multiplicity, block_matrix)`.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError("tensor power n must be a positive integer")

        M = np.array(matrix, dtype=complex)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"matrix is not square: {M.shape}")

        dim = M.shape[0]
        if dim not in (2, 3):
            raise ValueError(f"Matrix dimension {dim} not supported. Only 2x2 and 3x3 matrices are supported.")

        # Trivial power returns the matrix itself with multiplicity 1
        if n == 1:
            return [(1.0, M)]

        # Dispatch to the correct helper
        if dim == 2:
            return self._block_decomposition_2x2(M, n, debug)
        else:  # dim == 3
            return self._block_decomposition_3x3(M, n, debug)

    def _block_decomposition_2x2(self, M: np.ndarray, n: int, debug: bool) -> List[Tuple[float, np.ndarray]]:
        """Helper to get the block decomposition of a 2x2 matrix."""
        # Utilize the existing SL(2) blockform function
        bfm = m2_tens_p_bfm(n, M)

        # Cast integer multiplicities to float for uniform return types
        result = [(float(mult), block) for mult, block in bfm]

        if debug:
            print(f"{'Dim':<6} {'Mult':<6}")
            print("-" * 15)
            for mult, block in result:
                print(f"{block.shape[0]:<6} {mult:<6.0f}")

        return result

    def _block_decomposition_3x3(self, M: np.ndarray, n: int, debug: bool) -> List[Tuple[float, np.ndarray]]:
        """Helper to get the block decomposition of a 3x3 matrix using SU(3) Pieri rules."""
        pieri_result = self._get_pieri_result(n, debug=debug)
        self._preload_reps(pieri_result)

        det_M = np.linalg.det(M)
        blockform = []

        if debug:
            print(f"{'A_lbl':<10} {'B_lbl':<10} {'Mult':<6} {'m':<4} {'Dim':<6}")
            print("-" * 45)

        for entry in pieri_result["solution"]:
            a_raw, b_raw = entry["candidate"]
            k = float(entry["selected"])  # Multiplicity
            m = float(entry["m_removed"])  # Determinant power

            a_lbl = label_to_rep(a_raw)
            b_lbl = label_to_rep(b_raw)

            # Evaluate representation matrices
            pi_a = eval_rep_matrix(a_lbl, M)
            pi_b = eval_rep_matrix(b_lbl, M)

            # Reconstruct the block: det(M)^m * (pi_a ⊗ pi_b)
            scalar = det_M**m
            block = scalar * np.kron(pi_a, pi_b)

            blockform.append((k, block))

            if debug:
                print(f"{str(a_raw):<10} {str(b_raw):<10} {k:<6.0f} {m:<4.0f} {block.shape[0]:<6}")

        return blockform
