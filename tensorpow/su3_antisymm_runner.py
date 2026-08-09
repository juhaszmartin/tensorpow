"""SU(3) antisymmetric representation generator.

Generates and saves the 3×3 antisymmetric (wedge product) representation
of SU(3) matrices. There is only one such irrep (the adjoint representation),
computed as the exterior power ∧² of the defining representation.
"""

import sympy as sp
from sympy import sqrt

from .file_handler import save_pi_matrix_compressed


def compute_antisymmetric_representation():
    """Compute the 3×3 antisymmetric representation matrix piM."""
    a, b, c, d, e, f, g, h, i = sp.symbols("a b c d e f g h i")
    M = sp.Matrix([[a, b, c], [d, e, f], [g, h, i]])

    e1 = sp.Matrix([1, 0, 0])
    e2 = sp.Matrix([0, 1, 0])
    e3 = sp.Matrix([0, 0, 1])

    b1 = (e1 * e2.T - e2 * e1.T) / sqrt(2)
    b2 = (e1 * e3.T - e3 * e1.T) / sqrt(2)
    b3 = (e2 * e3.T - e3 * e2.T) / sqrt(2)
    basis = [b1, b2, b3]

    piM = sp.zeros(3, 3)
    for j, bj in enumerate(basis):
        for k, bk in enumerate(basis):
            Mbk = M * bk * M.T
            piM[j, k] = sp.simplify(sp.trace(bj.T * Mbk))

    return piM


if __name__ == "__main__":
    print("Computing SU(3) antisymmetric representation...")
    piM = compute_antisymmetric_representation()

    print("Generated 3×3 antisymmetric representation matrix:")
    print(piM)

    k_val = 2
    filename = f"piM_antisym_{k_val}"
    print(f"\nSaving to {filename}...")
    save_pi_matrix_compressed(piM, k_val, filename)
    print("✅ Done!")
