import multiprocessing as mp
import sympy as sp
import itertools
import math
import time
import os
import argparse
from .file_handler import save_pi_matrix_compressed

def sym_basis_k_symbolic(k, d=3):
    basis = []
    factorial = sp.factorial
    sqrt = sp.sqrt

    def generate(remain, parts):
        if len(parts) == d - 1:
            parts = parts + [remain]
            a, b, c = parts
            norm = sqrt(factorial(a) * factorial(b) * factorial(c) / factorial(k))
            basis.append((tuple(parts), sp.simplify(norm)))
            return

        for i in range(remain, -1, -1):
            generate(remain - i, parts + [i])

    generate(k, [])
    return basis

def pi_symmetric_multinomial_direct(M, basis):
    d = M.rows
    n = len(basis)
    if n == 0:
        return sp.zeros(0)

    max_k = sum(basis[0][0])
    basis_map = {tuple(vec): i for i, (vec, norm) in enumerate(basis)}
    facts = [math.factorial(i) for i in range(max_k + 1)]
    m_vars = list(M)
    num_m_vars = len(m_vars)
    m_indices = [(u, v) for u in range(d) for v in range(d)]

    power_cache = [[sp.Integer(1)] * (max_k + 1) for _ in range(num_m_vars)]
    for i in range(num_m_vars):
        base_sym = m_vars[i]
        curr = base_sym
        for e in range(1, max_k + 1):
            power_cache[i][e] = curr
            curr *= base_sym

    matrix_terms = {}
    iterable = itertools.combinations(range(max_k + num_m_vars - 1), num_m_vars - 1)

    for bars in iterable:
        exponents = []
        prev = -1
        for b in bars:
            exponents.append(b - prev - 1)
            prev = b
        exponents.append((max_k + num_m_vars - 1) - prev - 1)

        denom = 1
        for e in exponents:
            if e > 1:
                denom *= facts[e]
        coeff = facts[max_k] // denom

        r_vec = [0] * d
        c_vec = [0] * d
        for i, e in enumerate(exponents):
            if e > 0:
                u, v = m_indices[i]
                r_vec[u] += e
                c_vec[v] += e

        row_idx = basis_map.get(tuple(r_vec))
        col_idx = basis_map.get(tuple(c_vec))

        if row_idx is not None and col_idx is not None:
            if (row_idx, col_idx) not in matrix_terms:
                matrix_terms[(row_idx, col_idx)] = []
            matrix_terms[(row_idx, col_idx)].append((coeff, exponents))

    piM = sp.zeros(n, n)
    for (r, c), terms in matrix_terms.items():
        add_args = []
        norm_scalar = basis[r][1] * basis[c][1]
        for coeff, exps in terms:
            term_scalar = coeff * norm_scalar
            mul_args = [term_scalar]
            for i, e in enumerate(exps):
                if e > 0:
                    mul_args.append(power_cache[i][e])
            add_args.append(sp.Mul(*mul_args))
        piM[r, c] = sp.Add(*add_args)

    return piM


def worker_task(k):
    pid = os.getpid()
    start_time = time.time()
    filename = f"piM_sym_{k}"

    try:
        print(f"🚀 [PID {pid}] Starting k={k}...")
        d = 3
        basis = sym_basis_k_symbolic(k, d=d)
        M_sym = sp.Matrix(d, d, lambda i, j: sp.symbols(chr(97 + i * d + j)))
        piM = pi_symmetric_multinomial_direct(M_sym, basis)
        save_pi_matrix_compressed(piM, k, filename)
        dt = time.time() - start_time
        return (k, "SUCCESS", f"{dt:.2f}s")
    except Exception as e:
        return (k, "FAILED", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SU(3) symmetric representations.")
    parser.add_argument("--min-k", type=int, default=1, help="Minimum symmetric power to generate")
    parser.add_argument("--max-k", type=int, required=True, help="Maximum symmetric power to generate")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    k_range = list(range(args.min_k, args.max_k + 1))
    k_range.sort(reverse=True)

    total_cores = mp.cpu_count()
    num_workers = args.workers if args.workers else max(1, total_cores - 2)

    print(f"--- PARALLEL GENERATION ---")
    print(f"Total Cores: {total_cores}")
    print(f"Using Workers: {num_workers}")
    print(f"Tasks (k): {k_range}")
    print("---------------------------")

    start_global = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_task, k_range)

    print("\n" + "=" * 30)
    print("       FINAL REPORT       ")
    print("=" * 30)

    results.sort(key=lambda x: x[0])

    for k, status, msg in results:
        symbol = "✅" if status == "SUCCESS" else "❌"
        print(f"{symbol} k={k:<2} : {status} ({msg})")

    total_dt = time.time() - start_global
    print(f"\nTotal Run Time: {total_dt:.2f}s")
