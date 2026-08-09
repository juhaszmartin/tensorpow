import multiprocessing as mp
import sympy as sp
import time
import os
import math
import argparse
from .file_handler import CACHE_DIR

def worker_task(k):
    pid = os.getpid()
    start_time = time.time()
    try:
        print(f"🚀 [PID {pid}] Starting dimension k={k}...")
        a, b, c, d = sp.symbols("a b c d")
        mat = []
        for i in range(k + 1):
            row = []
            for j in range(k + 1):
                val = 0
                min_u = max(0, i - j)
                max_u = min(i, k - j)
                for u in range(min_u, max_u + 1):
                    term = math.comb(k - j, u) * math.comb(j, i - u) * (a ** (k - j - u)) * (c**u) * (b ** (j - i + u)) * (d ** (i - u))
                    val += term
                norm_squared = sp.Rational(math.comb(k, j), math.comb(k, i))
                norm = sp.sqrt(norm_squared)
                final_val = sp.expand(norm * val)
                row.append(str(final_val))
            mat.append(row)
        dt = time.time() - start_time
        return (k, "SUCCESS", f"{dt:.2f}s", mat)
    except Exception as e:
        return (k, "FAILED", str(e), None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SL(2) symmetric representations.")
    parser.add_argument("--max-k", type=int, required=True, help="Maximum symmetric power to generate")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    k_range = list(range(args.max_k + 1))
    total_cores = mp.cpu_count()
    num_workers = args.workers if args.workers else max(1, total_cores - 2)

    print(f"--- PARALLEL GENERATION SL(2) ---")
    print(f"Total Cores: {total_cores}")
    print(f"Using Workers: {num_workers}")
    print(f"Tasks (k): {k_range}")
    print("---------------------------------")

    start_global = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_task, k_range)

    print("\n" + "=" * 30)
    print("      FINAL REPORT      ")
    print("=" * 30)
    results.sort(key=lambda x: x[0])

    reps = []
    for k, status, msg, mat in results:
        symbol = "✅" if status == "SUCCESS" else "❌"
        print(f"{symbol} k={k:<2} (dim {k+1}) : {status} ({msg})")
        if status == "SUCCESS":
            reps.append(mat)
        else:
            reps.append([])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = CACHE_DIR / "sl2reps.txt"
    with open(filename, "w") as f:
        f.write(repr(reps))

    total_dt = time.time() - start_global
    print(f"\nRepresentations written out into {filename}")
    print(f"Total Run Time: {total_dt:.2f}s")
