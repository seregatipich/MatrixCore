"""Benchmark MatrixCore solvers against NumPy/SciPy on SPD systems.

Reports per-method wall-clock time and accuracy (relative error vs
numpy.linalg.solve) across a range of matrix sizes. The hand-written C solvers
are not expected to beat LAPACK; this quantifies the gap and confirms accuracy.

Run: python benchmarks/benchmark.py
"""

import time

import numpy as np
import scipy.linalg as sla

from matrixcore import solve_system

METHODS = [
    "gaussian_elimination",
    "lu_decomposition",
    "cholesky",
    "qr_decomposition",
    "conjugate_gradient",
]
SIZES = [16, 32, 64, 128]
REPEATS = 5


def spd(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1.0, 1.0, size=(n, n))
    return M @ M.T + n * np.eye(n)


def timed(fn, repeats=REPEATS):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best


def main():
    print(f"{'size':>5} {'method':>22} {'time_ms':>10} {'rel_err':>11} {'vs_scipy':>10}")
    print("-" * 62)
    for n in SIZES:
        A = spd(n)
        b = np.arange(1.0, n + 1.0)
        x_ref = np.linalg.solve(A, b)
        _, scipy_time = timed(lambda A=A, b=b: sla.solve(A, b, assume_a="pos"))

        for method in METHODS:
            x, t = timed(lambda A=A, b=b, m=method: solve_system(A, b, method=m))
            rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
            slowdown = t / scipy_time if scipy_time > 0 else float("inf")
            print(f"{n:>5} {method:>22} {t * 1e3:>10.3f} {rel_err:>11.2e} {slowdown:>9.1f}x")
        print(
            f"{n:>5} {'scipy.linalg.solve':>22} {scipy_time * 1e3:>10.3f} {'(reference)':>11} {'1.0x':>10}"
        )
        print("-" * 62)


if __name__ == "__main__":
    main()
