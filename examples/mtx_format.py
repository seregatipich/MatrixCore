"""Load and save matrices in Matrix Market (.mtx) format.

Run: python examples/mtx_format.py
"""

import tempfile
from pathlib import Path

import numpy as np

from matrixcore import load_matrix, save_matrix, solve_system


def main():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 4.0])

    workdir = Path(tempfile.mkdtemp())
    save_matrix(A, str(workdir / "A.mtx"))
    save_matrix(b, str(workdir / "b.mtx"))

    A_loaded = load_matrix(str(workdir / "A.mtx"))
    b_loaded = load_matrix(str(workdir / "b.mtx"))

    x = solve_system(A_loaded, b_loaded, method="lu_decomposition")
    print(f"Loaded A from {workdir / 'A.mtx'}")
    print(f"Solution: {x}  (expected [1, 1])")


if __name__ == "__main__":
    main()
