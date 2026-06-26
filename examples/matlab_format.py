"""Load and save matrices in MATLAB (.mat) format.

Run: python examples/matlab_format.py
"""

import tempfile
from pathlib import Path

import numpy as np

from matrixcore import load_matrix, save_matrix, solve_system


def main():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 4.0])

    workdir = Path(tempfile.mkdtemp())
    save_matrix(A, str(workdir / "system.mat"), variable_name="A")
    save_matrix(b, str(workdir / "rhs.mat"), variable_name="b")

    A_loaded = load_matrix(str(workdir / "system.mat"), variable_name="A")
    b_loaded = load_matrix(str(workdir / "rhs.mat"), variable_name="b")

    x = solve_system(A_loaded, b_loaded, method="cholesky")
    print(f"Loaded variable 'A' from {workdir / 'system.mat'}")
    print(f"Solution: {x}  (expected [1, 1])")


if __name__ == "__main__":
    main()
