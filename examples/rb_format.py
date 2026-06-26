"""Load and save matrices in Rutherford-Boeing (.rb) format.

Run: python examples/rb_format.py
"""

import tempfile
from pathlib import Path

import numpy as np

from matrixcore import load_matrix, save_matrix, solve_system


def main():
    A = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0])

    workdir = Path(tempfile.mkdtemp())
    save_matrix(A, str(workdir / "A.rb"))
    save_matrix(b, str(workdir / "b.rb"))

    A_loaded = load_matrix(str(workdir / "A.rb"))
    b_loaded = load_matrix(str(workdir / "b.rb"))

    x = solve_system(A_loaded, b_loaded, method="qr_decomposition")
    print(f"Loaded A from {workdir / 'A.rb'}")
    print(f"Solution: {np.round(x, 6)}")
    print(f"Residual ||Ax - b||: {np.linalg.norm(A_loaded @ x - b_loaded.ravel()):.2e}")


if __name__ == "__main__":
    main()
