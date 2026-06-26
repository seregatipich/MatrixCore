"""Solve a dense linear system Ax = b with MatrixCore.

Run: python examples/basic_solving.py
"""

import numpy as np

from matrixcore import list_available_solvers, recommend_solver, solve_system


def main():
    A = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0])

    print(f"{len(list_available_solvers())} solvers available")
    print(f"Recommended solver for this matrix: {recommend_solver(A)}")

    x = solve_system(A, b)
    print(f"Default (gaussian_elimination) solution: {x}")

    for method in ("lu_decomposition", "cholesky", "conjugate_gradient"):
        x, info = solve_system(A, b, method=method, return_info=True)
        print(
            f"{method:20s} x={np.round(x, 6)}  iters={info['iterations']}  residual={info['residual']:.2e}"
        )


if __name__ == "__main__":
    main()
