import numpy as np

from matrixcore.solvers import (_matrix_diagnostics, list_available_solvers,
                                solve_system, recommend_solver)

A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
b = np.array([5.0, 4.0], dtype=np.float64)

print(f"Available solvers: {list_available_solvers()}\n\n")


diagnostics = _matrix_diagnostics(A)
print(diagnostics)

print(f"\nRecommended solver: {recommend_solver(A)}\n")

solution_method = "lu_decomposition"
x = solve_system(A, b, method=solution_method)
print("\nSolution using default method:")
print(x)
