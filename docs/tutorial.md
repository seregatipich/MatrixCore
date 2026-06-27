# Tutorial

## Solving a system

```python
import numpy as np
from matrixcore import solve_system

A = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]])
b = np.array([7.0, 10.0, 9.0])

x = solve_system(A, b)                       # default gaussian_elimination
x = solve_system(A, b, method="lu_decomposition")
x, info = solve_system(A, b, method="conjugate_gradient", return_info=True)
```

`info` is a dict with `iterations`, `residual`, and `error_code`.

## Picking a method

```python
from matrixcore import recommend_solver, list_available_solvers

recommend_solver(A)        # heuristic choice from matrix diagnostics
list_available_solvers()   # all 50 method names
```

Some methods require structure: `cholesky`, `conjugate_gradient`,
`gradient_descent`, `minres`, and `eigenvalue_decomposition` expect a symmetric
positive-definite matrix; `back_substitution` and `forward_substitution` expect a
triangular matrix.

## Reading and writing matrices

```python
from matrixcore import load_matrix, save_matrix

save_matrix(A, "A.mtx")                            # Matrix Market
save_matrix(A, "system.mat", variable_name="A")    # MATLAB
save_matrix(A, "A.rb")                             # Rutherford-Boeing

A = load_matrix("A.mtx")                            # format inferred from extension
A = load_matrix("system.mat", variable_name="A")
```

All readers return a dense `float64` array ready for `solve_system`.

## Handling errors

```python
from matrixcore import solve_system, SingularMatrixError, NotSPDError, ConvergenceError

try:
    x = solve_system(A, b, method="cholesky")
except NotSPDError:
    x = solve_system(A, b, method="lu_decomposition")
```
