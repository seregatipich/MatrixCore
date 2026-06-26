# MatrixCore

[![CI](https://github.com/seregatipich/MatrixCore/actions/workflows/ci.yml/badge.svg)](https://github.com/seregatipich/MatrixCore/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MatrixCore** is a Python library for solving dense systems of linear equations
`Ax = b`, backed by **30 solvers written from scratch in C**. The numerical core
depends only on the C standard library (`libm`); NumPy and SciPy are used purely
for the Python interface and matrix file I/O, never for the solve itself.

> **Positioning.** MatrixCore is an **educational / research** library: a clean,
> readable, dependency-free implementation of classic linear-algebra algorithms
> you can study and extend. It is *not* a replacement for the LAPACK-backed
> routines in `scipy.linalg` / `numpy.linalg` — for production numerical work,
> use those. MatrixCore is for learning how the algorithms actually work and for
> prototyping solver variants.

## Features

- Solve `Ax = b` for dense square matrices with 30 selectable algorithms
- A pure-C backend (direct, iterative, and decomposition-based methods)
- A clean Cython/NumPy interface with input coercion and typed exceptions
- Automatic solver recommendation from matrix diagnostics
- Read/write matrices in Matrix Market (`.mtx`), MATLAB (`.mat`), and
  Rutherford-Boeing (`.rb`) formats
- Typed (`py.typed` + stubs), linted (ruff), type-checked (mypy), and tested
  (pytest + Hypothesis) against NumPy ground truth

> Dense square systems only. Sparse solving is intentionally out of scope for
> now; sparse inputs loaded from file are densified before solving.

## Installation

MatrixCore is not yet published to PyPI. Install from source (a C compiler is
required to build the extension):

```bash
git clone https://github.com/seregatipich/MatrixCore.git
cd MatrixCore
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick start

```python
import numpy as np
from matrixcore import solve_system

A = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]])
b = np.array([7.0, 10.0, 9.0])

x = solve_system(A, b)                      # default: gaussian_elimination
print(x)

x = solve_system(A, b, method="cholesky")   # choose a specific method
x, info = solve_system(A, b, method="conjugate_gradient", return_info=True)
print(info)  # {'iterations': ..., 'residual': ..., 'error_code': 0}
```

Inputs are coerced automatically: Python lists, integer arrays, Fortran-order
arrays, and `scipy.sparse` matrices are all accepted and converted to a dense
C-contiguous `float64` array.

## Choosing a solver

```python
from matrixcore import recommend_solver, list_available_solvers

recommend_solver(A)        # e.g. 'cholesky' for an SPD matrix
list_available_solvers()   # all 30 method names
```

## Matrix file I/O

```python
from matrixcore import load_matrix, save_matrix, solve_system

# Format is inferred from the extension, or set it explicitly with format=.
save_matrix(A, "A.mtx")                       # Matrix Market
A = load_matrix("A.mtx")

save_matrix(A, "system.mat", variable_name="A")   # MATLAB
A = load_matrix("system.mat", variable_name="A")

save_matrix(A, "A.rb")                        # Rutherford-Boeing
A = load_matrix("A.rb")

x = solve_system(load_matrix("A.mtx"), load_matrix("b.mtx"))
```

## Error handling

Failures raise precise exceptions from `matrixcore.exceptions`, all subclasses of
`MatrixCoreError`:

```python
from matrixcore import solve_system, SingularMatrixError, NotSPDError

try:
    solve_system(singular_matrix, b)
except SingularMatrixError:
    ...
```

| Exception | Raised when |
|---|---|
| `SingularMatrixError` | matrix is singular / numerically singular |
| `NotSPDError` | matrix is not SPD for the chosen method (e.g. `cholesky`) |
| `ConvergenceError` | an iterative method did not converge |
| `InconsistentSystemError` | the system has no solution |
| `MultipleSolutionsError` | the system has infinitely many solutions |
| `InvalidParameterError` | invalid parameters or unknown method name |

## Available solvers

**Direct:** `gaussian_elimination`, `gauss_jordan`, `back_substitution`,
`forward_substitution`, `lu_decomposition`, `cholesky`, `qr_decomposition`,
`matrix_inversion`, `cramers_rule`, `row_echelon`, `reduced_row_echelon`,
`triangularization`

**Iterative:** `jacobi`, `gauss_seidel`, `sor`, `conjugate_gradient`,
`gradient_descent`, `minres`, `gmres`, `bicg`, `iterative_refinement`

**Specialized:** `normal_equations`, `orthogonal_projection`, `svd`,
`pseudoinverse`, `block_matrix`, `partitioning`, `matrix_rank`, `determinant`,
`eigenvalue_decomposition`

Some methods require a particular matrix structure (`cholesky`,
`conjugate_gradient`, `gradient_descent`, `minres`, and
`eigenvalue_decomposition` require a symmetric positive-definite matrix;
`back_substitution` / `forward_substitution` require a triangular matrix).

## Project structure

```text
MatrixCore/
├── matrixcore/
│   ├── __init__.py        # Public API exports
│   ├── solvers.pyx        # Cython interface to the C core
│   ├── solvers.pyi        # Type stubs for the compiled module
│   ├── exceptions.py      # Exception hierarchy + error-code mapping
│   └── io/                # mtx / matlab / rb readers and writers
├── src/
│   ├── solvers.c          # 30 solvers, written from scratch in C
│   └── solvers.h          # C API and error codes
├── tests/                 # pytest + Hypothesis suite
├── examples/              # runnable usage examples
├── benchmarks/            # accuracy/speed comparison vs scipy
├── setup.py               # Cython extension build
└── pyproject.toml         # packaging + tool configuration
```

## Development

```bash
pip install -e ".[test,dev]"
pytest                     # run the test suite
pytest --cov=matrixcore    # with coverage
ruff check . && ruff format --check .
mypy matrixcore
```

## License

MIT — see [LICENSE](LICENSE).
