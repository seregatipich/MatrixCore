# MatrixCore

**MatrixCore** is a high-performance Python library for solving systems of linear equations using a fast C backend. Designed for speed and precision, MatrixCore gives you the power of native C solvers with the simplicity of a Pythonic interface.

## 🚀 Features

- 🔧 Solve systems of linear equations `Ax = b`
- ⚡ Powered by a C implementation for maximum speed
- 🧠 Clean Python API via Cython
- ✅ Supports dense square matrices (sparse support coming soon)
- 📊 Supports multiple matrix file formats:
  - MATLAB (.mat)
  - Rutherford-Boeing (.rb)
  - Matrix Market (.mtx)
- 🔬 Accurate and efficient for educational and research use

## 📦 Installation

To install the released version from PyPI:
```bash
  pip install matrixcore
```


For development installation from source:
```bash
  git clone [https://github.com/seregatipich/matrixcore.git](https://github.com/seregatipich/matrixcore.git) cd matrixcore pip install -e .
```

## 🧮 How It Works

MatrixCore uses a highly optimized C backend to perform linear algebra operations. The library implements 30 different algorithms for solving linear systems, giving you unprecedented flexibility to choose the right solver for your specific use case.

The Cython layer provides a seamless interface between Python and the C code, ensuring both high performance and ease of use. Whether you're working with dense or sparse matrices, symmetric or non-symmetric systems, MatrixCore offers the right solver for your needs. See the Available Solvers section below for a complete list of implemented methods.

## 🧩 Available Solvers

MatrixCore offers a comprehensive suite of solvers for linear equation systems:

- **Direct Methods**
  - Gaussian Elimination (`'gaussian_elimination'`)
  - Gauss-Jordan Elimination (`'gauss_jordan'`)
  - Back Substitution (`'back_substitution'`)
  - Forward Substitution (`'forward_substitution'`)
  - LU Decomposition (`'lu_decomposition'`)
  - Cholesky Decomposition (`'cholesky'`)
  - QR Decomposition (`'qr_decomposition'`)
  - Matrix Inversion Method (`'matrix_inversion'`)
  - Cramer's Rule (`'cramers_rule'`)
  - Row Reduction to Row-Echelon Form (`'row_echelon'`)
  - Row Reduction to Reduced Row-Echelon Form (`'reduced_row_echelon'`)
  - Triangularization (`'triangularization'`)

- **Iterative Methods**
  - Jacobi Iterative Method (`'jacobi'`)
  - Gauss-Seidel Method (`'gauss_seidel'`)
  - Successive Over-Relaxation (`'sor'`)
  - Conjugate Gradient Method (`'conjugate_gradient'`)
  - Gradient Descent Method (`'gradient_descent'`)
  - Minimal Residual Method (`'minres'`)
  - Generalized Minimal Residual Method (`'gmres'`)
  - Biconjugate Gradient Method (`'bicg'`)
  - Iterative Refinement (`'iterative_refinement'`)

- **Specialized Methods**
  - Normal Equations for Least Squares (`'normal_equations'`)
  - Orthogonal Projection Method (`'orthogonal_projection'`)
  - SVD (Singular Value Decomposition) (`'svd'`)
  - Pseudoinverse Method (`'pseudoinverse'`)
  - Block Matrix Method (`'block_matrix'`)
  - Partitioning Method (`'partitioning'`)
  - Matrix Rank Method (`'matrix_rank'`)
  - Determinant Method (`'determinant'`)
  - Matrix Eigenvalue Decomposition (`'eigenvalue_decomposition'`)

## 📝 Usage Examples

### Basic Solving

```python
import numpy as np from matrixcore import solve_system

# Create a system Ax = b
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=np.float64)
b = np.array([7, 10, 9], dtype=np.float64)

# Solve the system
x = solve_system(A, b)
print(f"Solution: {x}")
```


### Loading and Solving from Matrix Files

```python
from matrixcore import load_matrix, solve_system

# Load matrix from MATLAB format
A = load_matrix("matrix.mat", format="matlab", variable_name="A")
b = load_matrix("matrix.mat", format="matlab", variable_name="b")

# Load matrix from Rutherford-Boeing format
A = load_matrix("matrix.rb", format="rb")

# Load matrix from Matrix Market format
A = load_matrix("matrix.mtx", format="mtx")
b = load_matrix("vector.mtx", format="mtx")

# Solve the system
x = solve_system(A, b)
```


### Saving Matrices to Different Formats

```python
from matrixcore import save_matrix

# Save matrix to MATLAB format
save_matrix(A, "output.mat", format="matlab", variable_name="A")

# Save matrix to Rutherford-Boeing format
save_matrix(A, "output.rb", format="rb")

# Save matrix to Matrix Market format
save_matrix(A, "output.mtx", format="mtx")
```

### Advanced Options

```python
from matrixcore import solve_system, list_available_solvers

# List all available solvers
solvers = list_available_solvers()
print(f"Available solvers: {solvers}")

# Specify the algorithm to use
x = solve_system(A, b, method='conjugate_gradient')

# Try different solvers for performance comparison
x_gauss = solve_system(A, b, method='gaussian_elimination')
x_lu = solve_system(A, b, method='lu_decomposition')
x_cholesky = solve_system(A, b, method='cholesky_decomposition')

# Get additional information about the solution
x, info = solve_system(A, b, method='gmres', return_info=True)
print(f"Solution: {x}")
print(f"Condition number: {info['condition_number']}")
print(f"Iterations: {info['iterations']}")
print(f"Convergence rate: {info['convergence_rate']}")
```

## 📊 Project Structure

```text
matrixcore/
├── matrixcore/           # Python wrapper
│   ├── __init__.py       # Package initialization and API exports
│   ├── solver.pyx        # Cython interface to C implementation
│   ├── io/               # Matrix I/O functionality
│   │   ├── __init__.py   
│   │   ├── matlab.py     # MATLAB format support
│   │   ├── rb.py         # Rutherford-Boeing format support
│   │   └── mtx.py        # Matrix Market format support
├── src/                  # C source code
│   ├── solver.c          # C implementation of matrix solvers
│   └── solver.h          # C header file with function declarations
├── tests/                # Test suite
│   ├── test_solver.py    # Test cases for solver functionality
│   ├── test_io.py        # Test cases for matrix I/O functionality
│   └── benchmark.py      # Performance benchmarks
├── examples/             # Example usage scripts
│   ├── basic_solving.py
│   ├── matlab_format.py
│   ├── rb_format.py
│   └── mtx_format.py
├── setup.py              # Build configuration
├── pyproject.toml        # Python project metadata
└── README.md             # Project documentation
```