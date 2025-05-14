# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as np

# Import the solver information struct and functions from solvers.h
cdef extern from "solvers.h":
    # Define the solver_info struct
    ctypedef struct solver_info:
        int iterations
        double residual
        int error_code

    # Import all solver functions
    int gaussian_elimination(double *A, double *b, double *x, int n, solver_info *info)
    int gauss_jordan(double *A, double *b, double *x, int n, solver_info *info)
    int back_substitution(double *A, double *b, double *x, int n, solver_info *info)
    int forward_substitution(double *A, double *b, double *x, int n, solver_info *info)
    int lu_decomposition(double *A, double *b, double *x, int n, solver_info *info)
    int cholesky(double *A, double *b, double *x, int n, solver_info *info)
    int qr_decomposition(double *A, double *b, double *x, int n, solver_info *info)
    int matrix_inversion(double *A, double *b, double *x, int n, solver_info *info)
    int cramers_rule(double *A, double *b, double *x, int n, solver_info *info)
    int row_echelon(double *A, double *b, double *x, int n, solver_info *info)
    int reduced_row_echelon(double *A, double *b, double *x, int n, solver_info *info)
    int triangularization(double *A, double *b, double *x, int n, solver_info *info)
    int jacobi(double *A, double *b, double *x, int n, solver_info *info)
    int gauss_seidel(double *A, double *b, double *x, int n, solver_info *info)
    int sor(double *A, double *b, double *x, int n, solver_info *info)
    int conjugate_gradient(double *A, double *b, double *x, int n, solver_info *info)
    int gradient_descent(double *A, double *b, double *x, int n, solver_info *info)
    int minres(double *A, double *b, double *x, int n, solver_info *info)
    int gmres(double *A, double *b, double *x, int n, solver_info *info)
    int bicg(double *A, double *b, double *x, int n, solver_info *info)
    int iterative_refinement(double *A, double *b, double *x, int n, solver_info *info)
    int normal_equations(double *A, double *b, double *x, int n, solver_info *info)
    int orthogonal_projection(double *A, double *b, double *x, int n, solver_info *info)
    int svd(double *A, double *b, double *x, int n, solver_info *info)
    int pseudoinverse(double *A, double *b, double *x, int n, solver_info *info)
    int block_matrix(double *A, double *b, double *x, int n, solver_info *info)
    int partitioning(double *A, double *b, double *x, int n, solver_info *info)
    int matrix_rank(double *A, double *b, double *x, int n, solver_info *info)
    int determinant(double *A, double *b, double *x, int n, solver_info *info)
    int eigenvalue_decomposition(double *A, double *b, double *x, int n, solver_info *info)

    # Import the main solver function
    int solve_linear_system(double *A, double *b, double *x, int n, const char *method, solver_info *info)

def solve_system(np.ndarray[double, ndim=2, mode="c"] A,
                 np.ndarray[double, ndim=1, mode="c"] b,
                 method='gaussian_elimination',
                 return_info=False):
    """
    Solve a linear system Ax = b using the specified method.

    Parameters
    ----------
    A : numpy.ndarray
        Coefficient matrix (n x n)
    b : numpy.ndarray
        Right-hand side vector (n)
    method : str, optional
        Solver method to use, default is 'gaussian_elimination'
    return_info : bool, optional
        Whether to return solver information, default is False

    Returns
    -------
    numpy.ndarray or tuple
        Solution vector x if return_info=False, otherwise a tuple (x, info_dict)
        where info_dict contains solver information
    """
    # Check input dimensions
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Dimension mismatch: A is {A.shape[0]}x{A.shape[1]} but b has length {b.shape[0]}")

    # Get problem size
    cdef int n = A.shape[0]

    # Create output array for the solution
    cdef np.ndarray[double, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

    # Create solver_info struct to collect information
    cdef solver_info info

    # Convert method string to bytes
    cdef bytes method_bytes = method.encode('utf-8')

    # Call the C solver function
    cdef int result = solve_linear_system(&A[0, 0], &b[0], &x[0], n, method_bytes, &info)

    # Check for errors
    if result != 0:
        if info.error_code == 1:
            raise ValueError("Matrix is singular or nearly singular")
        elif info.error_code == 2:
            raise ValueError("Method not implemented or not applicable")
        elif info.error_code == 3:
            raise ValueError("Iterative method failed to converge")
        else:
            raise RuntimeError(f"Solver failed with error code {info.error_code}")

    # Return result based on return_info flag
    if return_info:
        info_dict = {
            'iterations': info.iterations,
            'residual': info.residual,
            'error_code': info.error_code
        }
        return x, info_dict
    else:
        return x

def list_available_solvers():
    """
    Returns a list of all available solver methods.

    Returns
    -------
    list
        List of available solver method names as strings
    """
    return [
        'gaussian_elimination',
        'gauss_jordan',
        'back_substitution',
        'forward_substitution',
        'lu_decomposition',
        'cholesky',
        'qr_decomposition',
        'matrix_inversion',
        'cramers_rule',
        'row_echelon',
        'reduced_row_echelon',
        'triangularization',
        'jacobi',
        'gauss_seidel',
        'sor',
        'conjugate_gradient',
        'gradient_descent',
        'minres',
        'gmres',
        'bicg',
        'iterative_refinement',
        'normal_equations',
        'orthogonal_projection',
        'svd',
        'pseudoinverse',
        'block_matrix',
        'partitioning',
        'matrix_rank',
        'determinant',
        'eigenvalue_decomposition'
    ]

def gaussian_elimination_solver(np.ndarray[double, ndim=2, mode="c"] A,
                                np.ndarray[double, ndim=1, mode="c"] b,
                                return_info=False):
    """Solve a linear system using Gaussian Elimination."""
    return solve_system(A, b, method='gaussian_elimination', return_info=return_info)

def gauss_jordan_solver(np.ndarray[double, ndim=2, mode="c"] A,
                        np.ndarray[double, ndim=1, mode="c"] b,
                        return_info=False):
    """Solve a linear system using Gauss-Jordan elimination."""
    return solve_system(A, b, method='gauss_jordan', return_info=return_info)

def back_substitution_solver(np.ndarray[double, ndim=2, mode="c"] A,
                             np.ndarray[double, ndim=1, mode="c"] b,
                             return_info=False):
    """Solve a linear system using Back Substitution."""
    return solve_system(A, b, method='back_substitution', return_info=return_info)

def forward_substitution_solver(np.ndarray[double, ndim=2, mode="c"] A,
                                np.ndarray[double, ndim=1, mode="c"] b,
                                return_info=False):
    """Solve a linear system using Forward Substitution."""
    return solve_system(A, b, method='forward_substitution', return_info=return_info)

def lu_decomposition_solver(np.ndarray[double, ndim=2, mode="c"] A,
                            np.ndarray[double, ndim=1, mode="c"] b,
                            return_info=False):
    """Solve a linear system using LU Decomposition."""
    return solve_system(A, b, method='lu_decomposition', return_info=return_info)

def cholesky_solver(np.ndarray[double, ndim=2, mode="c"] A,
                    np.ndarray[double, ndim=1, mode="c"] b,
                    return_info=False):
    """Solve a linear system using Cholesky Decomposition."""
    return solve_system(A, b, method='cholesky', return_info=return_info)

def qr_decomposition_solver(np.ndarray[double, ndim=2, mode="c"] A,
                            np.ndarray[double, ndim=1, mode="c"] b,
                            return_info=False):
    """Solve a linear system using QR Decomposition."""
    return solve_system(A, b, method='qr_decomposition', return_info=return_info)

def matrix_inversion_solver(np.ndarray[double, ndim=2, mode="c"] A,
                            np.ndarray[double, ndim=1, mode="c"] b,
                            return_info=False):
    """Solve a linear system using Matrix Inversion."""
    return solve_system(A, b, method='matrix_inversion', return_info=return_info)

def cramers_rule_solver(np.ndarray[double, ndim=2, mode="c"] A,
                        np.ndarray[double, ndim=1, mode="c"] b,
                        return_info=False):
    """Solve a linear system using Cramer's Rule."""
    return solve_system(A, b, method='cramers_rule', return_info=return_info)

def row_echelon_solver(np.ndarray[double, ndim=2, mode="c"] A,
                       np.ndarray[double, ndim=1, mode="c"] b,
                       return_info=False):
    """Solve a linear system using Row Echelon Form."""
    return solve_system(A, b, method='row_echelon', return_info=return_info)

def reduced_row_echelon_solver(np.ndarray[double, ndim=2, mode="c"] A,
                               np.ndarray[double, ndim=1, mode="c"] b,
                               return_info=False):
    """Solve a linear system using Reduced Row Echelon Form."""
    return solve_system(A, b, method='reduced_row_echelon', return_info=return_info)

def triangularization_solver(np.ndarray[double, ndim=2, mode="c"] A,
                             np.ndarray[double, ndim=1, mode="c"] b,
                             return_info=False):
    """Solve a linear system using Triangularization."""
    return solve_system(A, b, method='triangularization', return_info=return_info)

def jacobi_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Jacobi iterative method."""
    return solve_system(A, b, method='jacobi', return_info=return_info)

def gauss_seidel_solver(np.ndarray[double, ndim=2, mode="c"] A,
                        np.ndarray[double, ndim=1, mode="c"] b,
                        return_info=False):
    """Solve a linear system using Gauss-Seidel iterative method."""
    return solve_system(A, b, method='gauss_seidel', return_info=return_info)

def sor_solver(np.ndarray[double, ndim=2, mode="c"] A,
               np.ndarray[double, ndim=1, mode="c"] b,
               return_info=False):
    """Solve a linear system using Successive Over-Relaxation (SOR) method."""
    return solve_system(A, b, method='sor', return_info=return_info)

def conjugate_gradient_solver(np.ndarray[double, ndim=2, mode="c"] A,
                              np.ndarray[double, ndim=1, mode="c"] b,
                              return_info=False):
    """Solve a linear system using Conjugate Gradient method."""
    return solve_system(A, b, method='conjugate_gradient', return_info=return_info)

def gradient_descent_solver(np.ndarray[double, ndim=2, mode="c"] A,
                            np.ndarray[double, ndim=1, mode="c"] b,
                            return_info=False):
    """Solve a linear system using Gradient Descent method."""
    return solve_system(A, b, method='gradient_descent', return_info=return_info)

def minres_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Minimal Residual method (MINRES)."""
    return solve_system(A, b, method='minres', return_info=return_info)

def gmres_solver(np.ndarray[double, ndim=2, mode="c"] A,
                 np.ndarray[double, ndim=1, mode="c"] b,
                 return_info=False):
    """Solve a linear system using Generalized Minimal Residual method (GMRES)."""
    return solve_system(A, b, method='gmres', return_info=return_info)

def bicg_solver(np.ndarray[double, ndim=2, mode="c"] A,
                np.ndarray[double, ndim=1, mode="c"] b,
                return_info=False):
    """Solve a linear system using Biconjugate Gradient method."""
    return solve_system(A, b, method='bicg', return_info=return_info)

def iterative_refinement_solver(np.ndarray[double, ndim=2, mode="c"] A,
                                np.ndarray[double, ndim=1, mode="c"] b,
                                return_info=False):
    """Solve a linear system using Iterative Refinement method."""
    return solve_system(A, b, method='iterative_refinement', return_info=return_info)

def normal_equations_solver(np.ndarray[double, ndim=2, mode="c"] A,
                            np.ndarray[double, ndim=1, mode="c"] b,
                            return_info=False):
    """Solve a linear system using Normal Equations method."""
    return solve_system(A, b, method='normal_equations', return_info=return_info)

def orthogonal_projection_solver(np.ndarray[double, ndim=2, mode="c"] A,
                                 np.ndarray[double, ndim=1, mode="c"] b,
                                 return_info=False):
    """Solve a linear system using Orthogonal Projection method."""
    return solve_system(A, b, method='orthogonal_projection', return_info=return_info)

def svd_solver(np.ndarray[double, ndim=2, mode="c"] A,
               np.ndarray[double, ndim=1, mode="c"] b,
               return_info=False):
    """Solve a linear system using Singular Value Decomposition (SVD)."""
    return solve_system(A, b, method='svd', return_info=return_info)

def pseudoinverse_solver(np.ndarray[double, ndim=2, mode="c"] A,
                         np.ndarray[double, ndim=1, mode="c"] b,
                         return_info=False):
    """Solve a linear system using Pseudoinverse method."""
    return solve_system(A, b, method='pseudoinverse', return_info=return_info)

def block_matrix_solver(np.ndarray[double, ndim=2, mode="c"] A,
                        np.ndarray[double, ndim=1, mode="c"] b,
                        return_info=False):
    """Solve a linear system using Block Matrix method."""
    return solve_system(A, b, method='block_matrix', return_info=return_info)

def partitioning_solver(np.ndarray[double, ndim=2, mode="c"] A,
                        np.ndarray[double, ndim=1, mode="c"] b,
                        return_info=False):
    """Solve a linear system using Partitioning method."""
    return solve_system(A, b, method='partitioning', return_info=return_info)

def matrix_rank_solver(np.ndarray[double, ndim=2, mode="c"] A,
                       np.ndarray[double, ndim=1, mode="c"] b,
                       return_info=False):
    """Solve a linear system using Matrix Rank method."""
    return solve_system(A, b, method='matrix_rank', return_info=return_info)

def determinant_solver(np.ndarray[double, ndim=2, mode="c"] A,
                       np.ndarray[double, ndim=1, mode="c"] b,
                       return_info=False):
    """Solve a linear system using Determinant method."""
    return solve_system(A, b, method='determinant', return_info=return_info)

def eigenvalue_decomposition_solver(np.ndarray[double, ndim=2, mode="c"] A,
                                    np.ndarray[double, ndim=1, mode="c"] b,
                                    return_info=False):
    """Solve a linear system using Eigenvalue Decomposition method."""
    return solve_system(A, b, method='eigenvalue_decomposition', return_info=return_info)
