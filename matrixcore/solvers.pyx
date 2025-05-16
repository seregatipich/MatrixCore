# distutils: language = c
# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np

cimport numpy as np

import logging

from libc.math cimport fabs, isinf, isnan

np.import_array()

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
    b : : numpy.ndarray
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

def recommend_solver(A):
    """
    Recommend the most appropriate solver based on matrix diagnostics.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix to analyze

    Returns
    -------
    str
        Recommended solver method name
    """
    # Get matrix diagnostics
    diag = _matrix_diagnostics(A)

    # For small matrices, direct methods are usually efficient
    if diag['m'] <= 100 and diag['n'] <= 100:
        if diag['m'] == diag['n']:  # Square matrix
            if diag['is_sym'] and diag['is_posdef']:
                return "cholesky"
            elif diag['is_tri_up']:
                return "back_substitution"
            elif diag['is_tri_lo']:
                return "forward_substitution"
            elif diag['rank_est'] < min(diag['m'], diag['n']):  # Rank deficient
                return "svd"
            else:
                return "lu_decomposition"
        else:  # Rectangular matrix
            if diag['m'] > diag['n']:  # Overdetermined
                return "qr_decomposition"
            else:  # Underdetermined
                return "svd"

    # For large matrices, consider sparsity and conditioning
    else:
        if diag['is_sparse']:
            if diag['is_sym'] and diag['is_posdef']:
                return "conjugate_gradient"
            elif diag['diag_dom']:
                return "jacobi"
            else:
                return "gmres"
        else:  # Dense large matrix
            if diag['is_sym'] and diag['is_posdef']:
                return "cholesky"
            elif diag['cond_est'] < 1e6:  # Well-conditioned
                return "lu_decomposition"
            else:  # Ill-conditioned
                return "qr_decomposition"

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

def _matrix_diagnostics(A):
    """
    Compute matrix diagnostics for solver selection.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix to analyze

    Returns
    -------
    dict
        Dictionary containing matrix diagnostics
    """
    # Initialize results dictionary with defaults
    diagnostics = {
        'm': 0, 'n': 0, 'nnz': 0, 'sparsity': 0.0,
        'is_sparse': False, 'is_sym': False,
        'is_tri_up': False, 'is_tri_lo': False,
        'diag_dom': False, 'cond_est': float('inf'),
        'is_posdef': False, 'rank_est': 0
    }

    # Basic matrix properties
    cdef int m, n, nnz
    cdef double sparsity
    cdef bint is_sparse = False

    try:
        m, n = A.shape
        diagnostics['m'] = m
        diagnostics['n'] = n

        # Handle sparse matrix vs dense array
        if hasattr(A, 'nnz'):  # Check for sparse matrix without importing scipy
            nnz = A.nnz
            is_sparse = True
        else:
            # For dense matrices, count non-zeros
            nnz = np.count_nonzero(A)
            is_sparse = False

        diagnostics['nnz'] = nnz

        # Calculate sparsity
        sparsity = nnz / (m * n) if m * n > 0 else 0.0
        diagnostics['sparsity'] = sparsity
        diagnostics['is_sparse'] = sparsity < 0.05

        # Check for symmetry (only square matrices can be symmetric)
        if m == n:
            _check_symmetry(A, diagnostics)

        # Check for triangular structure
        _check_triangular(A, diagnostics)

        # Check for diagonal dominance
        if m == n:
            _check_diag_dominance(A, diagnostics)

        # Estimate condition number
        if m == n:
            _estimate_condition(A, diagnostics)

        # Check for positive definiteness
        if m == n and diagnostics['is_sym']:
            _check_posdef(A, diagnostics)

        # Estimate rank
        _estimate_rank(A, diagnostics)

    except Exception:
        # Ensure we return all keys even if something fails
        pass

    return diagnostics

cdef void _check_symmetry(A, dict diagnostics):
    """Check if matrix is symmetric"""
    cdef int m = diagnostics['m']
    cdef double norm_diff, norm_A

    try:
        if hasattr(A, 'toarray'):  # Sparse matrix
            diff = A - A.T
            if hasattr(diff, 'sum'):
                norm_diff = abs(diff).sum()
                norm_A = abs(A).sum()
            else:
                # Fall back to numpy methods if sparse methods unavailable
                diff_arr = diff.toarray() if hasattr(diff, 'toarray') else diff
                norm_diff = np.abs(diff_arr).sum()
                A_arr = A.toarray() if hasattr(A, 'toarray') else A
                norm_A = np.abs(A_arr).sum()
        else:
            # Dense symmetry check
            diff = A - A.T
            norm_diff = np.linalg.norm(diff, ord=1)
            norm_A = np.linalg.norm(A, ord=1)

        diagnostics['is_sym'] = norm_diff <= 1e-12 * norm_A if norm_A > 0 else True
    except:
        pass

cdef void _check_triangular(A, dict diagnostics):
    """Check if matrix is triangular"""
    try:
        if hasattr(A, 'nonzero'):  # Sparse or dense matrix with nonzero method
            rows, cols = A.nonzero()
            if len(rows) > 0:
                diagnostics['is_tri_up'] = np.all(rows <= cols)
                diagnostics['is_tri_lo'] = np.all(rows >= cols)
        else:
            # Fallback
            indices = np.nonzero(A)
            if len(indices[0]) > 0:
                diagnostics['is_tri_up'] = np.all(indices[0] <= indices[1])
                diagnostics['is_tri_lo'] = np.all(indices[0] >= indices[1])
    except:
        pass

cdef void _check_diag_dominance(A, dict diagnostics):
    """Check if matrix is diagonally dominant"""
    cdef int i, m = diagnostics['m']
    cdef double row_sum
    cdef np.ndarray[double, ndim=1] diag

    try:
        if hasattr(A, 'diagonal'):  # Sparse matrix with diagonal method
            diag = A.diagonal()

            if hasattr(A, 'toarray'):  # Sparse matrix
                diag_dom = True
                for i in range(m):
                    row = A[i].toarray().flatten() if hasattr(A[i], 'toarray') else A[i]
                    row_sum = np.sum(np.abs(row)) - np.abs(row[i])
                    if np.abs(diag[i]) < row_sum:
                        diag_dom = False
                        break
                diagnostics['diag_dom'] = diag_dom
            else:  # Dense matrix
                diag = np.diag(A)
                row_sums = np.sum(np.abs(A), axis=1) - np.abs(diag)
                diagnostics['diag_dom'] = np.all(np.abs(diag) >= row_sums)
    except:
        pass

cdef void _estimate_condition(A, dict diagnostics):
    """Estimate condition number of matrix"""
    cdef int i, m = diagnostics['m'], n = diagnostics['n']
    cdef double sigma_max = 0, sigma_min_est = float('inf')
    cdef double norm_x, sigma_min_squared, alpha, beta

    try:
        if hasattr(A, 'todense') and m <= 2000:
            # For small sparse matrices, convert to dense for SVD
            s = np.linalg.svd(A.todense(), compute_uv=False)
            diagnostics['cond_est'] = s[0] / s[-1] if s[-1] > 0 else float('inf')
        elif not hasattr(A, 'todense') and m <= 2000:
            # Use SVD for small dense matrices
            s = np.linalg.svd(A, compute_uv=False)
            diagnostics['cond_est'] = s[0] / s[-1] if s[-1] > 0 else float('inf')
        else:
            # Power iteration for larger matrices
            AT = A.T.copy() if not hasattr(A, 'T') else A.T

            # Estimate largest singular value with power iteration
            x = np.random.randn(n)
            x = x / np.linalg.norm(x)

            for i in range(10):  # 10 iterations
                if hasattr(A, 'dot'):
                    y = A.dot(x)
                    x = AT.dot(y)
                else:
                    y = np.dot(A, x)
                    x = np.dot(AT, x)

                norm_x = np.linalg.norm(x)
                if norm_x > 0:
                    x = x / norm_x
                    sigma_max = norm_x

            # Simplified condition number estimation for performance
            # Use a heuristic instead of the full CG method
            diagnostics['cond_est'] = sigma_max * m  # Rough estimate
    except:
        pass

cdef void _check_posdef(A, dict diagnostics):
    """Check if matrix is positive definite"""
    cdef int m = diagnostics['m']

    try:
        if not hasattr(A, 'todense') and m <= 1000:
            # Try Cholesky for small dense matrices
            try:
                np.linalg.cholesky(A)
                diagnostics['is_posdef'] = True
            except np.linalg.LinAlgError:
                # Not positive definite
                pass
        elif not hasattr(A, 'todense') and m <= 2000:
            # Use eigenvalues for medium-sized dense matrices
            try:
                eigs = np.linalg.eigvalsh(A)
                diagnostics['is_posdef'] = np.all(eigs > -1e-10)
            except:
                pass
    except:
        pass

cdef void _estimate_rank(A, dict diagnostics):
    """Estimate matrix rank"""
    cdef int sketch_size, m = diagnostics['m'], n = diagnostics['n']
    cdef double tol

    try:
        sketch_size = min(50, min(m, n))

        if not hasattr(A, 'todense') and min(m, n) <= 1000:
            # For smaller dense matrices, use full SVD
            s = np.linalg.svd(A, compute_uv=False)
            tol = max(m, n) * np.finfo(float).eps * s[0]
            diagnostics['rank_est'] = np.sum(s > tol)
        else:
            # Random sketch approach
            Omega = np.random.randn(n, sketch_size)

            if hasattr(A, 'dot'):
                Y = A.dot(Omega)
            else:
                Y = np.dot(A, Omega)

            Q, R = np.linalg.qr(Y, mode='reduced')

            if hasattr(A, 'dot'):
                B = np.dot(Q.T, A)
            else:
                B = np.dot(Q.T, A)

            s = np.linalg.svd(B, compute_uv=False)
            tol = max(m, n) * np.finfo(float).eps * s[0]
            diagnostics['rank_est'] = np.sum(s > tol)
    except:
        pass

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
