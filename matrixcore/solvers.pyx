# distutils: language = c
# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np

cimport numpy as np

from matrixcore.exceptions import InvalidParameterError, error_for_code

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
    int crout(double *A, double *b, double *x, int n, solver_info *info)
    int ldlt(double *A, double *b, double *x, int n, solver_info *info)
    int thomas(double *A, double *b, double *x, int n, solver_info *info)
    int givens_qr(double *A, double *b, double *x, int n, solver_info *info)
    int modified_gram_schmidt(double *A, double *b, double *x, int n, solver_info *info)
    int classical_gram_schmidt(double *A, double *b, double *x, int n, solver_info *info)
    int lq_decomposition(double *A, double *b, double *x, int n, solver_info *info)
    int richardson(double *A, double *b, double *x, int n, solver_info *info)
    int ssor(double *A, double *b, double *x, int n, solver_info *info)
    int bicgstab(double *A, double *b, double *x, int n, solver_info *info)
    int cgs(double *A, double *b, double *x, int n, solver_info *info)
    int tfqmr(double *A, double *b, double *x, int n, solver_info *info)
    int qmr(double *A, double *b, double *x, int n, solver_info *info)
    int gcr(double *A, double *b, double *x, int n, solver_info *info)
    int lsqr(double *A, double *b, double *x, int n, solver_info *info)
    int cgnr(double *A, double *b, double *x, int n, solver_info *info)
    int preconditioned_conjugate_gradient(double *A, double *b, double *x, int n, solver_info *info)
    int conjugate_residual(double *A, double *b, double *x, int n, solver_info *info)
    int symmlq(double *A, double *b, double *x, int n, solver_info *info)
    int chebyshev(double *A, double *b, double *x, int n, solver_info *info)

    # Import the main solver function
    int solve_linear_system(double *A, double *b, double *x, int n, const char *method, solver_info *info) nogil

def _as_dense_f64(array, name):
    """Coerce input to a contiguous float64 numpy array, densifying sparse input."""
    if hasattr(array, "toarray"):  # scipy.sparse matrix
        array = array.toarray()
    return np.ascontiguousarray(array, dtype=np.float64)


def solve_system(A, b, method='gaussian_elimination', return_info=False):
    """
    Solve a linear system Ax = b using the specified method.

    Parameters
    ----------
    A : array_like
        Coefficient matrix (n x n). Non-contiguous, non-float64, or scipy.sparse
        inputs are converted to a dense C-contiguous float64 array.
    b : array_like
        Right-hand side vector (n).
    method : str, optional
        Solver method to use, default is 'gaussian_elimination'. See
        :func:`list_available_solvers` for the full list.
    return_info : bool, optional
        Whether to return solver information, default is False.

    Returns
    -------
    numpy.ndarray or tuple
        Solution vector x if return_info=False, otherwise a tuple (x, info_dict)
        where info_dict contains 'iterations', 'residual' and 'error_code'.

    Raises
    ------
    SingularMatrixError, NotSPDError, ConvergenceError, InvalidParameterError
        Subclasses of :class:`matrixcore.exceptions.MatrixCoreError` describing
        the precise failure reported by the C core.
    """
    cdef np.ndarray[double, ndim=2, mode="c"] A_c = _as_dense_f64(A, "A")
    b_dense = np.atleast_1d(np.squeeze(_as_dense_f64(b, "b")))
    cdef np.ndarray[double, ndim=1, mode="c"] b_c = np.ascontiguousarray(b_dense, dtype=np.float64)

    if A_c.shape[0] != A_c.shape[1]:
        raise ValueError(f"Matrix A must be square, got shape {A_c.shape[0]}x{A_c.shape[1]}")
    if A_c.shape[0] != b_c.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A is {A_c.shape[0]}x{A_c.shape[1]} but b has length {b_c.shape[0]}"
        )
    if A_c.shape[0] == 0:
        raise ValueError("Matrix A must have at least one row")
    if not np.isfinite(A_c).all() or not np.isfinite(b_c).all():
        raise InvalidParameterError("inputs must contain only finite values (no NaN or inf)")

    if method is None:
        method = 'gaussian_elimination'
    elif isinstance(method, bytes):
        method = method.decode('utf-8')
    elif not isinstance(method, str):
        raise InvalidParameterError(f"method must be a string, got {type(method).__name__}")

    cdef int n = A_c.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)
    cdef solver_info info
    cdef bytes method_bytes = method.encode('utf-8')
    cdef const char* method_ptr = method_bytes
    cdef double* A_ptr = &A_c[0, 0]
    cdef double* b_ptr = &b_c[0]
    cdef double* x_ptr = &x[0]
    cdef int result

    with nogil:
        result = solve_linear_system(A_ptr, b_ptr, x_ptr, n, method_ptr, &info)

    # The C return value is the authoritative error code (info may be unset on
    # some failure paths), so map exceptions from it rather than info.error_code.
    if result != 0:
        raise error_for_code(result, method=method)

    if return_info:
        info_dict = {
            'iterations': info.iterations,
            'residual': info.residual,
            'error_code': info.error_code,
        }
        return x, info_dict
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
        'eigenvalue_decomposition',
        'crout',
        'ldlt',
        'thomas',
        'givens_qr',
        'modified_gram_schmidt',
        'classical_gram_schmidt',
        'lq_decomposition',
        'richardson',
        'ssor',
        'bicgstab',
        'cgs',
        'tfqmr',
        'qmr',
        'gcr',
        'lsqr',
        'cgnr',
        'preconditioned_conjugate_gradient',
        'conjugate_residual',
        'symmlq',
        'chebyshev',
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

def crout_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Crout LU Decomposition."""
    return solve_system(A, b, method='crout', return_info=return_info)

def ldlt_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using LDL^T Decomposition."""
    return solve_system(A, b, method='ldlt', return_info=return_info)

def thomas_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Thomas Algorithm (tridiagonal)."""
    return solve_system(A, b, method='thomas', return_info=return_info)

def givens_qr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using QR Decomposition via Givens Rotations."""
    return solve_system(A, b, method='givens_qr', return_info=return_info)

def modified_gram_schmidt_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Modified Gram-Schmidt QR."""
    return solve_system(A, b, method='modified_gram_schmidt', return_info=return_info)

def classical_gram_schmidt_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Classical Gram-Schmidt QR."""
    return solve_system(A, b, method='classical_gram_schmidt', return_info=return_info)

def lq_decomposition_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using LQ Decomposition."""
    return solve_system(A, b, method='lq_decomposition', return_info=return_info)

def richardson_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Richardson Iteration."""
    return solve_system(A, b, method='richardson', return_info=return_info)

def ssor_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Symmetric Successive Over-Relaxation (SSOR)."""
    return solve_system(A, b, method='ssor', return_info=return_info)

def bicgstab_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using BiCGSTAB."""
    return solve_system(A, b, method='bicgstab', return_info=return_info)

def cgs_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Conjugate Gradient Squared (CGS)."""
    return solve_system(A, b, method='cgs', return_info=return_info)

def tfqmr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Transpose-Free QMR (TFQMR)."""
    return solve_system(A, b, method='tfqmr', return_info=return_info)

def qmr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Quasi-Minimal Residual (QMR)."""
    return solve_system(A, b, method='qmr', return_info=return_info)

def gcr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Generalized Conjugate Residual (GCR)."""
    return solve_system(A, b, method='gcr', return_info=return_info)

def lsqr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using LSQR."""
    return solve_system(A, b, method='lsqr', return_info=return_info)

def cgnr_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Conjugate Gradient on Normal Equations (CGNR)."""
    return solve_system(A, b, method='cgnr', return_info=return_info)

def preconditioned_conjugate_gradient_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Preconditioned Conjugate Gradient (Jacobi)."""
    return solve_system(A, b, method='preconditioned_conjugate_gradient', return_info=return_info)

def conjugate_residual_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Conjugate Residual."""
    return solve_system(A, b, method='conjugate_residual', return_info=return_info)

def symmlq_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using SYMMLQ."""
    return solve_system(A, b, method='symmlq', return_info=return_info)

def chebyshev_solver(np.ndarray[double, ndim=2, mode="c"] A,
                  np.ndarray[double, ndim=1, mode="c"] b,
                  return_info=False):
    """Solve a linear system using Chebyshev Semi-Iteration."""
    return solve_system(A, b, method='chebyshev', return_info=return_info)
