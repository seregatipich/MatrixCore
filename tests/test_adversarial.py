"""Regression tests for adversarial inputs (issues found by external review).

These guard against solvers returning a wrong answer with success, and against
the Python layer accepting pathological input.
"""

import numpy as np
import pytest

from matrixcore import (
    InvalidParameterError,
    MatrixCoreError,
    NotSPDError,
    SingularMatrixError,
    solve_system,
)


def _solves(A, x, b):
    A = np.asarray(A, dtype=float)
    return np.linalg.norm(A @ x - np.asarray(b, dtype=float)) / max(np.linalg.norm(b), 1.0) < 1e-6


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_elimination",
        "lu_decomposition",
        "row_echelon",
        "reduced_row_echelon",
        "gauss_jordan",
        "triangularization",
        "cramers_rule",
        "matrix_inversion",
    ],
)
def test_rank_deficient_is_rejected(method):
    """A singular system must not be reported as a unique solution."""
    A = np.array([[0.0, 1.0], [0.0, 2.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(MatrixCoreError):
        solve_system(A, b, method=method)


@pytest.mark.parametrize(
    "method",
    [
        "minres",
        "eigenvalue_decomposition",
        "conjugate_gradient",
        "gradient_descent",
        "ldlt",
        "preconditioned_conjugate_gradient",
        "conjugate_residual",
        "symmlq",
        "chebyshev",
    ],
)
def test_non_symmetric_rejected_by_symmetric_solvers(method):
    A = np.array([[3.0, 1.0], [5.0, 3.0]])  # symmetric part is nonsingular
    b = np.array([1.0, 1.0])
    with pytest.raises(NotSPDError):
        solve_system(A, b, method=method)


@pytest.mark.parametrize("method", ["conjugate_gradient", "gradient_descent"])
def test_negative_definite_rejected(method):
    A = np.array([[-2.0, 0.0], [0.0, -3.0]])
    b = np.array([1.0, 1.0])
    with pytest.raises(NotSPDError):
        solve_system(A, b, method=method)


@pytest.mark.parametrize("method", ["block_matrix", "partitioning"])
def test_singular_leading_block_still_solves(method):
    """Nonsingular matrix with a singular leading block must still be solved."""
    A = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    b = np.array([1.0, 2.0, 3.0, 4.0])
    x = solve_system(A, b, method=method)
    assert _solves(A, x, b)
    assert np.allclose(x, [3.0, 4.0, 1.0, 2.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected(bad):
    A = np.array([[bad, 1.0], [1.0, 3.0]])
    with pytest.raises(InvalidParameterError):
        solve_system(A, np.array([1.0, 2.0]))


def test_empty_matrix_rejected():
    with pytest.raises(ValueError):
        solve_system(np.zeros((0, 0)), np.zeros(0))


def test_singular_via_dispatcher_sets_correct_exception():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(SingularMatrixError):
        solve_system(A, b, method="lu_decomposition")


@pytest.mark.parametrize("method", ["gmres", "svd", "pseudoinverse", "gaussian_elimination"])
def test_inconsistent_singular_not_false_success(method):
    """No exact solution exists; a solver must not report success with x=0."""
    A = np.zeros((2, 2))
    b = np.array([1.0, 0.0])
    with pytest.raises(MatrixCoreError):
        solve_system(A, b, method=method)


def test_method_none_uses_default():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 4.0])
    assert np.allclose(solve_system(A, b, method=None), [1.0, 1.0])


def test_method_bytes_accepted():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 4.0])
    assert np.allclose(solve_system(A, b, method=b"lu_decomposition"), [1.0, 1.0])


def test_method_wrong_type_raises():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 4.0])
    with pytest.raises(InvalidParameterError):
        solve_system(A, b, method=123)


@pytest.mark.parametrize(
    "method", ["qr_decomposition", "gradient_descent", "bicg", "lu_decomposition"]
)
def test_overflow_inputs_never_return_nan_success(method):
    """Finite but overflow-prone inputs must never yield a NaN 'success'."""
    A = 1e154 * np.eye(2)
    b = 1e154 * np.ones(2)
    try:
        x = solve_system(A, b, method=method)
    except MatrixCoreError:
        return  # rejecting is acceptable
    assert np.all(np.isfinite(x)), f"{method} returned non-finite x with success"
