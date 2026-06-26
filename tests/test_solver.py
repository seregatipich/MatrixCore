"""Correctness and contract tests for the solver layer."""

import numpy as np
import pytest

from matrixcore import (
    ConvergenceError,
    InvalidParameterError,
    NotSPDError,
    SingularMatrixError,
    list_available_solvers,
    solve_system,
)
from tests.conftest import REL_TOL, spd_matrix, systems_for

ALL_SOLVERS = list_available_solvers()


def test_thirty_solvers_available():
    assert len(ALL_SOLVERS) == 30
    assert len(set(ALL_SOLVERS)) == 30


@pytest.mark.parametrize("method", ALL_SOLVERS)
def test_solver_matches_numpy(method):
    """Every solver reproduces numpy.linalg.solve on systems it should handle."""
    for A, b in systems_for(method):
        x_ref = np.linalg.solve(A, b)
        x = solve_system(A.copy(), b.copy(), method=method)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        assert np.isfinite(rel_err)
        assert rel_err < REL_TOL, f"{method}: rel_err={rel_err:.2e}"


@pytest.mark.parametrize("method", ALL_SOLVERS)
def test_solver_does_not_mutate_inputs(method):
    for A, b in systems_for(method):
        A0, b0 = A.copy(), b.copy()
        solve_system(A, b, method=method)
        assert np.array_equal(A, A0), f"{method} mutated A"
        assert np.array_equal(b, b0), f"{method} mutated b"


@pytest.mark.parametrize("method", ALL_SOLVERS)
def test_return_info_contract(method):
    A, b = systems_for(method)[0]
    x, info = solve_system(A, b, method=method, return_info=True)
    assert set(info) == {"iterations", "residual", "error_code"}
    assert info["error_code"] == 0
    assert info["iterations"] >= 0
    assert np.isfinite(info["residual"])
    assert info["residual"] < 1e-3


def test_default_method_is_gaussian():
    A, b = spd_matrix(5), np.arange(1.0, 6.0)
    assert np.allclose(solve_system(A, b), np.linalg.solve(A, b))


def test_unknown_method_raises():
    A, b = spd_matrix(3), np.arange(1.0, 4.0)
    with pytest.raises(InvalidParameterError):
        solve_system(A, b, method="does_not_exist")


def test_singular_matrix_raises():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(SingularMatrixError):
        solve_system(A, b, method="gaussian_elimination")


def test_non_spd_raises_for_cholesky():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(NotSPDError):
        solve_system(A, b, method="cholesky")


def test_non_square_raises():
    with pytest.raises(ValueError):
        solve_system(np.ones((2, 3)), np.ones(2))


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        solve_system(np.eye(3), np.ones(2))


def test_accepts_list_and_integer_inputs():
    x = solve_system([[4, 1], [1, 3]], [5, 4], method="lu_decomposition")
    assert np.allclose(x, [1.0, 1.0])


def test_accepts_fortran_order_input():
    A = np.asfortranarray([[4.0, 1.0], [1.0, 3.0]])
    x = solve_system(A, np.array([5.0, 4.0]))
    assert np.allclose(x, [1.0, 1.0])


def test_accepts_sparse_input():
    sparse = pytest.importorskip("scipy.sparse")
    A = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    x = solve_system(A, np.array([5.0, 4.0]))
    assert np.allclose(x, [1.0, 1.0])


def test_accepts_column_vector_rhs():
    A = spd_matrix(4)
    b_col = np.arange(1.0, 5.0).reshape(-1, 1)
    x = solve_system(A, b_col)
    assert x.shape == (4,)
    assert np.allclose(x, np.linalg.solve(A, np.arange(1.0, 5.0)))


def test_iterative_method_reports_iterations():
    A, b = spd_matrix(6), np.arange(1.0, 7.0)
    _, info = solve_system(A, b, method="jacobi", return_info=True)
    assert info["iterations"] > 0


def test_convergence_error_is_runtimeerror_subclass():
    assert issubclass(ConvergenceError, RuntimeError)
