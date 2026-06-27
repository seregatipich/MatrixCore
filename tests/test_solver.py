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


def test_fifty_solvers_available():
    assert len(ALL_SOLVERS) == 50
    assert len(set(ALL_SOLVERS)) == 50


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


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 10])
@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11])
def test_gmres_converges_on_well_conditioned(n, seed):
    """GMRES must converge on well-conditioned general and SPD systems."""
    from tests.conftest import general_matrix, spd_matrix

    rng = np.random.default_rng(seed)
    for A in (general_matrix(n, seed=seed * 7 + n), spd_matrix(n)):
        b = rng.uniform(-1.0, 1.0, size=n)
        x = solve_system(A, b, method="gmres")
        assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6


@pytest.mark.parametrize("n", [12, 20, 30])
def test_cramers_rule_handles_larger_n(n):
    """Cramer's rule (GE-determinant based) is not artificially capped at n=10."""
    from tests.conftest import spd_matrix

    A = spd_matrix(n)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method="cramers_rule")
    assert np.allclose(x, np.linalg.solve(A, b))


@pytest.mark.parametrize(
    "method",
    ["bicgstab", "cgs", "tfqmr", "qmr", "gcr", "lsqr", "cgnr", "richardson", "ssor"],
)
@pytest.mark.parametrize("n", [4, 6, 10])
def test_new_iterative_solvers_converge_on_general(method, n):
    """New nonsymmetric/stationary iterative solvers converge on well-conditioned systems."""
    from tests.conftest import general_matrix

    rng = np.random.default_rng(n)
    A = general_matrix(n, seed=n)
    b = rng.uniform(-1.0, 1.0, size=n)
    x, info = solve_system(A, b, method=method, return_info=True)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6
    assert info["iterations"] >= 0


@pytest.mark.parametrize(
    "method",
    ["preconditioned_conjugate_gradient", "conjugate_residual", "symmlq", "chebyshev"],
)
@pytest.mark.parametrize("n", [4, 6, 10])
def test_new_spd_iterative_solvers_converge(method, n):
    """New SPD Krylov/semi-iterative solvers converge on SPD systems."""
    A = spd_matrix(n)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method=method)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6


@pytest.mark.parametrize(
    "method",
    [
        "crout",
        "ldlt",
        "givens_qr",
        "modified_gram_schmidt",
        "classical_gram_schmidt",
        "lq_decomposition",
    ],
)
@pytest.mark.parametrize("n", [3, 5, 8])
def test_new_direct_solvers_match_numpy(method, n):
    """New direct/decomposition solvers reproduce numpy on SPD systems of varied size."""
    A = spd_matrix(n)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method=method)
    assert np.allclose(x, np.linalg.solve(A, b))


@pytest.mark.parametrize("n", [3, 5, 8, 12])
def test_thomas_matches_numpy_on_tridiagonal(n):
    """Thomas algorithm solves tridiagonal systems exactly."""
    A = spd_matrix(n)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method="thomas")
    assert np.allclose(x, np.linalg.solve(A, b))
