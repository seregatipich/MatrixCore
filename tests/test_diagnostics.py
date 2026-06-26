"""Tests for solver recommendation and matrix diagnostics."""

import numpy as np
import pytest

from matrixcore import list_available_solvers, recommend_solver
from matrixcore.diagnostics import _matrix_diagnostics
from tests.conftest import general_matrix, spd_matrix


def test_recommend_returns_known_solver():
    A = spd_matrix(5)
    assert recommend_solver(A) in list_available_solvers()


def test_recommend_cholesky_for_spd():
    assert recommend_solver(spd_matrix(6)) == "cholesky"


def test_recommend_back_substitution_for_upper_triangular():
    A = np.triu(np.array([[2.0, 1.0, 3.0], [0.0, 4.0, 1.0], [0.0, 0.0, 5.0]]))
    assert recommend_solver(A) == "back_substitution"


def test_recommend_forward_substitution_for_lower_triangular():
    A = np.tril(np.array([[2.0, 0.0, 0.0], [1.0, 4.0, 0.0], [3.0, 1.0, 5.0]]))
    assert recommend_solver(A) == "forward_substitution"


def test_recommend_lu_for_general_square():
    assert recommend_solver(general_matrix(5)) == "lu_decomposition"


@pytest.mark.parametrize("builder", [spd_matrix, general_matrix])
def test_diagnostics_keys_and_shape(builder):
    diag = _matrix_diagnostics(builder(5))
    for key in ("m", "n", "nnz", "is_sym", "is_posdef", "rank_est", "cond_est"):
        assert key in diag
    assert diag["m"] == 5
    assert diag["n"] == 5


def test_diagnostics_detects_symmetry_and_posdef():
    diag = _matrix_diagnostics(spd_matrix(5))
    assert diag["is_sym"] is True
    assert diag["is_posdef"] is True


def test_diagnostics_full_rank():
    diag = _matrix_diagnostics(general_matrix(5))
    assert diag["rank_est"] == 5


def test_recommend_overdetermined_returns_qr():
    assert recommend_solver(np.ones((5, 3))) == "qr_decomposition"


def test_recommend_underdetermined_returns_svd():
    assert recommend_solver(np.ones((3, 5))) == "svd"


def test_recommend_rank_deficient_square_returns_svd():
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])  # rank 2
    assert recommend_solver(A) == "svd"


def test_recommend_large_dense_spd_returns_cholesky():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((120, 120))
    A = M @ M.T + 120.0 * np.eye(120)  # dense, symmetric positive definite
    assert recommend_solver(A) == "cholesky"


def test_recommend_large_sparse_spd_returns_conjugate_gradient():
    A = spd_matrix(120)  # tridiagonal -> detected as sparse SPD
    assert recommend_solver(A) == "conjugate_gradient"


def test_recommend_large_well_conditioned_returns_lu():
    rng = np.random.default_rng(0)
    A = 10.0 * np.eye(120) + 0.01 * rng.standard_normal((120, 120))
    assert recommend_solver(A) == "lu_decomposition"


def test_recommend_large_ill_conditioned_returns_qr():
    A = np.triu(np.ones((120, 120)))
    np.fill_diagonal(A, np.logspace(0, -9, 120))  # non-symmetric, ill-conditioned
    assert recommend_solver(A) == "qr_decomposition"


def test_recommend_large_sparse_returns_valid_solver():
    sparse = pytest.importorskip("scipy.sparse")
    A = sparse.csr_matrix(spd_matrix(120))  # tridiagonal -> sparse
    assert recommend_solver(A) in list_available_solvers()


def test_diagnostics_handles_sparse_input():
    sparse = pytest.importorskip("scipy.sparse")
    diag = _matrix_diagnostics(sparse.csr_matrix(spd_matrix(120)))
    assert diag["m"] == 120
    assert diag["is_sparse"] is True
