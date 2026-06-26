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
