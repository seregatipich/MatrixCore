"""Property-based tests: random well-conditioned systems must solve accurately."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from matrixcore import solve_system

SETTINGS = settings(max_examples=60, deadline=None)


def _spd(seed, n):
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1.0, 1.0, size=(n, n))
    return M @ M.T + n * np.eye(n)  # symmetric positive definite, well-conditioned


def _diagonally_dominant(seed, n):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    return A + n * 2.0 * np.eye(n)


@SETTINGS
@given(
    seed=st.integers(0, 5000),
    n=st.integers(2, 7),
    method=st.sampled_from(
        [
            "gaussian_elimination",
            "lu_decomposition",
            "qr_decomposition",
            "cholesky",
            "conjugate_gradient",
        ]
    ),
)
def test_spd_systems_solve_accurately(seed, n, method):
    A = _spd(seed, n)
    b = np.random.default_rng(seed + 1).uniform(-1.0, 1.0, size=n)
    x = solve_system(A, b, method=method)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6


@SETTINGS
@given(
    seed=st.integers(0, 5000),
    n=st.integers(2, 7),
    method=st.sampled_from(
        [
            "gaussian_elimination",
            "lu_decomposition",
            "qr_decomposition",
            "gauss_jordan",
            "matrix_inversion",
        ]
    ),
)
def test_general_systems_solve_accurately(seed, n, method):
    A = _diagonally_dominant(seed, n)
    b = np.random.default_rng(seed + 7).uniform(-1.0, 1.0, size=n)
    x = solve_system(A, b, method=method)
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6


@SETTINGS
@given(seed=st.integers(0, 5000), n=st.integers(2, 7))
def test_solution_is_independent_of_method(seed, n):
    A = _spd(seed, n)
    b = np.random.default_rng(seed + 3).uniform(-1.0, 1.0, size=n)
    x_lu = solve_system(A, b, method="lu_decomposition")
    x_chol = solve_system(A, b, method="cholesky")
    assert np.allclose(x_lu, x_chol, atol=1e-6)
