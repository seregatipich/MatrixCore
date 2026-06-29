"""Scale-robustness regression tests (failures found via the SuiteSparse dataset).

Real-world matrices exposed solvers that used *absolute* thresholds where a
*relative* (scale-aware) test is required: large-scale SPD matrices rejected as
non-symmetric, tiny-entry well-conditioned matrices reported singular by QR /
normal-equation methods, determinant under/overflow in Cramer's rule, and
row-echelon methods lacking partial pivoting.
"""

import numpy as np
import pytest

from matrixcore import solve_system


def _well_conditioned(n, seed, cond=5.0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    R, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * np.linspace(1.0, cond, n)) @ R


def _large_scale_spd(n, seed, scale):
    """SPD matrix of magnitude ~scale, symmetric only to relative machine precision."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * np.linspace(1.0, scale, n)) @ Q.T
    asymmetry = np.triu(rng.standard_normal((n, n)), 1) * (scale * 1e-14)
    return A + asymmetry


def _resid(A, x, b):
    return np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0)


@pytest.mark.parametrize(
    "method",
    ["cholesky", "ldlt", "eigenvalue_decomposition", "conjugate_gradient", "minres", "symmlq"],
)
def test_spd_solvers_accept_large_scale_spd(method):
    """A large-magnitude SPD matrix must not be rejected by an absolute symmetry tolerance."""
    A = _large_scale_spd(24, 0, scale=1e6)
    b = A @ np.ones(24)
    x = solve_system(A, b, method=method)
    assert _resid(A, x, b) < 1e-6


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_elimination",
        "lu_decomposition",
        "qr_decomposition",
        "lq_decomposition",
        "normal_equations",
        "orthogonal_projection",
        "crout",
        "givens_qr",
    ],
)
def test_direct_solvers_handle_tiny_entries(method):
    """A well-conditioned matrix with tiny entries must not be reported singular."""
    A = 1e-6 * _well_conditioned(30, 1, cond=5.0)
    b = A @ np.ones(30)
    x = solve_system(A, b, method=method)
    assert _resid(A, x, b) < 1e-6


@pytest.mark.parametrize("method", ["cramers_rule", "determinant"])
def test_determinant_methods_survive_underflow(method):
    """Determinant-based solvers must handle a tiny-scale matrix whose det underflows."""
    A = 0.05 * _well_conditioned(40, 2, cond=5.0)
    b = A @ np.ones(40)
    x = solve_system(A, b, method=method)
    assert _resid(A, x, b) < 1e-6


@pytest.mark.parametrize("method", ["row_echelon", "reduced_row_echelon"])
def test_row_echelon_uses_partial_pivoting_for_accuracy(method):
    """A tiny (nonzero) leading pivot must be swapped away; first-nonzero pivoting is unstable.

    Without partial pivoting the small (0,0) pivot causes catastrophic growth and a
    residual orders of magnitude worse than a properly pivoted elimination.
    """
    A = np.array([[1e-9, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    b = A @ np.ones(3)
    x = solve_system(A, b, method=method)
    assert _resid(A, x, b) < 1e-10


@pytest.mark.parametrize("method", ["row_echelon", "reduced_row_echelon"])
def test_row_echelon_solves_zero_leading_pivot(method):
    """A nonsingular matrix with a zero leading pivot must still be solved."""
    A = np.array([[0.0, 2.0], [3.0, 4.0]])
    b = np.array([2.0, 7.0])
    x = solve_system(A, b, method=method)
    assert np.allclose(x, np.linalg.solve(A, b))
