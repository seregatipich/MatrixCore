"""Robustness regression tests for solver convergence and pivoting.

These guard against failures discovered by adversarial stress testing: iterative
methods using a scale-dependent absolute tolerance (false ConvergenceError on
large or zero right-hand sides), Chebyshev/Richardson relying on loose Gershgorin
eigenvalue bounds, BiCGSTAB/CGNR rejecting the zero RHS, and Crout LU lacking the
partial pivoting needed for nonsingular matrices with a zero leading pivot.
"""

import numpy as np
import pytest

from matrixcore import solve_system

KRYLOV_GENERAL = ["gmres", "bicg", "bicgstab", "cgs", "tfqmr", "qmr", "gcr", "lsqr", "cgnr"]
STATIONARY = ["jacobi", "gauss_seidel", "sor", "richardson", "ssor"]
SPD_ITERATIVE = [
    "conjugate_gradient",
    "gradient_descent",
    "minres",
    "preconditioned_conjugate_gradient",
    "conjugate_residual",
    "symmlq",
    "chebyshev",
]
SEMI_ITERATIVE = ["iterative_refinement"]
ALL_ITERATIVE = KRYLOV_GENERAL + STATIONARY + SPD_ITERATIVE + SEMI_ITERATIVE


def _spd(n, seed, cond=20.0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * np.linspace(1.0, cond, n)) @ Q.T


def _well_conditioned_general(n, seed, cond=10.0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    R, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * np.linspace(1.0, cond, n)) @ R


def _diagonally_dominant(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    A += n * np.eye(n)
    return A


def _in_class_matrix(method, n, seed):
    if method in SPD_ITERATIVE:
        return _spd(n, seed)
    return _diagonally_dominant(n, seed)


def _relative_residual(A, x, b):
    return np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0)


@pytest.mark.parametrize("method", ALL_ITERATIVE)
def test_zero_rhs_returns_zero_solution(method):
    """x = 0 is the exact solution when b = 0; no solver may raise on it."""
    A = _in_class_matrix(method, 5, 3)
    b = np.zeros(5)
    x = solve_system(A, b, method=method)
    assert np.allclose(x, 0.0)


@pytest.mark.parametrize(
    "method", ["qmr", "gcr", "sor", "richardson", "ssor", "symmlq"] + SEMI_ITERATIVE
)
@pytest.mark.parametrize("scale", [1e6, 1e9])
def test_large_rhs_does_not_trip_absolute_tolerance(method, scale):
    """A scaled RHS must converge: the stopping test must be relative, not absolute."""
    A = _in_class_matrix(method, 5, 1)
    b = np.arange(1.0, 6.0) * scale
    x = solve_system(A, b, method=method)
    assert _relative_residual(A, x, b) < 1e-6


@pytest.mark.parametrize("cond", [5.0, 20.0, 100.0])
@pytest.mark.parametrize("n", [2, 4, 8])
def test_chebyshev_converges_on_dense_spd(n, cond):
    """Chebyshev must converge on any well-conditioned dense SPD system."""
    A = _spd(n, 0, cond=cond)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method="chebyshev")
    assert _relative_residual(A, x, b) < 1e-6


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cond", [50.0, 80.0])
def test_richardson_converges_on_ill_conditioned_spd(cond, seed):
    """Richardson must converge on moderately ill-conditioned dense SPD systems.

    The optimal SPD step 2/(lambda_min+lambda_max) replaces the loose Gershgorin
    step; fixed-step Richardson is inherently slow, so cond is kept within the reach
    of the iteration budget at the 1e-10 tolerance.
    """
    A = _spd(8, seed, cond=cond)
    b = np.arange(1.0, 9.0)
    x = solve_system(A, b, method="richardson")
    assert _relative_residual(A, x, b) < 1e-6


@pytest.mark.parametrize("n", [16, 30])
def test_bicgstab_converges_on_diagonally_dominant(n):
    """BiCGSTAB must converge on well-conditioned diagonally-dominant systems at larger n."""
    A = _diagonally_dominant(n, 0)
    b = np.arange(1.0, n + 1.0)
    x = solve_system(A, b, method="bicgstab")
    assert _relative_residual(A, x, b) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_bicgstab_converges_on_well_conditioned_general(seed):
    """BiCGSTAB converges on well-conditioned non-normal systems within its reliable size range."""
    A = _well_conditioned_general(10, seed, cond=2.0)
    b = np.arange(1.0, 11.0)
    x = solve_system(A, b, method="bicgstab")
    assert _relative_residual(A, x, b) < 1e-6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sor_converges_on_nonsymmetric_diagonally_dominant(seed):
    """SOR must converge on nonsymmetric diagonally-dominant systems Gauss-Seidel solves."""
    A = _diagonally_dominant(2, seed)
    b = np.array([1.0, 2.0])
    x = solve_system(A, b, method="sor")
    assert _relative_residual(A, x, b) < 1e-6


CROUT_PIVOT_CASES = {
    "zero_leading_pivot": (np.array([[0.0, 2.0], [3.0, 4.0]]), np.array([2.0, 7.0])),
    "permutation": (
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([1.0, 2.0, 3.0]),
    ),
    "tiny_leading_pivot": (np.array([[1e-13, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0])),
}


@pytest.mark.parametrize("case", list(CROUT_PIVOT_CASES))
def test_crout_solves_matrix_needing_row_swap(case):
    """Crout LU must pivot: a nonsingular matrix with a zero leading pivot is solvable."""
    A, b = CROUT_PIVOT_CASES[case]
    x = solve_system(A, b, method="crout")
    assert np.allclose(x, np.linalg.solve(A, b))
