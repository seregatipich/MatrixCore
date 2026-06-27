"""Shared fixtures and helpers for the MatrixCore test suite."""

import numpy as np
import pytest

# Solvers that require a symmetric positive-definite (or at least symmetric) matrix.
SPD_ONLY = frozenset(
    {
        "cholesky",
        "conjugate_gradient",
        "gradient_descent",
        "minres",
        "eigenvalue_decomposition",
        "ldlt",
        "preconditioned_conjugate_gradient",
        "conjugate_residual",
        "symmlq",
        "chebyshev",
    }
)
# Solvers that operate on a triangular matrix only.
TRIANGULAR = {"back_substitution": "upper", "forward_substitution": "lower"}
# Solvers that operate on a tridiagonal matrix only.
TRIDIAGONAL_ONLY = frozenset({"thomas"})

REL_TOL = 1e-5


def spd_matrix(n):
    """Well-conditioned symmetric positive-definite tridiagonal matrix."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = 4.0
        if i + 1 < n:
            A[i, i + 1] = -1.0
            A[i + 1, i] = -1.0
    return A


def general_matrix(n, seed=0):
    """Strongly diagonally-dominant, non-symmetric, nonsingular matrix."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    A += 10.0 * np.eye(n)
    return np.ascontiguousarray(A, dtype=np.float64)


def upper_matrix(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.triu(rng.uniform(1.0, 2.0, size=(n, n))) + n * np.eye(n)


def lower_matrix(n, seed=2):
    rng = np.random.default_rng(seed)
    return np.tril(rng.uniform(1.0, 2.0, size=(n, n))) + n * np.eye(n)


def systems_for(method, n=6):
    """Yield (A, b) systems a given solver is mathematically expected to handle."""
    b = np.arange(1.0, n + 1.0)
    if method in TRIANGULAR:
        A = upper_matrix(n) if TRIANGULAR[method] == "upper" else lower_matrix(n)
        return [(A, b)]
    if method in SPD_ONLY or method in TRIDIAGONAL_ONLY:
        # spd_matrix is symmetric, positive-definite, and tridiagonal, so it
        # satisfies both the SPD-only and tridiagonal-only solver requirements.
        return [(spd_matrix(n), b), (spd_matrix(4), np.array([2.0, -1.0, 0.5, 3.0]))]
    return [(spd_matrix(n), b), (general_matrix(n), b)]


@pytest.fixture
def spd6():
    return spd_matrix(6), np.arange(1.0, 7.0)


@pytest.fixture
def general6():
    return general_matrix(6), np.arange(1.0, 7.0)
