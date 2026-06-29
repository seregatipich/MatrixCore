"""Validate solvers against real matrices from the SuiteSparse Matrix Collection.

This is an opt-in test: it needs the ``ssgetpy`` package (``pip install -e ".[dataset]"``)
and network access to download matrices. It skips cleanly when either is unavailable,
so it never breaks an offline CI run. It both exercises MatrixCore's Matrix Market
reader on real files and confirms the solvers handle a span of real-world systems
(structural, circuit, chemical, graph) including the small-entry and pivoting-sensitive
matrices that motivated the scale-robustness fixes.
"""

import os

import numpy as np
import pytest

ssgetpy = pytest.importorskip("ssgetpy")

from matrixcore import load_matrix, solve_system  # noqa: E402  (after importorskip guard)

# (name, methods expected to solve it) — chosen so every matrix is solvable to a
# small residual by these (backward-stable / appropriate) methods.
CASES = [
    ("bcsstk01", ["cholesky", "ldlt", "lu_decomposition", "gaussian_elimination"]),
    ("west0067", ["row_echelon", "reduced_row_echelon", "lu_decomposition", "qr_decomposition"]),
    ("bcsstm02", ["cramers_rule", "determinant", "cholesky", "lu_decomposition"]),
    ("bfwb62", ["qr_decomposition", "lu_decomposition", "gaussian_elimination", "cramers_rule"]),
    ("cage5", ["cramers_rule", "lu_decomposition", "gaussian_elimination", "row_echelon"]),
]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("suitesparse")


def _load(name, destdir):
    try:
        hits = ssgetpy.search(name=name, limit=1)
        if not hits:
            pytest.skip(f"SuiteSparse matrix {name!r} not in index")
        hits[0].download(format="MM", destpath=str(destdir), extract=True)
    except Exception as exc:  # noqa: BLE001 - offline / network failure -> skip
        pytest.skip(f"could not download SuiteSparse matrix {name!r}: {exc}")
    path = os.path.join(str(destdir), name, f"{name}.mtx")
    if not os.path.exists(path):
        pytest.skip(f"SuiteSparse matrix file missing for {name!r}")
    return np.ascontiguousarray(np.asarray(load_matrix(path), dtype=float))


@pytest.mark.parametrize("name,methods", CASES, ids=[case[0] for case in CASES])
def test_solvers_on_suitesparse_matrices(name, methods, cache):
    A = _load(name, cache)
    assert A.shape[0] == A.shape[1]
    b = A @ np.ones(A.shape[0])
    for method in methods:
        x = solve_system(A.copy(), b.copy(), method=method)
        rel_residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert rel_residual < 1e-6, f"{name}/{method}: rel residual {rel_residual:.2e}"
