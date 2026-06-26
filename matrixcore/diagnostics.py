"""Matrix diagnostics and solver recommendation (pure Python heuristics).

These helpers inspect a matrix and suggest a suitable solver. They are kept out
of the compiled extension because they are plain NumPy logic, not part of the C
numerical core.
"""

import numpy as np


def recommend_solver(A):
    """Recommend a solver method name based on matrix diagnostics.

    Parameters
    ----------
    A : array_like
        The coefficient matrix to analyze.

    Returns
    -------
    str
        Recommended solver method name (see ``list_available_solvers``).
    """
    diag = _matrix_diagnostics(A)

    if diag["m"] <= 100 and diag["n"] <= 100:
        if diag["m"] == diag["n"]:
            if diag["is_sym"] and diag["is_posdef"]:
                return "cholesky"
            if diag["is_tri_up"]:
                return "back_substitution"
            if diag["is_tri_lo"]:
                return "forward_substitution"
            if diag["rank_est"] < min(diag["m"], diag["n"]):
                return "svd"
            return "lu_decomposition"
        if diag["m"] > diag["n"]:
            return "qr_decomposition"
        return "svd"

    if diag["is_sparse"]:
        if diag["is_sym"] and diag["is_posdef"]:
            return "conjugate_gradient"
        if diag["diag_dom"]:
            return "jacobi"
        return "gmres"
    if diag["is_sym"] and diag["is_posdef"]:
        return "cholesky"
    if diag["cond_est"] < 1e6:
        return "lu_decomposition"
    return "qr_decomposition"


def _matrix_diagnostics(A):
    """Compute a dictionary of diagnostic properties used by recommend_solver."""
    diagnostics = {
        "m": 0,
        "n": 0,
        "nnz": 0,
        "sparsity": 0.0,
        "is_sparse": False,
        "is_sym": False,
        "is_tri_up": False,
        "is_tri_lo": False,
        "diag_dom": False,
        "cond_est": float("inf"),
        "is_posdef": False,
        "rank_est": 0,
    }

    try:
        m, n = A.shape
        diagnostics["m"] = m
        diagnostics["n"] = n

        nnz = A.nnz if hasattr(A, "nnz") else int(np.count_nonzero(A))
        diagnostics["nnz"] = nnz

        sparsity = nnz / (m * n) if m * n > 0 else 0.0
        diagnostics["sparsity"] = sparsity
        diagnostics["is_sparse"] = sparsity < 0.05

        if m == n:
            _check_symmetry(A, diagnostics)
        _check_triangular(A, diagnostics)
        if m == n:
            _check_diag_dominance(A, diagnostics)
            _estimate_condition(A, diagnostics)
            if diagnostics["is_sym"]:
                _check_posdef(A, diagnostics)
        _estimate_rank(A, diagnostics)
    except Exception:
        pass  # always return a fully-populated dict

    return diagnostics


def _check_symmetry(A, diagnostics):
    try:
        diff = A - A.T
        if hasattr(A, "toarray"):
            norm_diff = abs(diff).sum()
            norm_A = abs(A).sum()
        else:
            norm_diff = np.linalg.norm(diff, ord=1)
            norm_A = np.linalg.norm(A, ord=1)
        diagnostics["is_sym"] = bool(norm_diff <= 1e-12 * norm_A) if norm_A > 0 else True
    except Exception:
        pass


def _check_triangular(A, diagnostics):
    try:
        rows, cols = A.nonzero() if hasattr(A, "nonzero") else np.nonzero(A)
        if len(rows) > 0:
            diagnostics["is_tri_up"] = bool(np.all(rows <= cols))
            diagnostics["is_tri_lo"] = bool(np.all(rows >= cols))
    except Exception:
        pass


def _check_diag_dominance(A, diagnostics):
    try:
        if hasattr(A, "toarray"):
            return  # diagonal dominance left to dense inputs
        diag = np.diag(A)
        row_sums = np.sum(np.abs(A), axis=1) - np.abs(diag)
        diagnostics["diag_dom"] = bool(np.all(np.abs(diag) >= row_sums))
    except Exception:
        pass


def _estimate_condition(A, diagnostics):
    try:
        dense = A.todense() if hasattr(A, "todense") else A
        s = np.linalg.svd(dense, compute_uv=False)
        diagnostics["cond_est"] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    except Exception:
        pass


def _check_posdef(A, diagnostics):
    try:
        if hasattr(A, "todense"):
            return
        np.linalg.cholesky(A)
        diagnostics["is_posdef"] = True
    except np.linalg.LinAlgError:
        pass
    except Exception:
        pass


def _estimate_rank(A, diagnostics):
    try:
        dense = A.todense() if hasattr(A, "todense") else A
        s = np.linalg.svd(dense, compute_uv=False)
        tol = max(diagnostics["m"], diagnostics["n"]) * np.finfo(float).eps * s[0]
        diagnostics["rank_est"] = int(np.sum(s > tol))
    except Exception:
        pass
