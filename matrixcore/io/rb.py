"""Rutherford-Boeing (.rb) reading and writing.

The Rutherford-Boeing / Harwell-Boeing sparse format uses rigid Fortran-style
fixed-width fields, so this module delegates to ``scipy.io`` rather than
re-implementing the column-pointer parser. Matrices are returned dense.
"""

import numpy as np
from scipy import io as scipy_io
from scipy import sparse


def read_rb(file_path):
    """Read a Rutherford-Boeing file into a dense float64 numpy array."""
    matrix = scipy_io.hb_read(file_path)
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(matrix, dtype=np.float64)


def write_rb(matrix, file_path):
    """Write a matrix to a Rutherford-Boeing file (stored in sparse CSC form)."""
    dense = np.ascontiguousarray(matrix, dtype=np.float64)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    scipy_io.hb_write(file_path, sparse.csc_matrix(dense))
