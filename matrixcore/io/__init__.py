"""Unified matrix file I/O for MatrixCore.

``load_matrix`` / ``save_matrix`` dispatch to a format-specific reader/writer,
auto-detecting the format from the file extension when ``format`` is omitted.
All readers return a dense float64 numpy array suitable for
:func:`matrixcore.solve_system`.
"""

import os

from matrixcore.io.matlab import read_matlab, write_matlab
from matrixcore.io.mtx import read_matrix_market, write_matrix_market
from matrixcore.io.rb import read_rb, write_rb

_EXTENSION_TO_FORMAT = {
    ".mtx": "mtx",
    ".mat": "matlab",
    ".rb": "rb",
}

_FORMAT_ALIASES = {
    "mtx": "mtx",
    "matrix_market": "mtx",
    "matlab": "matlab",
    "mat": "matlab",
    "rb": "rb",
    "rutherford_boeing": "rb",
    "harwell_boeing": "rb",
}


def _resolve_format(file_path, format):
    if format is not None:
        key = format.lower()
        if key not in _FORMAT_ALIASES:
            raise ValueError(f"Unknown format {format!r}; supported: mtx, matlab, rb")
        return _FORMAT_ALIASES[key]
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in _EXTENSION_TO_FORMAT:
        raise ValueError(
            f"Cannot infer format from extension {extension!r}; pass format= explicitly"
        )
    return _EXTENSION_TO_FORMAT[extension]


def load_matrix(file_path, format=None, **kwargs):
    """Load a matrix from a file as a dense float64 numpy array.

    Parameters
    ----------
    file_path : str
        Path to the matrix file.
    format : {'mtx', 'matlab', 'rb'}, optional
        Explicit format. Inferred from the file extension when omitted.
    **kwargs
        Forwarded to the format reader (e.g. ``variable_name`` for MATLAB).
    """
    resolved = _resolve_format(file_path, format)
    if resolved == "mtx":
        return read_matrix_market(file_path, **kwargs)
    if resolved == "matlab":
        return read_matlab(file_path, **kwargs)
    return read_rb(file_path, **kwargs)


def save_matrix(matrix, file_path, format=None, **kwargs):
    """Save a matrix to a file in the given (or extension-inferred) format."""
    resolved = _resolve_format(file_path, format)
    if resolved == "mtx":
        return write_matrix_market(matrix, file_path, **kwargs)
    if resolved == "matlab":
        return write_matlab(matrix, file_path, **kwargs)
    return write_rb(matrix, file_path, **kwargs)


__all__ = [
    "load_matrix",
    "save_matrix",
    "read_matrix_market",
    "write_matrix_market",
    "read_matlab",
    "write_matlab",
    "read_rb",
    "write_rb",
]
