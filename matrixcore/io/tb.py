"""
Matrix Market Format I/O Module for MatrixCore

This module provides functionality to read and write matrices in the Matrix Market
format (.mtx files) and convert them to the format used by MatrixCore.

The Matrix Market format is a text-based file format for representing sparse matrices.
More information: https://math.nist.gov/MatrixMarket/formats.html
"""

import numpy as np
import re
from scipy import sparse


def read_matrix_market(file_path):
    """
    Read a matrix from a Matrix Market (.mtx) file and convert it to the format
    used by MatrixCore.

    Parameters:
    -----------
    file_path : str
        The path to the Matrix Market file.

    Returns:
    --------
    numpy.ndarray
        The matrix in dense format if it's a dense matrix, or a scipy.sparse matrix
        if it's a sparse matrix.
    """
    with open(file_path, 'r') as f:
        # Read the header line
        header = f.readline().strip()
        if not header.startswith('%%MatrixMarket'):
            raise ValueError("File does not appear to be in Matrix Market format")

        # Parse the header
        header_parts = header.split()
        if len(header_parts) < 4:
            raise ValueError("Invalid Matrix Market header format")

        matrix_type = header_parts[2].lower()
        format_type = header_parts[3].lower()

        # Skip comment lines
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()

        # Read dimensions
        dimensions = line.split()
        rows, cols = int(dimensions[0]), int(dimensions[1])

        if matrix_type == 'matrix':
            if format_type == 'array':
                # Dense matrix
                data = []
                for line in f:
                    if line.strip() and not line.startswith('%'):
                        data.append(float(line.strip()))

                # Reshape to get the matrix
                matrix = np.array(data, dtype=np.float64).reshape((rows, cols), order='F')
                return matrix

            elif format_type == 'coordinate':
                # Sparse matrix
                try:
                    entries = int(dimensions[2])
                except IndexError:
                    raise ValueError("Invalid Matrix Market coordinate format")

                row_indices = []
                col_indices = []
                values = []

                for _ in range(entries):
                    line = f.readline()
                    if not line:
                        break

                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    # Matrix Market indices are 1-based
                    row_indices.append(int(parts[0]) - 1)
                    col_indices.append(int(parts[1]) - 1)
                    values.append(float(parts[2]))

                # Create a sparse matrix in COO format, then convert to CSR
                coo_matrix = sparse.coo_matrix(
                    (values, (row_indices, col_indices)),
                    shape=(rows, cols)
                )

                # Convert to CSR format for efficient operations
                matrix = coo_matrix.tocsr()

                # For MatrixCore, convert to dense if needed
                # Depending on your implementation, you may want to return
                # the sparse matrix directly or convert to dense
                return matrix

        else:
            raise ValueError(f"Unsupported Matrix Market type: {matrix_type}")


def write_matrix_market(matrix, file_path, comment=None, field='real', symmetry='general'):
    """
    Write a matrix to a Matrix Market (.mtx) file.

    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse matrix
        The matrix to write to the file.
    file_path : str
        The path to write the Matrix Market file to.
    comment : str, optional
        A comment to include in the header of the file.
    field : str, optional
        The field of the matrix elements (default: 'real').
        Options: 'real', 'complex', 'integer', 'pattern'.
    symmetry : str, optional
        The symmetry of the matrix (default: 'general').
        Options: 'general', 'symmetric', 'skew-symmetric', 'hermitian'.
    """
    with open(file_path, 'w') as f:
        # Write the header
        f.write(f"%%MatrixMarket matrix {('coordinate' if sparse.issparse(matrix) else 'array')} {field} {symmetry}\n")

        # Write any comments
        if comment:
            for line in comment.split('\n'):
                f.write(f"% {line}\n")

        # Convert to appropriate format if needed
        if sparse.issparse(matrix):
            # Convert to COO format for easier iteration
            matrix = matrix.tocoo()

            # Write dimensions and number of non-zero entries
            f.write(f"{matrix.shape[0]} {matrix.shape[1]} {matrix.nnz}\n")

            # Write the entries (1-indexed)
            for i, j, v in zip(matrix.row, matrix.col, matrix.data):
                f.write(f"{i + 1} {j + 1} {v}\n")
        else:
            # Dense matrix
            rows, cols = matrix.shape
            f.write(f"{rows} {cols}\n")

            # Write the entries (column-major order)
            for j in range(cols):
                for i in range(rows):
                    f.write(f"{matrix[i, j]}\n")


def load_matrix(file_path):
    """
    Load a matrix from a Matrix Market file and return it in the format
    used by MatrixCore.

    This is a convenience function that calls read_matrix_market().

    Parameters:
    -----------
    file_path : str
        The path to the Matrix Market file.

    Returns:
    --------
    numpy.ndarray or scipy.sparse matrix
        The matrix loaded from the file.
    """
    return read_matrix_market(file_path)


def save_matrix(matrix, file_path, **kwargs):
    """
    Save a matrix to a Matrix Market file.

    This is a convenience function that calls write_matrix_market().

    Parameters:
    -----------
    matrix : numpy.ndarray or scipy.sparse matrix
        The matrix to save.
    file_path : str
        The path to save the Matrix Market file to.
    **kwargs :
        Additional keyword arguments to pass to write_matrix_market().
    """
    write_matrix_market(matrix, file_path, **kwargs)