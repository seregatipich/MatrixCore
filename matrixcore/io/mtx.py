"""Matrix Market (.mtx) reading and writing.

A hand-written parser for the Matrix Market text format, kept dependency-light
(NumPy only). Both ``array`` (dense) and ``coordinate`` (sparse) layouts are
read and returned as a dense float64 array, since MatrixCore solves dense
systems. See https://math.nist.gov/MatrixMarket/formats.html.
"""

import numpy as np

_FIELDS = {"real", "integer", "pattern"}
_SYMMETRIES = {"general", "symmetric", "skew-symmetric"}


def read_matrix_market(file_path):
    """Read a Matrix Market file into a dense float64 numpy array."""
    with open(file_path) as handle:
        header = handle.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError("File does not start with a %%MatrixMarket banner")

        parts = header.split()
        if len(parts) < 5 or parts[1].lower() != "matrix":
            raise ValueError(f"Unsupported Matrix Market header: {header!r}")

        layout, field, symmetry = parts[2].lower(), parts[3].lower(), parts[4].lower()
        if field not in _FIELDS:
            raise ValueError(f"Unsupported Matrix Market field: {field!r}")
        if symmetry not in _SYMMETRIES:
            raise ValueError(f"Unsupported Matrix Market symmetry: {symmetry!r}")

        size_line = _next_data_line(handle)
        dims = size_line.split()
        rows, cols = int(dims[0]), int(dims[1])
        matrix = np.zeros((rows, cols), dtype=np.float64)

        if layout == "array":
            values = []
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("%"):
                    values.append(float(stripped))
            if symmetry == "general":
                matrix = np.array(values, dtype=np.float64).reshape((rows, cols), order="F")
            else:
                # symmetric/skew-symmetric arrays store the packed lower triangle,
                # column by column.
                sign = 1.0 if symmetry == "symmetric" else -1.0
                index = 0
                for j in range(cols):
                    for i in range(j, rows):
                        value = values[index]
                        index += 1
                        matrix[i, j] = value
                        if i != j:
                            matrix[j, i] = sign * value
        elif layout == "coordinate":
            nnz = int(dims[2])
            for _ in range(nnz):
                tokens = _next_data_line(handle).split()
                i, j = int(tokens[0]) - 1, int(tokens[1]) - 1
                value = 1.0 if field == "pattern" else float(tokens[2])
                matrix[i, j] = value
                if symmetry == "symmetric" and i != j:
                    matrix[j, i] = value
                elif symmetry == "skew-symmetric" and i != j:
                    matrix[j, i] = -value
        else:
            raise ValueError(f"Unsupported Matrix Market layout: {layout!r}")

    return matrix


def write_matrix_market(matrix, file_path, comment=None):
    """Write a dense matrix to a Matrix Market file in ``array`` layout."""
    dense = np.ascontiguousarray(matrix, dtype=np.float64)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    rows, cols = dense.shape
    with open(file_path, "w") as handle:
        handle.write("%%MatrixMarket matrix array real general\n")
        if comment:
            for line in comment.splitlines():
                handle.write(f"% {line}\n")
        handle.write(f"{rows} {cols}\n")
        for j in range(cols):
            for i in range(rows):
                handle.write(f"{float(dense[i, j]):.17g}\n")


def _next_data_line(handle):
    for line in handle:
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            return stripped
    raise ValueError("Unexpected end of Matrix Market file while reading data")
