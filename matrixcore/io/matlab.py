"""MATLAB (.mat) reading and writing.

MATLAB's binary container (v5 and the HDF5-based v7.3) is intricate, so this
module delegates to ``scipy.io`` rather than re-implementing a binary parser.
Matrices are returned as dense float64 arrays for the solver.
"""

import numpy as np
from scipy import io as scipy_io


def read_matlab(file_path, variable_name=None):
    """Read a variable from a MATLAB .mat file as a dense float64 array.

    If ``variable_name`` is None, the single non-metadata variable is returned;
    when several are present a ValueError lists the available names.
    """
    contents = scipy_io.loadmat(file_path)
    variables = {name: value for name, value in contents.items() if not name.startswith("__")}
    if not variables:
        raise ValueError(f"No variables found in {file_path!r}")

    if variable_name is None:
        if len(variables) > 1:
            raise ValueError(
                f"Multiple variables in {file_path!r}: {sorted(variables)}; "
                "pass variable_name to select one"
            )
        variable_name = next(iter(variables))
    elif variable_name not in variables:
        raise ValueError(
            f"Variable {variable_name!r} not found in {file_path!r}; available: {sorted(variables)}"
        )

    return np.ascontiguousarray(variables[variable_name], dtype=np.float64)


def write_matlab(matrix, file_path, variable_name="A"):
    """Write a matrix to a MATLAB .mat file under ``variable_name``."""
    dense = np.ascontiguousarray(matrix, dtype=np.float64)
    scipy_io.savemat(file_path, {variable_name: dense})
