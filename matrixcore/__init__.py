"""MatrixCore: 30 hand-written C solvers for dense linear systems Ax = b.

The numerical core is implemented from scratch in C (``src/solvers.c``) and
exposed through a Cython/NumPy interface. NumPy and SciPy are used only for
input handling and file I/O, never for the solve itself.
"""

from importlib.metadata import PackageNotFoundError, version

from matrixcore.exceptions import (
    ConvergenceError,
    InconsistentSystemError,
    InvalidParameterError,
    MatrixCoreError,
    MultipleSolutionsError,
    NotSPDError,
    SingularMatrixError,
    SolverMemoryError,
)
from matrixcore.solvers import (
    list_available_solvers,
    recommend_solver,
    solve_system,
)

try:
    __version__ = version("matrixcore")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "solve_system",
    "recommend_solver",
    "list_available_solvers",
    "MatrixCoreError",
    "SingularMatrixError",
    "NotSPDError",
    "ConvergenceError",
    "InconsistentSystemError",
    "MultipleSolutionsError",
    "InvalidParameterError",
    "SolverMemoryError",
    "__version__",
]
