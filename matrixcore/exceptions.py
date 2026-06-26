"""Exception hierarchy for MatrixCore.

The C core reports outcomes through integer error codes defined in
``src/solvers.h``. :func:`error_for_code` translates those codes into the
typed exceptions below so that Python callers can catch precise failure modes
instead of inspecting integers.
"""

SOLVER_SUCCESS = 0
SOLVER_MEMORY_ERROR = 1
SOLVER_SINGULAR_MATRIX = 2
SOLVER_INVALID_PARAMETERS = 3
SOLVER_NOT_CONVERGED = 4
SOLVER_INCONSISTENT_SYSTEM = 5
SOLVER_NOT_SPD_MATRIX = 6
SOLVER_NO_CONVERGENCE = 7
SOLVER_NO_SOLUTION = 8
SOLVER_MULTIPLE_SOLUTIONS = 9
SOLVER_INPUT_ERROR = 10


class MatrixCoreError(Exception):
    """Base class for every error raised by MatrixCore."""


class SolverMemoryError(MatrixCoreError, MemoryError):
    """The C core could not allocate working memory."""


class SingularMatrixError(MatrixCoreError, ValueError):
    """The coefficient matrix is singular or numerically singular."""


class NotSPDError(MatrixCoreError, ValueError):
    """The matrix is not symmetric positive definite for the chosen method."""


class ConvergenceError(MatrixCoreError, RuntimeError):
    """An iterative method failed to converge within its iteration budget."""


class InconsistentSystemError(MatrixCoreError, ValueError):
    """The system has no solution (inconsistent)."""


class MultipleSolutionsError(MatrixCoreError, ValueError):
    """The system is rank deficient and has infinitely many solutions."""


class InvalidParameterError(MatrixCoreError, ValueError):
    """The solver received invalid parameters or an unknown method name."""


_CODE_TABLE = {
    SOLVER_MEMORY_ERROR: (SolverMemoryError, "the solver ran out of memory"),
    SOLVER_SINGULAR_MATRIX: (SingularMatrixError, "matrix is singular or nearly singular"),
    SOLVER_INVALID_PARAMETERS: (InvalidParameterError, "invalid solver parameters"),
    SOLVER_NOT_CONVERGED: (ConvergenceError, "iterative method did not converge"),
    SOLVER_INCONSISTENT_SYSTEM: (InconsistentSystemError, "system is inconsistent (no solution)"),
    SOLVER_NOT_SPD_MATRIX: (NotSPDError, "matrix is not symmetric positive definite"),
    SOLVER_NO_CONVERGENCE: (ConvergenceError, "iterative method did not converge"),
    SOLVER_NO_SOLUTION: (InconsistentSystemError, "system has no solution"),
    SOLVER_MULTIPLE_SOLUTIONS: (MultipleSolutionsError, "system has infinitely many solutions"),
    SOLVER_INPUT_ERROR: (InvalidParameterError, "invalid input or unknown solver method"),
}


def error_for_code(code, method=None):
    """Return an exception instance for a non-zero error code, or ``None`` for success."""
    if code == SOLVER_SUCCESS:
        return None
    exc_type, message = _CODE_TABLE.get(
        code, (MatrixCoreError, f"solver failed with error code {code}")
    )
    if method is not None:
        message = f"{message} (method='{method}')"
    return exc_type(message)
