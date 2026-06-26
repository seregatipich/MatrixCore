"""Tests for the error-code to exception mapping."""

import pytest

from matrixcore import exceptions as exc


def test_success_maps_to_none():
    assert exc.error_for_code(exc.SOLVER_SUCCESS) is None


@pytest.mark.parametrize(
    "code,expected",
    [
        (exc.SOLVER_MEMORY_ERROR, exc.SolverMemoryError),
        (exc.SOLVER_SINGULAR_MATRIX, exc.SingularMatrixError),
        (exc.SOLVER_INVALID_PARAMETERS, exc.InvalidParameterError),
        (exc.SOLVER_NOT_CONVERGED, exc.ConvergenceError),
        (exc.SOLVER_INCONSISTENT_SYSTEM, exc.InconsistentSystemError),
        (exc.SOLVER_NOT_SPD_MATRIX, exc.NotSPDError),
        (exc.SOLVER_NO_CONVERGENCE, exc.ConvergenceError),
        (exc.SOLVER_NO_SOLUTION, exc.InconsistentSystemError),
        (exc.SOLVER_MULTIPLE_SOLUTIONS, exc.MultipleSolutionsError),
        (exc.SOLVER_INPUT_ERROR, exc.InvalidParameterError),
    ],
)
def test_code_maps_to_expected_exception(code, expected):
    error = exc.error_for_code(code)
    assert isinstance(error, expected)
    assert isinstance(error, exc.MatrixCoreError)


def test_unknown_code_maps_to_base_error():
    error = exc.error_for_code(999)
    assert isinstance(error, exc.MatrixCoreError)
    assert "999" in str(error)


def test_method_name_included_in_message():
    error = exc.error_for_code(exc.SOLVER_SINGULAR_MATRIX, method="cholesky")
    assert "cholesky" in str(error)


def test_exception_subclassing_for_builtins():
    assert issubclass(exc.SingularMatrixError, ValueError)
    assert issubclass(exc.ConvergenceError, RuntimeError)
    assert issubclass(exc.SolverMemoryError, MemoryError)
