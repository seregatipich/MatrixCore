"""Tests for the matrix file I/O layer."""

import numpy as np
import pytest

from matrixcore import load_matrix, save_matrix, solve_system
from matrixcore.io import read_matrix_market

FORMATS = ["mtx", "matlab", "rb"]
EXT = {"mtx": ".mtx", "matlab": ".mat", "rb": ".rb"}


@pytest.fixture
def matrix():
    return np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])


@pytest.mark.parametrize("fmt", FORMATS)
def test_matrix_roundtrip(tmp_path, matrix, fmt):
    path = tmp_path / f"A{EXT[fmt]}"
    save_matrix(matrix, str(path))
    loaded = load_matrix(str(path))
    assert loaded.shape == matrix.shape
    assert loaded.dtype == np.float64
    assert np.allclose(loaded, matrix)


@pytest.mark.parametrize("fmt", FORMATS)
def test_load_then_solve(tmp_path, matrix, fmt):
    b = np.array([1.0, 2.0, 3.0])
    save_matrix(matrix, str(tmp_path / f"A{EXT[fmt]}"))
    save_matrix(b, str(tmp_path / f"b{EXT[fmt]}"))
    A2 = load_matrix(str(tmp_path / f"A{EXT[fmt]}"))
    b2 = load_matrix(str(tmp_path / f"b{EXT[fmt]}"))
    x = solve_system(A2, b2, method="lu_decomposition")
    assert np.allclose(x, np.linalg.solve(matrix, b))


def test_explicit_format_overrides_extension(tmp_path, matrix):
    path = tmp_path / "A.mtx"
    save_matrix(matrix, str(path))
    assert np.allclose(load_matrix(str(path), format="matrix_market"), matrix)


def test_matlab_variable_name(tmp_path, matrix):
    path = tmp_path / "K.mat"
    save_matrix(matrix, str(path), variable_name="K")
    assert np.allclose(load_matrix(str(path), variable_name="K"), matrix)


def test_matrix_market_coordinate_symmetric(tmp_path):
    path = tmp_path / "coo.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate real symmetric\n3 3 3\n1 1 2.0\n2 2 2.0\n2 1 1.0\n"
    )
    expected = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(read_matrix_market(str(path)), expected)


def test_unknown_extension_raises():
    with pytest.raises(ValueError, match="infer format"):
        load_matrix("matrix.unknown")


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        load_matrix("matrix", format="zzz")


def test_bad_matrix_market_banner_raises(tmp_path):
    path = tmp_path / "bad.mtx"
    path.write_text("not a matrix market file\n3 3\n")
    with pytest.raises(ValueError, match="MatrixMarket"):
        read_matrix_market(str(path))


def test_matrix_market_unsupported_field_raises(tmp_path):
    path = tmp_path / "complex.mtx"
    path.write_text("%%MatrixMarket matrix array complex general\n2 2\n")
    with pytest.raises(ValueError, match="field"):
        read_matrix_market(str(path))


def test_matrix_market_array_symmetric(tmp_path):
    path = tmp_path / "sym.mtx"
    # Lower triangle of a symmetric 2x2 stored column-major: a11,a21,a22.
    path.write_text("%%MatrixMarket matrix array real symmetric\n2 2\n2.0\n1.0\n3.0\n")
    expected = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert np.allclose(read_matrix_market(str(path)), expected)


def test_matlab_multiple_variables_requires_name(tmp_path):
    from matrixcore.io import read_matlab, write_matlab

    path = tmp_path / "multi.mat"
    write_matlab(np.eye(2), str(path), variable_name="A")
    # Append a second variable via scipy directly.
    from scipy import io as scipy_io

    data = {k: v for k, v in scipy_io.loadmat(str(path)).items() if not k.startswith("__")}
    data["B"] = np.eye(3)
    scipy_io.savemat(str(path), data)

    with pytest.raises(ValueError, match="Multiple variables"):
        read_matlab(str(path))
    assert np.allclose(read_matlab(str(path), variable_name="B"), np.eye(3))


def test_matlab_missing_variable_raises(tmp_path):
    from matrixcore.io import read_matlab, write_matlab

    path = tmp_path / "one.mat"
    write_matlab(np.eye(2), str(path), variable_name="A")
    with pytest.raises(ValueError, match="not found"):
        read_matlab(str(path), variable_name="missing")
