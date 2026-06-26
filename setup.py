import sys

from Cython.Build import cythonize
from setuptools import Extension, setup


def numpy_include():
    import numpy

    return numpy.get_include()


math_libraries = [] if sys.platform == "win32" else ["m"]

extensions = [
    Extension(
        "matrixcore.solvers",
        ["matrixcore/solvers.pyx", "src/solvers.c"],
        include_dirs=[numpy_include(), "src"],
        libraries=math_libraries,
        language="c",
    )
]

setup(ext_modules=cythonize(extensions, language_level=3))
