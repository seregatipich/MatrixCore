import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Define the extension module
extensions = [
    Extension(
        "matrixcore.solvers",  # This must match your module path
        ["matrixcore/solvers.pyx", "src/solvers.c"],  # Include both Cython and C source
        include_dirs=[np.get_include(), "src"],  # Include NumPy headers and your src directory
        libraries=["m"],  # Link to the math library
        language="c"  # Specify the language
    )
]

setup(
    name="MatrixCore",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),
)