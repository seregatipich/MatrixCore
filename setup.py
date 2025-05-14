from setuptools import setup, find_packages, Extension
import numpy
import os

# Get the path to the src directory
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

# List all C source files in src directory
c_sources = [os.path.join(src_dir, 'solvers.c')]  # Add any other C files needed

# Define Cython extension
extensions = [
    Extension(
        "matrixcore.solver",  # Name of the extension
        ["matrixcore/solver.pyx"] + c_sources,  # Source files - include both Cython and C files
        include_dirs=[
            numpy.get_include(),
            src_dir,  # Add the src directory to include path
        ],
    )
]

setup(
    name="matrixcore",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=["numpy"],
    setup_requires=["cython", "numpy"],
)