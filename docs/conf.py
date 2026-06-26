"""Sphinx configuration for the MatrixCore documentation."""

from importlib.metadata import version as get_version

project = "MatrixCore"
copyright = "2025, Sergei Poluektov"
author = "Sergei Poluektov"

try:
    release = get_version("matrixcore")
except Exception:  # pragma: no cover - docs built from a source tree without install
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"MatrixCore {release}"

autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
