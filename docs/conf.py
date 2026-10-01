from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "Miletos"
author = "Miletos Collaboration"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx"]
autodoc_mock_imports = ["celerite", "ephesos", "lygos", "miletos.main", "pcat"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
