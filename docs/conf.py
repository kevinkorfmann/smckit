"""Sphinx configuration for smckit documentation."""

project = "smckit"
author = "Kevin Korfmann"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static", "gallery"]

myst_enable_extensions = ["colon_fence"]
