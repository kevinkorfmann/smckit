"""Sphinx configuration for smckit documentation."""

project = "smckit"
author = "Kevin Korfmann"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static", "gallery"]
html_css_files = ["custom.css"]
html_js_files = ["pretext-layout.js"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "style_external_links": False,
}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
