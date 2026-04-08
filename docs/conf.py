"""Sphinx configuration for smckit documentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _render_method_status_matrix() -> None:
    data = json.loads((SRC / "smckit" / "data" / "method_status.json").read_text(encoding="utf-8"))
    target_dir = ROOT / "docs" / "_generated"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "method_status_matrix.rst"

    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Method",
        "     - Upstream",
        "     - Native",
        "     - Native default eligible",
        "     - Tracked agreement",
        "     - Notes",
    ]
    for entry in data:
        eligible = "✓" if entry["native_default_eligible"] else "✗"
        lines.extend(
            [
                f"   * - {entry['display_name']}",
                f"     - {entry['upstream']}",
                f"     - {entry['native']}",
                f"     - {eligible}",
                f"     - `{entry['tracked_agreement']}`",
                f"     - {entry['notes']}",
            ]
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


_render_method_status_matrix()

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
