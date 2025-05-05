# SPDX-FileCopyrightText: 2023 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a
full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata


ROOTDIR = Path(__file__).absolute().parent.parent


# -- Project information -----------------------------------------------

project = "cernml-coi-optimizers"
dist = metadata.distribution(project)

copyright = "2023-2025 GSI Helmholtzzentrum für Schwerionenforschung"
author = "Penny Madysa"
release = dist.version
version = release.partition("+")[0]

for entry in dist.metadata.get_all("Project-URL", []):
    url: str
    kind, url = entry.split(", ")
    if kind == "gitlab":
        gitlab_url = url.removesuffix("/")
        license_url = f"{gitlab_url}-/blob/master/COPYING"
        issues_url = f"{gitlab_url}/-/issues"
        break
else:
    gitlab_url = ""
    license_url = ""
    issues_url = ""

# -- General configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
sys.path.append(str(Path("./_ext").resolve()))
extensions = [
    "fix_napoleon_attributes_type",
    "fix_xrefs",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_inline_tabs",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".DS_Store",
    "Thumbs.db",
    "_build",
]

# Don't repeat the class name for methods and attributes in the page
# table of content of class API docs.
toc_object_entries_show_parents = "hide"

# A list of prefixes that are ignored for sorting the Python module
# index.
modindex_common_prefix = ["cernml.", "cernml.optimizers."]

# Avoid role annotations as much as possible.
default_role = "py:obj"

# Use one line per argument for long signatures.
maximum_signature_line_length = 89

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = "python_docs_theme"
html_last_updated_fmt = "%b %d %Y"
html_theme_options = {
    "root_url": "https://acc-py.web.cern.ch/",
    "root_name": "Acc-Py Documentation server",
    "license_url": license_url,
    "issues_url": issues_url,
}
templates_path = ["./_theme/"]

# -- Options for Autodoc -----------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_type_aliases = {
    "Bounds": "cernml.optimizers.Bounds",
    "Objective": "cernml.optimizers.Objective",
    "Solve": "~cernml.optimizers.Solve",
    "NDArray": "~numpy.typing.NDArray",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for Intersphinx -------------------------------------------


def acc_py_docs_link(repo: str) -> str:
    """A URL pointing to the Acc-Py docs server."""
    return f"https://acc-py.web.cern.ch/gitlab/{repo}/docs/stable/"


intersphinx_mapping = {
    "coi": (acc_py_docs_link("geoff/cernml-coi"), None),
    "ces": (acc_py_docs_link("geoff/optimizers/cernml-extremum-seeking"), None),
    "imp": ("https://importlib-metadata.readthedocs.io/en/latest/", None),
    "mpl": ("https://matplotlib.org/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "std": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skopt": ("https://scikit-optimize.github.io/stable/", None),
    "bobyqa": ("https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/", None),
    "xopt": ("https://xopt.xopt.org/", None),
}


# -- Options for custom extension FixXrefs -----------------------------


fix_xrefs_try_typing = True
fix_xrefs_try_class_as_obj = True
fix_xrefs_rules = [
    {"pattern": r"^Constraint$", "reftarget": ("const", "cernml.coi.Constraint")},
    {"pattern": r"^NDArray$", "reftarget": ("const", "numpy.typing.NDArray")},
    {"pattern": r"^np\.", "reftarget": ("sub", "numpy."), "contnode": ("sub", "")},
    {
        "pattern": r"^t\.",
        "reftarget": ("sub", "typing."),
        "contnode": ("sub", ""),
    },
]
