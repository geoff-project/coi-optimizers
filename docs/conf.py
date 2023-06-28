# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = import-outside-toplevel
# pylint: disable = invalid-name
# pylint: disable = redefined-builtin
# pylint: disable = too-many-arguments
# pylint: disable = unused-argument

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a
full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------

from __future__ import annotations

import inspect
import pathlib
import typing as t
from importlib import import_module

from docutils import nodes
from sphinx.ext import intersphinx

try:
    import importlib_metadata
except ImportError:
    # Starting with Python 3.10 (see pyproject.toml).
    # pylint: disable = ungrouped-imports
    import importlib.metadata as importlib_metadata  # type: ignore

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


ROOTDIR = pathlib.Path(__file__).absolute().parent.parent


# -- Project information -----------------------------------------------

project = "cernml-coi-optimizers"
copyright = "2023 GSI Helmholtzzentrum für Schwerionenforschung"
author = "Nico Madysa"
release = importlib_metadata.version(project)

# -- General configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_inline_tabs",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

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

# Avoid role annotations as much as possible.
default_role = "py:obj"

# -- Options for Autodoc -----------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

# -- Options for Intersphinx -------------------------------------------


def acc_py_docs_link(repo: str) -> str:
    """A URL pointing to the Acc-Py docs server."""
    return f"https://acc-py.web.cern.ch/gitlab/{repo}/docs/stable/"


intersphinx_mapping = {
    "coi": (acc_py_docs_link("geoff/cernml-coi"), None),
    "imp": ("https://importlib-metadata.readthedocs.io/en/latest/", None),
    "mpl": ("https://matplotlib.org/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "std": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skopt": ("https://scikit-optimize.github.io/stable/", None),
    "bobyqa": ("https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/", None),
}

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = "sphinxdoc"

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css". html_static_path = ["_static"]


# -- Custom code -------------------------------------------------------


def replace_modname(modname: str) -> None:
    """Change the module that a list of objects publicly belongs to.

    This package follows the pattern to have private modules called
    :samp:`_{name}` that expose a number of classes and functions that
    are meant for public use. The parent package then exposes these like
    this::

        from ._name import Thing

    However, these objects then still expose the private module via
    their ``__module__`` attribute::

        assert Thing.__module__ == 'parent._name'

    This function iterates through all exported members of the package
    or module *modname* (as determined by either ``__all__`` or
    `vars()`) and fixes each one's module of origin up to be the
    *modname*. It does so recursively for all public attributes (i.e.
    those whose name does not have a leading underscore).
    """
    todo: t.List[t.Any] = [import_module(modname)]
    while todo:
        parent = todo.pop()
        for pubname in pubnames(parent):
            obj = inspect.getattr_static(parent, pubname)
            private_modname = getattr(obj, "__module__", "")
            if private_modname and _is_true_prefix(modname, private_modname):
                obj.__module__ = modname
                todo.append(obj)


def pubnames(obj: t.Any) -> t.Iterator[str]:
    """Return an iterator over the public names in an object."""
    return iter(
        t.cast(t.List[str], getattr(obj, "__all__", None))
        or (
            name
            for name, _ in inspect.getmembers_static(obj)
            if not name.startswith("_")
        )
    )


def _is_true_prefix(prefix: str, full: str) -> bool:
    return full.startswith(prefix) and full != prefix


replace_modname("cernml.optimizers")


def _fix_crossrefs(
    app: Sphinx, env: BuildEnvironment, node: nodes.Inline, contnode: nodes.TextElement
) -> t.Optional[nodes.Element]:
    # pylint: disable = too-many-return-statements
    if node["reftarget"] == "importlib.metadata.EntryPoint":
        node["reftarget"] = "std:entry-points"
        node["reftype"] = "ref"
        node["refdomain"] = "std"
        res = intersphinx.missing_reference(app, env, node, contnode)
        if res:
            res.children = [contnode]
        return res
    if node["reftarget"] == "importlib.metadata.Distribution":
        node["reftarget"] = "std:distributions"
        node["reftype"] = "ref"
        node["refdomain"] = "std"
        res = intersphinx.missing_reference(app, env, node, contnode)
        if res:
            res.children = [contnode]
        return res
    # Autodoc fails to resolve `Constraint` in
    # `Optimizer.make_solve_func()`.
    if node["reftarget"] == "Constraint":
        node["reftarget"] = "cernml.coi.Constraint"
        node["reftype"] = "attr"
        return intersphinx.missing_reference(app, env, node, contnode)
    # Autodoc fails to resolve `t.Sequence` in
    # `Optimizer.make_solve_func()`.
    if node["reftarget"] == "t.Sequence":
        node["reftarget"] = "typing.Sequence"
        contnode = t.cast(nodes.TextElement, nodes.Text("Sequence"))
        return intersphinx.missing_reference(app, env, node, contnode)
    if node["reftarget"] == "t.Optional":
        node["reftarget"] = "typing.Optional"
        node["reftype"] = "data"
        contnode.children = [nodes.Text("Optional")]
        return intersphinx.missing_reference(app, env, node, contnode)
    # Autodoc fails to resolve `np.ndarray` in `Bounds` and
    # `OptimizeResult`.
    if node["reftarget"].startswith("np."):
        target = node["reftarget"].split(".", 1)[1]
        node["reftarget"] = "numpy." + target
        contnode = t.cast(nodes.TextElement, nodes.Text(target))
        return intersphinx.missing_reference(app, env, node, contnode)
    return None


def setup(app: Sphinx) -> None:
    """Set up hooks into Sphinx."""
    app.connect("missing-reference", _fix_crossrefs)
    # app.connect("autodoc-before-process-signature", _fix_decorator_return_value)
