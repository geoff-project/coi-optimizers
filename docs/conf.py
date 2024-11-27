# SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
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

import inspect
import pathlib
import sys
import typing as t
from importlib import import_module

from docutils import nodes
from sphinx import addnodes
from sphinx.ext import intersphinx

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

if t.TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


ROOTDIR = pathlib.Path(__file__).absolute().parent.parent


# -- Project information -----------------------------------------------

project = "cernml-coi-optimizers"
dist = metadata.distribution(project)

copyright = "2023-2024 GSI Helmholtzzentrum für Schwerionenforschung"
author = "Nico Madysa"
release = dist.version
version = release.partition("+")[0]
html_last_updated_fmt = "%b %d %Y"

for entry in dist.metadata.get_all("Project-URL", []):
    kind, url = entry.split(", ")
    if kind == "gitlab":
        gitlab_url = url
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
extensions = [
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

# Avoid role annotations as much as possible.
default_role = "py:obj"

# Use one line per argument for long signatures.
maximum_signature_line_length = 88

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = "python_docs_theme"
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
napoleon_use_ivar = True

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
}

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
    todo: list[t.Any] = [import_module(modname)]
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
        t.cast(list[str], getattr(obj, "__all__", None))
        or (
            name
            for name, _ in inspect.getmembers_static(obj)
            if not name.startswith("_")
        )
    )


def _is_true_prefix(prefix: str, full: str) -> bool:
    return full.startswith(prefix) and full != prefix


replace_modname("cernml.coi")
replace_modname("cernml.optimizers")


def retry_internal_xref(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> nodes.reference | None:
    """Retry a failed Python reference with laxer requirements.

    Autodoc often tries to look up type aliases as classes even though
    they're classified as data. You can catch those cases and forward
    them to `retry_internal_xref()`, which will look them up with the
    more general `py:obj` role. This is more likely to find them.
    """
    domain = env.domains[node["refdomain"]]
    return domain.resolve_xref(
        env, node["refdoc"], app.builder, "obj", node["reftarget"], node, contnode
    )


def adjust_pending_xref(
    **kwargs: t.Any,
) -> t.Callable[
    [Sphinx, BuildEnvironment, addnodes.pending_xref, nodes.TextElement],
    nodes.reference | None,
]:
    """Return a function that can fix a certain broken cross reference.

    The returned function can be used as a ``missing-reference``
    handler. It will take the ``pending_xref`` that failed to resolve
    and will adjust its attributes as given by the arguments to this
    function. It will then resolve it again using Intersphinx.
    """

    def _replace_text_node(node: nodes.reference, new: str) -> None:
        [text] = node.findall(nodes.Text)
        parent = text.parent
        assert parent
        parent.replace(text, nodes.Text(new))

    def _inner(
        app: Sphinx,
        env: BuildEnvironment,
        node: addnodes.pending_xref,
        contnode: nodes.TextElement,
    ) -> nodes.reference | None:
        node.update_all_atts(kwargs, replace=True)
        res = intersphinx.missing_reference(app, env, node, contnode)
        if res:
            # `intersphinx.missing_reference()` may change the inner
            # text. Replace it with the text that we want. (The text
            # that we want is the original minus all leading module
            # names.)
            target = contnode.astext().rsplit(".")[-1]
            _replace_text_node(res, target)
        return res

    return _inner


crossref_fixers = {
    # Neither stdlib nor `importlib_metadata` provide full API docs for
    # EntryPoint and Distribution. Thus, we simply link to the
    # corresponding user guide entries. Note that when resolving these
    # references, Intersphinx *always* ignores the contnode and inserts
    # its own text. Thus, we have to be very forceful when fixing it up.
    "importlib.metadata.EntryPoint": adjust_pending_xref(
        reftarget="std:entry-points", refdomain="std", reftype="ref"
    ),
    "importlib.metadata.Distribution": adjust_pending_xref(
        reftarget="std:distributions", refdomain="std", reftype="ref"
    ),
    # Autodoc thinks this is a class, but it's not. (It *should* be
    # data, but no, it's an attribute.)
    "Constraint": adjust_pending_xref(
        reftarget="cernml.coi.Constraint", reftype="data"
    ),
    # Autodoc is very bad at resolving `typing` members, so we need to
    # give it a push.
    "t.Sequence": adjust_pending_xref(reftarget="typing.Sequence"),
    "t.Optional": adjust_pending_xref(reftarget="typing.Optional", reftype="data"),
    # Autodoc fails to resolve type annotations on named tuples and data
    # classes.
    "NDArray": adjust_pending_xref(reftarget="numpy.typing.NDArray", reftype="data"),
    "np.double": adjust_pending_xref(reftarget="numpy.double"),
}


def fix_all_crossrefs(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> nodes.Element | None:
    """Handler for all missing references."""
    fixer = crossref_fixers.get(node["reftarget"])
    if fixer:
        return fixer(app, env, node, contnode)
    return retry_internal_xref(app, env, node, contnode)


def setup(app: Sphinx) -> None:
    """Set up hooks into Sphinx."""
    app.connect("missing-reference", fix_all_crossrefs)
