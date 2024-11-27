# SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Defines the optimizer registry."""

from __future__ import annotations

import sys
import typing as t
import warnings

from ._interface import Optimizer

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

if t.TYPE_CHECKING:
    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self

__all__ = (
    "EP_GROUP",
    "DuplicateOptimizerWarning",
    "OptimizerNotFound",
    "OptimizerSpec",
    "OptimizerWithSpec",
    "TypeWarning",
    "make",
    "register",
    "registry",
    "spec",
)

EP_GROUP = "cernml.optimizers"


class TypeWarning(Warning):
    """An object does not have the expected type, but might still work."""


class DuplicateOptimizerWarning(Warning):
    """Two entry points have chosen the same name."""


class OptimizerNotFound(KeyError):
    """A requested optimizer was not found in the registry."""


registry: dict[str, OptimizerSpec] = {}


class OptimizerWithSpec(Optimizer):
    """Subclass of `Optimizer` guaranteed to have `spec` set.

    This is only used for type hinting purposes. You should not subclass
    this class yourself.
    """

    # pylint: disable = too-few-public-methods

    spec: OptimizerSpec

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:  # noqa: ARG002
        raise TypeError("this class is for type checking only, do not instantiate it")


class OptimizerSpec:
    """Specification of an optimizer.

    You should not instantiate this class yourself. It is meant as the
    value type of the `registry` dict.

    There are three ways to specify an optimizer:

    1. By calling `register()` with a subclass of `Optimizer`;
    2. By calling `register()` with a string like
       :samp:`"{module}:{attr}"` that points to a subclass of
       `Optimizer`;
    3. By defining an `entry point`_ in the group *cernml.optimizers*
       that points at a subclass of `Optimizer`.

    In case 1, the spec contains a reference to the subclass. You can
    get it by calling `load()`, which will not actually do any loading.

    In cases 2 and 3, calling `load()` will dynamically look up the
    class by importing the given module and looking up the attr in it.
    The result is then cached in the spec and returned.

    .. _entry point: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """

    def __init__(self, entry_point: metadata.EntryPoint) -> None:
        self._ep = entry_point
        self._type: type[Optimizer] | None = None

    @classmethod
    def from_optimizer(
        cls,
        name: str,
        optimizer: type[Optimizer],
        dist: metadata.Distribution | None = None,
    ) -> Self:
        """Create a spec for a loaded optimizer class."""
        value = f"{optimizer.__module__}:{optimizer.__qualname__}"
        entry_point = metadata.EntryPoint(name=name, value=value, group=EP_GROUP)
        if dist:
            vars(entry_point)["dist"] = dist
        self = cls(entry_point)
        self._type = optimizer
        return self

    def __str__(self) -> str:
        name = f"{self.dist.name}/{self.name}" if self.dist else self.name
        value = str(self._type) if self._type else repr(self.value)
        return f"<{type(self).__name__}({name!r}, {value})>"

    @property
    def name(self) -> str:
        """The registered name of the optimizer.

        This name is unique among all available optimizers. If two
        optimizers share the same name, a `DuplicateOptimizerWarning` is
        issued and the one loaded later wins.
        """
        return self._ep.name

    @property
    def value(self) -> str:
        """A string that points at the optimizer class.

        For optimizers defined dynamically via `register()`, this string
        is not necessarily correct.
        """
        return self._ep.value

    @property
    def dist(self) -> metadata.Distribution | None:
        """If available, the distribution_ providing the optimizer.

        Typically, this is `None` if the optimizer was registered
        dynamically via `register()`.

        .. _distribution: https://packaging.python.org/en/latest/glossary/#term-Distribution-Package
        """
        return self._ep.dist

    def load(self) -> type[Optimizer]:
        """Dynamically load the optimizer class, if necessary.

        If the optimizer class has not been loaded yet, this imports the
        necessary package and loads it. If it has already been loaded
        (or registered dynamically by passing a `type` object to
        `register()`), this just returns the optimizer class.

        Raises:
            TypeError: if the object loaded dynamically is not a
                subclass of `Optimizer`.
        """
        if self._type:
            return self._type
        self._type = result = self._ep.load()
        _warn_if_not_optimizer(result)
        return result

    def make(self, *args: t.Any, **kwargs: t.Any) -> OptimizerWithSpec:
        """Create a new optimizer instance.

        This loads the optimizer class via `load()` and creates a new
        instance. All arguments are forwarded to the initializer.
        """
        optimizer = self.load()
        result = optimizer(*args, **kwargs)
        return self._add_spec(result)

    def _add_spec(self, opt: Optimizer) -> OptimizerWithSpec:
        opt.spec = self
        return t.cast(OptimizerWithSpec, opt)


def spec(name: str) -> OptimizerSpec:
    """Return the spec of the optimizer registered under *name*.

    Raises:
        OptimizerNotFound: if no optimizer was registered under that
            name.
    """
    try:
        return registry[name]
    except KeyError:
        raise OptimizerNotFound(name) from None


def make(name: str, *args: t.Any, **kwargs: t.Any) -> OptimizerWithSpec:
    """Create an instance of the optimizer registered under *name*.

    The optimizer type is looked up via `spec()`. Upon success, the
    optimizer is loaded and instantiated via `OptimizerSpec.make()`. All
    arguments beyond *name* are forwarded to the initializer.
    """
    return spec(name).make(*args, **kwargs)


def register(
    name: str,
    optimizer: str | type[Optimizer],
    dist: metadata.Distribution | None = None,
) -> None:
    """Dynamically register a new `Optimizer`.

    .. note::
        Consider defining an `entry point`_ in the group
        *cernml.optimizers* instead.

    This is usually called directly after the definition of a new
    subclass of `Optimizer`. You can either pass the type object
    directly:

        >>> class MyOptimizer(Optimizer):
        ...     def make_solve_func(self, *args, **kwargs): ...
        >>> register("MyOptimizer", MyOptimizer)

    Or you can pass a string that points at the subclass in the usual
    `entry point`_ syntax:

        >>> register("My Optimizer", "package.module:variable.nested")

    When calling `make()`, the above example would first ``import
    package``, then ``import package.module``, then look up
    ``package.module.variable.nested``.

    If *dist* is passed, it's the
    :class:`importlib.metadata.Distribution` that the entry point
    belongs to. You usually get it by calling
    :samp:`importlib.metadata.distribution("{name in setup.py file}")`.

    .. _entry point: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    """
    if isinstance(optimizer, str):
        entry_point = metadata.EntryPoint(name, optimizer, EP_GROUP)
        if dist:
            vars(entry_point)["dist"] = dist
        new_spec = OptimizerSpec(entry_point)
    elif callable(optimizer):
        _warn_if_not_optimizer(optimizer)
        new_spec = OptimizerSpec.from_optimizer(name, optimizer, dist)
    else:
        raise TypeError(
            f"optimizer {name} must be a string or "
            f"an Optimizer subclass, not {optimizer!r}"
        )
    if name in registry:
        _warn_duplicate(prev=registry[name], new=new_spec)
    registry[name] = new_spec


def _warn_if_not_optimizer(opt: type[Optimizer]) -> None:
    if not isinstance(opt, type):
        warnings.warn(f"not a type: {opt!r}", TypeWarning, stacklevel=3)
    elif not issubclass(opt, Optimizer):
        warnings.warn(
            f"not a subclass of Optimizer: {opt!r}", TypeWarning, stacklevel=3
        )


def _warn_duplicate(prev: OptimizerSpec, new: OptimizerSpec) -> None:
    warnings.warn(
        f"overriding {prev!s} with {new!s} for name {new.name!r}",
        DuplicateOptimizerWarning,
        stacklevel=3,
    )


def __init() -> None:
    """Load all optimizers declared statically via entry points.

    If two entry points from different distributions share the same
    naming, a warning is printed.

    This function is called upon import of this package.
    """
    for entry_point in metadata.entry_points(group=EP_GROUP):
        name = entry_point.name
        new_spec = OptimizerSpec(entry_point)
        if name in registry:
            _warn_duplicate(prev=registry[name], new=new_spec)  # pragma: no cover
        registry[name] = new_spec


__init()
