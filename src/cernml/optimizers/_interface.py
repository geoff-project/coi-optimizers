# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Defines the `Optimizer` interface."""

from __future__ import annotations

import abc
import dataclasses
import typing as t

import numpy as np

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = unused-import, ungrouped-imports
    from cernml.coi import Constraint

    from ._registration import OptimizerSpec

__all__ = [
    "AnyOptimizer",
    "Bounds",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "SolveFunc",
]


class Bounds(t.NamedTuple):
    """Lower and upper search space bounds as a named tuple.

    The names has been chosen to be compatible with
    `scipy.optimize.Bounds`.
    """

    lb: np.ndarray
    ub: np.ndarray


@dataclasses.dataclass
class OptimizeResult:
    """A summary of the optimization procedure.

    Attributes:
        x: The solution of the optimization.
        fun: The objective function at x.
        success: If True, the optimizer exited successfully.
        message: Description of the cause of the termination
        nit: The number of iterations performed by the optimizer.
        nfev: The number of evaluations of the objective function.
    """

    x: np.ndarray  # pylint: disable=invalid-name
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


Objective = t.Callable[[np.ndarray], float]
"""Type alias for the objective function *x* -> *f*(*x*)."""

SolveFunc = t.Callable[[Objective, np.ndarray], OptimizeResult]
"""Type alias for the solve function."""


AnyOptimizer = t.TypeVar("AnyOptimizer", bound="Optimizer")
"""Constrained type variable of anything that implements `Optimizer`."""


class Optimizer(abc.ABC):
    """The central definition of a single-objective optimizer.

    An optimizer, for this package, is a builder_ for a *solve
    function*. The solve function must have a well-defined interface: It
    accepts an *objective function* and an *initial point x₀* and
    return an `OptimizeResult`.

    .. _builder: https://en.wikipedia.org/wiki/Builder_pattern

    An objective function accepts a parameter *x* and returns the
    objective value that shall be minimized by the solver.

    The purpose of the optimizer is to contain all hyper parameters of
    the optimization algorithm. When building the solve function, all of
    these parameters should be bound to the actual function that
    executes the algorithm.

    The optimizer should allow setting the hyper parameters:

    1. as parameters to the initializer;
    2. as attributes on the optimizer instance;
    3. via the `cernml.coi.Configurable` API.

    Example:

        >>> from cernml.coi import Config, Configurable, ConfigValues
        >>> from gym.spaces import Box
        >>> class RandomSearchOptimizer(Optimizer, Configurable):
        ...     def __init__(self, maxfun: int = 100) -> None:
        ...         self.maxfun = maxfun
        ...     def get_config(self) -> Config:
        ...         return Config().add(
        ...             "maxfun", self.maxfun, range=(0, 1_000_000)
        ...         )
        ...     def apply_config(self, values: ConfigValues) -> None:
        ...         self.maxfun = values.maxfun
        ...     def make_solve_func(
        ...         self,
        ...         bounds: Bounds,
        ...         constraints: t.Sequence[Constraint],
        ...     ) -> SolveFunc:
        ...         space = Box(bounds.lb, bounds.ub)
        ...         def is_valid(x: np.ndarray) -> bool:
        ...             return all(c(x) >= 0 for c in constraints)
        ...         def solve_func(
        ...             objective: Objective, x0: np.ndarray
        ...         ) -> OptimizeResult:
        ...             best_x = x0
        ...             best_o = objective(x0)
        ...             for _ in range(1, self.maxfun):
        ...                 next_x = space.sample()
        ...                 valid = is_valid(next_x)
        ...                 next_o = objective(next_x)
        ...                 if valid and not next_o >= best_o:
        ...                     best_x, best_o = next_x, next_o
        ...             return OptimizeResult(
        ...                 x=best_x,
        ...                 fun=best_o,
        ...                 success=True,
        ...                 message="",
        ...                 nit=self.maxfun,
        ...                 nfev=self.maxfun,
        ...             )
        ...         return solve_func
        >>> from . import register, make
        >>> register("RandomSearch", RandomSearchOptimizer)
        >>> make("RandomSearch")
        <...RandomSearchOptimizer object at ...>
    """

    # pylint: disable = too-few-public-methods

    spec: t.Optional[OptimizerSpec] = None
    """The optimizers `~cernml.optimizers.registry` entry.

    If the optimizer was not created through
    `~cernml.optimizers.make()`, but instead through e.g. direct
    instantiation, this is `std:None`.
    """

    @abc.abstractmethod
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        """Create a new solve function.

        Subclasses of `Optimizer` should override this method.

        Args:
            bounds: A named tuple of the lower and upper bound of the
                search space. Both are 1D arrays of the same length *N*.
            constraints: A list of soft constraints that the optimizer
                should uphold. See `~scipy.optimize.LinearConstraint`
                and `~scipy.optimize.NonlinearConstraint` for details.
        """
        raise NotImplementedError()
