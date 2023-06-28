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
    """Summary of the optimization as returned by `SolveFunc`.

    This is a :doc:`dataclass <std:library/dataclasses>`.

    Attributes:
        x: The solution of the optimization.
        fun: The objective function at x.
        success: If True, the optimizer exited successfully.
        message: Description of the cause of the termination.
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

    This :term:`abstract base class` follows the Builder_ pattern to
    create a `solve function <cernml.optimizers.SolveFunc>`_. In the
    simplest case, it is used as follows:

    .. code-block:: python
        :linenos:

        class ConcreteOptimizer(Optimizer):
            ...

        def objective(x):
            ...

        opt = ConcreteOptimizer(...)
        solve = opt.make_solve_func(bounds, constraints)
        res = solve(objective, x0)
        print("Minimum f(x*) = {res.fun} at x* = {res.x}")

    .. _builder: https://en.wikipedia.org/wiki/Builder_pattern

    The purpose of this class is to contain all hyper-parameters of the
    optimization algorithm. When building the `solve function
    <cernml.optimizers.SolveFunc>`_, these hyper-parameters should be
    bound to the function that executes the algorithm.

    The optimizer should allow setting the hyper parameters:

    1. as parameters to the initializer;
    2. as attributes on the optimizer instance;
    3. via the `cernml.coi.Configurable` API.

    The :doc:`/examples/index` show how to
    :doc:`/examples/implement_an_optimizer`.

    Attributes:
        spec: This optimizer's entry in the `registry` if it was created
            via `make()`, otherwise `std:None`.
    """

    # pylint: disable = too-few-public-methods

    spec: t.Optional[OptimizerSpec] = None

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
