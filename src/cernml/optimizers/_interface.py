# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Defines the `Optimizer` interface."""

from __future__ import annotations

import abc
import dataclasses
import typing as t

import numpy as np

if t.TYPE_CHECKING:
    import sys

    from numpy.typing import NDArray

    from cernml.coi import Constraint

    from ._registration import OptimizerSpec

    if sys.version_info < (3, 10):
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias


__all__ = (
    "AnyOptimizer",
    "Bounds",
    "IgnoredArgumentWarning",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "Solve",
)


class IgnoredArgumentWarning(Warning):
    """An optimization algorithm ignores one of the given arguments.

    Most often, this affects either the `~cernml.coi.Constraint` list or
    the `Bounds` passed to `~Optimizer.make_solve_func()`.

    Example:
        Given the following optimization problem:

        >>> def objective(x):
        ...     return np.linalg.norm(x)
        >>> x0 = np.zeros(3)
        >>> bounds = (x0 - 2.0, x0 + 2.0)

        and the `BOBYQA <cernml.optimizers.bobyqa.Bobyqa>` optimization
        algorithm:

        >>> from cernml.optimizers import make
        >>> opt = make("BOBYQA")

        passing a constraint:

        >>> from scipy.optimize import LinearConstraint
        >>> c = LinearConstraint(np.diag(np.ones(3)), 1.0, np.inf)

        will raise this warning:

        >>> import warnings
        >>> warnings.simplefilter("error")
        >>> solve = opt.make_solve_func(bounds, constraints=[c])
        Traceback (most recent call last):
        ...
        IgnoredArgumentWarning: BOBYQA ignores constraints

        `COBYLA <cernml.optimizers.scipy.Cobyla>`, on the other hand,
        takes constraints into account.

        >>> warnings.simplefilter("error")
        >>> opt = make("COBYLA")
        >>> solve = opt.make_solve_func(bounds, constraints=[c])
        >>> res = solve(objective, x0)
        >>> res.success
        True
        >>> res.x
        array([1., 1., 1.])
    """


@dataclasses.dataclass
class OptimizeResult:
    """Summary of the optimization as returned by `Solve`.

    This :doc:`dataclass <std:library/dataclasses>` is modeled after
    `scipy.optimize.OptimizeResult`, but as a dataclass instead of
    a dict. Furthermore, it lacks all attributes related to derivatives
    (Jacobians, Hessians).

    Attributes:
        x: The solution of the optimization.
        fun: The objective function at x.
        success: If True, the optimizer exited successfully.
        message: Description of the cause of the termination.
        nit: The number of iterations performed by the optimizer.
        nfev: The number of evaluations of the objective function.
    """

    x: NDArray[np.double]
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


Objective: TypeAlias = t.Callable[["NDArray[np.double]"], t.SupportsFloat]

Solve: TypeAlias = t.Callable[[Objective, "NDArray[np.floating]"], OptimizeResult]

Bounds: TypeAlias = tuple["NDArray[np.floating]", "NDArray[np.floating]"]

AnyOptimizer = t.TypeVar("AnyOptimizer", bound="Optimizer")


class Optimizer(abc.ABC):
    """:term:`Abstract base class` for single-objective optimizers.

    This class follows the Builder_ pattern to create a `Solve`
    function. The goal of the solve function is to find the minimum of
    a given `Objective` function. In the simplest case, you use it via
    `make()` and `solve()` as follows:

    .. code-block:: python
        :linenos:

        class ConcreteOptimizer(Optimizer):
            ...

        def MyEnv(coi.SingleOptimizable):
            ...

        opt = ConcreteOptimizer(...)
        env = MyEnv()
        res = solve(opt, env)
        print("Minimum f(x*) = {res.fun} at x* = {res.x}")

    .. _builder: https://en.wikipedia.org/wiki/Builder_pattern

    Subclasses should contain no logic other than containing and
    validating all hyper-parameters of the optimization algorithm. When
    calling `make_solve_func()`, it should bind the hyper-parameters to
    the function that executes the algorithm.

    Subclasses should allow setting the hyper parameters in three ways:

    1. as parameters to the initializer;
    2. as attributes on the optimizer instance;
    3. via the `~cernml.coi.Configurable` API.

    The :doc:`/examples/index` show how to
    :doc:`/examples/implement_an_optimizer`.

    Attributes:
        spec: This optimizer's entry in the `registry` if it was created
            via `make()`, otherwise `std:None`.
    """

    spec: OptimizerSpec | None = None

    @abc.abstractmethod
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> Solve:
        """Create a new solve function.

        Subclasses of `Optimizer` should override this method. Users can
        call this method directly, or use the convenience wrapper
        `~cernml.optimizers.make_solve_func()`.

        Args:
            bounds: A named tuple of the lower and upper bound of the
                search space. Both are 1D arrays of the same length *N*.
            constraints: A list of soft constraints that the optimizer
                should uphold. See `~scipy.optimize.LinearConstraint`
                and `~scipy.optimize.NonlinearConstraint` for details.
        """
        raise NotImplementedError
