.. SPDX-FileCopyrightText: 2023 - 2025 CERN
.. SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
.. SPDX-FileNotice: All rights not expressly granted are reserved.
..
.. SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

The Optimizer Interface
=======================

.. currentmodule:: cernml.optimizers

The core interface of this package consists of the following :term:`abstract
base class` with exactly one abstract method and the :doc:`dataclass
<std:library/dataclasses>` it is expected to return. Several :ref:`type aliases
<type-aliases>` are available to make the definition more easily
understandable.

Classes
-------

.. autoclass:: Optimizer
   :members:

.. autoclass:: OptimizeResult
   :members:

.. autoexception:: IgnoredArgumentWarning

Type Aliases and Type Variables
-------------------------------

.. type:: Solve
    :canonical: Callable[[Objective, NDArray[np.double]], OptimizeResult]

    The function returned by `~Optimizer.make_solve_func()`.

    To actually run the optimization procedure, you have to call this function
    with the objective function and the initial value *x*\ ₀ as arguments. It
    may raise an exception when encountering any abnormal situations **except**
    divergent behavior in the optimizer. It should also forward exception
    raised by the `~Objective` function.

.. type:: Objective
    :canonical: Callable[[NDArray[np.double]], SupportsFloat]

    The objective function that is passed to `Solve`.

    It takes an evaluation point *x* and returns the corresponding objective
    value *f*\ (*x*). The point *x* must have the same shape as the `Bounds`
    that were passed to `~Optimizer.make_solve_func()`

    The goal of optimization is to find the *x* that minimizes the
    objective.

.. type:: Bounds
    :canonical: tuple[NDArray[np.floating], NDArray[np.floating]]

    This tuple describes the *lower* and *upper* search space bounds for those
    optimization algorithms that use such bounds. Both arrays must have the
    same shape.

.. type:: AnyOptimizer
    :canonical: TypeVar(bound=Optimizer)

    :ref:`Constrained type variable <std:typing-constrained-typevar>` that
    allows to be generic over any optimizer.

    If you use a type checker, it allows code like this to pass:

    .. code-block:: python

        from cernml.optimizers import Optimizer, AnyOptimizer

        class MyOptimizer(Optimizer):
            ...

        # If the annotation were just `Optimizer`, we
        # would lose the `MyOptimizer` after this call.
        def configure(opt: AnyOptimizer) -> AnyOptimizer:
            ...
            return opt

        opt = configure(MyOptimizer())

        # Type checker knows that `opt` is still a `MyOptimizer`.
        def require_concrete(opt: MyOptimizer) -> None:
            ...

        require_concrete(opt)
