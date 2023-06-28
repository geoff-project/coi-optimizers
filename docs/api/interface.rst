..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

The Optimizer Interface
=======================

.. currentmodule:: cernml.optimizers

.. autoclass:: Optimizer
   :members:

.. class:: SolveFunc(obj: Objective, x0: np.ndarray) -> OptimizeResult

    :ref:`Type alias <type-aliases>`  for the return type of
    `~Optimizer.make_solve_func()`.

    This function is called to actually run the optimization procedure. It may
    raise an exception when encountering any abnormal situations except
    divergent behavior in the optimizer. Any exception raised by the
    `~Objective` should be forwarded as well.

.. class:: Objective(x: ~numpy.ndarray) -> float

    :ref:`Type alias <type-aliases>` for the objective function passed to
    `~cernml.optimizers.SolveFunc`.

    This function takes an evaluation point *x* of the same shape as the
    `Bounds` passed to `~Optimizer.make_solve_func()` and returns the
    corresponding objective value. The goal of optimization is to find the *x*
    that minimizes the objective.

.. autoclass:: Bounds

.. autoclass:: OptimizeResult
   :members:

.. data:: AnyOptimizer
    :type: typing.TypeVar

    :ref:`Constrained type variable <std:typing-constrained-typevar>` that
    allows to be generic over any `Optimizer`.

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
