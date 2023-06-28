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

    Type alias for the return type of
    `~cernml.optimizers.Optimizer.make_solve_func()`.

.. class:: Objective(x: ~numpy.ndarray) -> float

    Type alias for the objective function that a user should pass to the
    `~cernml.optimizers.SolveFunc`.

    This is the function whose return value should be minimized by the
    optimization algorithm. It should be defined on the whole domain given by
    the `~cernml.optimizers.Bounds`.

.. autoclass:: Bounds

.. autoclass:: OptimizeResult
   :members:

.. class:: AnyOptimizer

    A constrained type variable that allows to be generic over any `Optimizer`.

    If you use a type checker, it allows code like this to pass:

    .. code-block:: python

        from cernml.optimizers import Optimizer, AnyOptimizer

        class MyOptimizer(Optimizer):
            ...

        def configure(opt: AnyOptimizer) -> AnyOptimizer:
            ...
            return opt

        opt = configure(MyOptimizer())  # opt still has type `MyOptimizer!`
