..
    SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Integration with the COI
========================

.. currentmodule:: cernml.optimizers

The following functions are simple wrappers that make it more convenient to use
`Optimizer` with classes that implement the :doc:`Common Optimization
Interfaces <coi:index>`.

.. autofunction:: solve

.. autofunction:: make_solve_func

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
