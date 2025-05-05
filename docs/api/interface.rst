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

Type Aliases
------------

.. attribute:: Solve
    :type: typing.TypeAlias

    :ref:`Type alias <type-aliases>`  for functions with signature
    (*obj*: `Objective`, *x0*: `~numpy.typing.NDArray`\ [`~numpy.double`])
    → `OptimizeResult`.

    This is the function returned by `~Optimizer.make_solve_func()`. You have
    to call it to actually run the optimization procedure. It may raise an
    exception when encountering any abnormal situations except divergent
    behavior in the optimizer. Any exception raised by the `~Objective` should
    be forwarded as well.

.. attribute:: Objective
    :type: typing.TypeAlias

    :ref:`Type alias <type-aliases>` for functions with signature
    (*x*: `~numpy.typing.NDArray`\ [`~numpy.double`])
    → `~typing.SupportsFloat`.

    This is the objective function that is passed to `Solve`. It takes an
    evaluation point *x* of the same shape as the `Bounds` passed to
    `~Optimizer.make_solve_func()` and returns the corresponding objective
    value. The goal of optimization is to find the *x* that minimizes the
    objective.

.. attribute:: Bounds
    :type: typing.TypeAlias

    :ref:`Type alias <type-aliases>` of
    `tuple`\ [`~numpy.typing.NDArray`\ [`~numpy.floating`],
    `~numpy.typing.NDArray`\ [`~numpy.floating`].

    This tuple describes the lower and upper search space bounds for those
    optimization algorithms that use such bounds. Both arrays must have the
    same shape.
