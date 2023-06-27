# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Definition and implementations of an ABC `Optimizer`.

This package defines `Optimizer`, a generic interface with which a
multitude of single-objective optimization algorithms can be wrapped.

It further provides wrappers for a number of third-party packages:

- :doc:`SciPy <scipy:index>`,
- :doc:`Scikit-Optimize <skopt:user_guide>`,
- :doc:`Py-BOBYQA <bobyqa:index>`,
- `CernML Extremum Seeking`_.

.. _CernML Extremum Seeking:
    https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking

These wrappers can be used either directly, or through the dynamic
*registration* feature provided by this package:

    >>> from cernml.optimizers import make
    >>> make("BOBYQA")
    <cernml.optimizers.bobyqa.Bobyqa object at ...>

The recommended way to register your own optimizers is statically as
`entry points`_. This guarantees that downstream applications will find
them as long as their package is installed:

.. tab:: pyproject.toml

    .. code-block:: toml

        [project.entry-points."cernml.optimizers"]
        MyOpt-v1 = "my_package.my_optimizer:MyOptimizer"

.. tab:: setup.cfg

    .. code-block:: ini

        [options.entry_points]
        cernml.optimizers =
            MyOpt-v1 = my_package.my_optimizer:MyOptimizer

.. tab:: setup.py

    .. code-block:: python

        setup(
            "my-distribution",
            # ...,
            entry_points={
                "cernml.optimizers": [
                    "MyOpt-v1 = my_package.my_optimizer:MyOptimizer",
                ],
            },
        )

Alternatively, you can also register your optimizers dynamically via
`register()`. In this case, they only become available once the
registering module has been imported:

.. tab:: string reference

    .. code-block:: python

        # my_package/my_optimizer.py
        from cernml.optimizers import Optimizer

        class MyOptimizer(Optimizer):
            ...

    .. code-block:: python

        # my_package/__init__.py

        from cernml.optimizers import register

        register("MyOpt-v1", "my_package.my_optimizer:MyOptimizer")

.. tab:: class object

    .. code-block:: python

        # my_package/my_optimizer.py
        from cernml.optimizers import Optimizer, register

        class MyOptimizer(Optimizer):
            ...

        register("MyOpt-v1", MyOptimizer)

    .. code-block:: python

        # my_package/__init__.py
        from .my_optimizer import MyOptimizer

If you pass a string reference, the optimizer module is only imported
when the user calls `make()`. If you pass the optimizer class directly,
it must be available at registration time.

Note that in any case, the *entry point name* (in this example
*MyOpt-v1*) should be unique among all installed packages.

.. _entry points: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from ._interface import (
    AnyOptimizer,
    Bounds,
    Objective,
    Optimizer,
    OptimizeResult,
    SolveFunc,
)
from ._registration import (
    EP_GROUP,
    DuplicateOptimizerWarning,
    OptimizerNotFound,
    OptimizerSpec,
    OptimizerWithSpec,
    TypeWarning,
    make,
    register,
    registry,
    spec,
)

__all__ = [
    "EP_GROUP",
    "AnyOptimizer",
    "Bounds",
    "DuplicateOptimizerWarning",
    "Objective",
    "Optimizer",
    "OptimizeResult",
    "OptimizerNotFound",
    "OptimizerSpec",
    "OptimizerWithSpec",
    "SolveFunc",
    "TypeWarning",
    "make",
    "register",
    "registry",
    "spec",
]
