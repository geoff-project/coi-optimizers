# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Definition and implementations of an ABC `Optimizer`.

This package defines `Optimizer`, a generic interface with which a
multitude of single-objective optimization algorithms can be wrapped.

It further provides wrappers for a number of third-party packages:

- :doc:`SciPy <scipy:index>`,
- :doc:`Scikit-Optimize <skopt:index>`,
- :doc:`Py-BOBYQA <bobyqa:index>`,
- :doc:`CernML Extremum Seeking <cernml-es:index>`.

These wrappers can be used either directly, or through the dynamic
*registration* feature provided by this package:

    >>> from cernml.optimizers import make
    >>> make("BOBYQA")
    <cernml.optimizers.bobyqa.Bobyqa object at ...>

You can register your own optimizers either statically as `entry
points`_ …:

.. code-block: toml

    # pyproject.toml
    [project.entry-points."cernml.optimizers"]
    MyOpt-v1 = "my_package.my_optimizer:MyOptimizer"

.. code-block: ini

    # setup.cfg
    [options.entry_points]
    cernml.optimizers =
        MyOpt-v1 = my_package.my_optimizer:MyOptimizer

.. code-block: python

    # setup.py
    ...
    setup(
        "my-distribution",
        # ...,
        entry_points={
            "cernml.optimizers": [
                "MyOpt-v1 = my_package.my_optimizer:MyOptimizer",
            ],
        },
    )

… or you can register them dynamically via the `register()` function:

.. code-block: python

    # my_package/my_optimizer.py
    from cernml.optimizers import Optimizer

    class MyOptimizer(Optimizer):
        ...

.. code-block: python

    # my_package/__init__.py

    from cernml.optimizers import register

    register("MyOpt-v1", "my_package.my_optimizer:MyOptimizer")

You can also pass the optimizer class directly, if it is available:

.. code-block: python

    # my_package/__init__.py
    from cernml.optimizers import register

    from .my_optimizer import MyOptimizer

    register("MyOpt-v1", MyOptimizer)

Note that that the *entry point name* (in this case *MyOpt-v1*) should
be unique among all installed packages.

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
    "OptimizeResult",
    "Optimizer",
    "OptimizerNotFound",
    "OptimizerSpec",
    "OptimizerWithSpec",
    "SolveFunc",
    "make",
    "register",
    "registry",
    "spec",
]
