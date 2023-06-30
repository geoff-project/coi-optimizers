# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Definition, implementation and instantiation of `Optimizer`.

This package defines `Optimizer`, a generic interface with which a
multitude of single-objective optimization algorithms can be wrapped.

It provides a registration service for implementors of this interface
via `register()`, `make()`, and `registry`.

Finally, it provides :doc:`third_party_wrappers`.

A more concise introduction is given in the :doc:`/guide`.
"""

from ._interface import (
    AnyOptimizer,
    Bounds,
    Objective,
    Optimizer,
    OptimizeResult,
    Solve,
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
    "Solve",
    "TypeWarning",
    "make",
    "register",
    "registry",
    "spec",
]
