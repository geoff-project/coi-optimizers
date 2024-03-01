..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

The Registry API
================

.. currentmodule:: cernml.optimizers

.. autofunction:: make

.. autofunction:: spec

.. autofunction:: register

.. data:: registry
    :type: dict[str, OptimizerSpec]
    :value: {…}

    The global registry of all optimizers.

    Upon import, this is filled with entries for all optimizers found via the
    entry point API.

    You should not modify this dict yourself. Call `register()` to
    dynamically add further optimizers.

    Example:

        >>> # Get a list of the names all registered optimizers:
        >>> from cernml.optimizers import registry
        >>> list(registry.keys())
        ['BOBYQA', 'COBYLA', ...]

.. data:: EP_GROUP
   :type: str
   :value: 'cernml.optimizers'

   The name of the entry point group under which the registry searches for
   optimizers.

.. autoclass:: OptimizerSpec
   :members: load, make, name, value, dist

.. autoclass:: OptimizerWithSpec

.. autoexception:: OptimizerNotFound

.. autoexception:: DuplicateOptimizerWarning

.. autoexception:: TypeWarning
