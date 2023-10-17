..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Optimizer Wrappers for the Common Optimization Interfaces
=========================================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

:doc:`CernML-COI <coi:index>` defines common interfaces that facilitate using
numerical optimization and reinforcement learning (RL) on the same optimization
problems. This makes it possible to unify both approaches into a generic
optimization application in the CERN Control Center.

CernML-COI-Optimizers defines an abstract optimizer interface for all numerical
optimization algorithms. It also a registration mechanism to find all
available algorithms. Finally, it comes with wrappers for the following
third-party packages:

- :doc:`SciPy <scipy:index>`,
- :doc:`Scikit-Optimize <skopt:user_guide>`,
- :doc:`Py-BOBYQA <bobyqa:index>`,
- :doc:`CernML Extremum Seeking <ces:index>`,

The optimizers are suitable to be hooked into the pre-implemented optimization
loop defined in CernML-COI-Loops_. However, you can also run their optimization
loops directly:

.. code-block:: python

    # Run `pip install cern_awake_env` for this particular example.
    import cern_awake_env
    from cernml import coi, optimizers

    env = coi.make("AwakeSimEnvH-v1")
    opt = optimizers.make("BOBYQA")

    result = optimizers.solve(opt, env)

This repository can be found online on CERN's Gitlab_.

.. _CernML-COI-Loops: https://gitlab.cern.ch/geoff/cernml-coi-loops/
.. _Gitlab: https://gitlab.cern.ch/geoff/cernml-coi-optimizers/

.. toctree::
   :maxdepth: 2

   guide
   examples/index
   api/index
   changelog
