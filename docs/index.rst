..
    SPDX-FileCopyrightText: 2020-2023 CERN
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Optimization Loops for the Common Optimization Interfaces
=========================================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

`CernML-COI`_ defines common interfaces that facilitate using numerical
optimization and reinforcement learning (RL) on the same optimization problems.
This makes it possible to unify both approaches into a generic optimization
application in the CERN Control Center.

CernML-COI-Optimizers defines an abstract optimizer interface for all numerical
optimization algorithms. It also a registration mechanism to find all
available algorithms. Finally, it comes with wrappers for the following
third-party packages:

- `Scipy`_,
- `Py-BOBYQA`_,
- `CernML-Extremum-Seeking`_,
- Bayesian optimization with `Scikit-Optimize`_.

The optimizers are suitable to be hooked into the pre-implemented optimization
loop defined in `CernML-COI-Loops`_.

This repository can be found online on CERN's `Gitlab`_.

.. _Gitlab: https://gitlab.cern.ch/geoff/cernml-coi-optimizers/
.. _CernML-COI: https://gitlab.cern.ch/geoff/cernml-coi/
.. _CernML-COI-Loops: https://gitlab.cern.ch/geoff/cernml-coi-loops/
.. _CernML-Extremum-Seeking: https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking
.. _Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _Py-BOBYQA: https://numericalalgorithmsgroup.github.io/pybobyqa/
.. _Scikit-Optimize: https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html

.. toctree::
   :maxdepth: 2

   api/index
   changelog
