.. SPDX-FileCopyrightText: 2023 - 2025 CERN
.. SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
.. SPDX-FileNotice: All rights not expressly granted are reserved.
..
.. SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

:tocdepth: 3

Wrappers for Third-Party Packages
=================================

SciPy
-----

.. automodule:: cernml.optimizers.scipy

   .. autoclass:: Cobyla
   .. autoclass:: NelderMeadSimplex
   .. autoclass:: Powell

Scikit-Optimize
---------------

.. automodule:: cernml.optimizers.skopt

   .. autoclass:: SkoptBayesian

Py-BOBYQA
---------

.. automodule:: cernml.optimizers.bobyqa

   .. autoclass:: Bobyqa
   .. autoexception:: BobyqaException

CernML Extremum Seeking
-----------------------

.. automodule:: cernml.optimizers.extremum_seeking

   .. autoclass:: ExtremumSeeking

Xopt
----

.. automodule:: cernml.optimizers.xopt

   .. autoclass:: XoptRcds
   .. autoclass:: XoptBayesian
   .. autoclass:: BaseXoptOptimizer
      :members: make_generator

Xopt configuration helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^

   The following enum classes serve to further configure `XoptBayesian`:

   .. autoclass:: BayesianMethod
      :members:
      :exclude-members: get_generator
   .. autoclass:: TurboController
      :members:
      :exclude-members: get_controller
