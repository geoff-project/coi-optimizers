..
    SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

Changelog
=========

.. currentmodule:: cernml.optimizers

Unreleased
----------

No changes yet!

v2.0.1
------

- FIX: Relax dependency on `cernml.coi`.

v2.0.0
------

- BREAKING: Drop support for Python 3.7 and 3.8. The new minimum Python version
  is 3.9.
- BREAKING: Update minimum :doc:`NumPy <np:index>` version to 1.22.
- BREAKING: Update minimum :doc:`cernml-extremum-seeking <ces:index>` version
  to 4.0.
- FIX: The documentation for :doc:`SciPy wrappers <api/third_party_wrappers>`
  now documents the meaning of each parameter.

v1.2.0
------

- ADD: compatibility with the new Gymnasium_ package. This is only used when
  calling `cernml.optimizers.make_solve_func()`. It now imports Gymnasium
  lazily and will fall back to the (deprecated) Gym package if it cannot find
  Gymnasium. This in in preparation for the next major release of the COI_.
- WARNING: This is the last release to support Gym and Python 3.7. Starting
  with v2.0, Gymnasium will be required and the minimum Python version will be
  3.9.

.. _Gymnasium: https://gymnasium.farama.org/
.. _COI: https://gitlab.cern.ch/geoff/cernml-coi/

v1.1.0
------

- ADD: Convenience functions `~cernml.optimizers.make_solve_func()` and
  `~cernml.optimizers.solve()`.
- FIX: Various small documentation improvements.

v1.0.2
------

- FIX: string representation of `OptimizerSpec`.
- FIX: registration warnings point at an internal function instead of the line
  that calls `register()`.

v1.0.1
------

- FIX: Mark package as type-annotated according to :pep:`561`.

v1.0.0
------

- Initial release.
