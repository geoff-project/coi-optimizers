.. SPDX-FileCopyrightText: 2023-2026 CERN
.. SPDX-FileCopyrightText: 2023-2026 GSI Helmholtzzentrum für Schwerionenforschung
.. SPDX-FileNotice: All rights not expressly granted are reserved.
..
.. SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

:tocdepth: 3

Changelog
=========

.. currentmodule:: cernml.optimizers

Unreleased
----------

No changes yet!

v4.x
----

v4.0.4
^^^^^^

Bug fixes
~~~~~~~~~
- Fix broadcasting of constraints into the optimization space for Xopt.

Other changes
~~~~~~~~~~~~~
- Harmonize docs theme of Geoff packages.

v4.0.3
^^^^^^

Other changes
~~~~~~~~~~~~~
- Fix PyPI classifiers.
- The package is now released on PyPI.

v4.0.2
^^^^^^

Other changes
~~~~~~~~~~~~~
- Update project links to point at the new website https://geoff.docs.cern.ch/.

v4.0.1
^^^^^^

Bug fixes
~~~~~~~~~
- Broken docs on Sphinx 9 and due to changed Xopt API.
- Tests broken by Xopt 2.7.
- Incompatibility between Xopt 2.7 and Botorch 0.18.

v4.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- Add to `.Bobyqa` a new parameter *scaling_within_bounds* that is true by default. This changes the behavior of the optimizer!
- Reduce the default values of `.Bobyqa`\ 's  parameters *rhobeg* (to 0.25) and *rhoend* (to 0.025). This is motivated by the new default option *scaling_within_bounds*, which scales parameters into the interval [0; 1] instead of the conventional [−1; +1].

Additions
~~~~~~~~~
- :ref:`Wrappers <Xopt wrappers>` around the Bayesian and RCDS optimizers of `Xopt <https://xopt.xopt.org/>`__.

Bug fixes
~~~~~~~~~
- :ref:`SciPy wrappers` erroneously returned a `numpy.bool` instead of a `bool` in `OptimizeResult.success`.
- Tests failed on Python 3.9 and any modern :doc:`importlib-metadata <imp:index>`.

v3.x
----

v3.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- Drop support for :doc:`cernml-coi <coi:index>` v0.8. The biggest change is that we now use the new :doc:`coi:api/typeguards` introduced with v0.9.

Bug fixes
~~~~~~~~~
- Relax dependency on `numpy`.

v2.x
----

v2.0.1
^^^^^^

Bug fixes
~~~~~~~~~
- Relax dependency on the COI_.

v2.0.0
^^^^^^

Breaking changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.7 and 3.8. The new minimum Python version is 3.9.
- Update minimum :doc:`NumPy <np:index>` version to 1.22.
- Update minimum :doc:`cernml-extremum-seeking <ces:index>` version to 4.0.

Bug fixes
~~~~~~~~~
- The documentation for :ref:`SciPy wrappers` now documents the meaning of each parameter.

v1.x
----

v1.2.0
^^^^^^

.. warning::
    This is the last release to support Gym and Python 3.7. Starting with v2.0,
    Gymnasium will be required and the minimum Python version will be 3.9.

Additions
~~~~~~~~~
- compatibility with the new Gymnasium_ package. This is only used when calling `cernml.optimizers.make_solve_func()`. It now imports Gymnasium lazily and will fall back to the (deprecated) Gym package if it cannot find Gymnasium. This in in preparation for the next major release of the COI_.

.. _Gymnasium: https://gymnasium.farama.org/
.. _COI: https://gitlab.cern.ch/geoff/cernml-coi/

v1.1.0
^^^^^^

Additions
~~~~~~~~~
- Convenience functions `~cernml.optimizers.make_solve_func()` and `~cernml.optimizers.solve()`.

Bug fixes
~~~~~~~~~
- Various small documentation improvements.

v1.0.2
^^^^^^

Bug fixes
~~~~~~~~~
- string representation of `OptimizerSpec`.
- registration warnings point at an internal function instead of the line that calls `register()`.

v1.0.1
^^^^^^

Bug fixes
~~~~~~~~~
- Mark package as type-annotated according to :pep:`561`.

v1.0.0
^^^^^^

- Initial release.
