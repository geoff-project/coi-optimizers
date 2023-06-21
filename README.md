<!--
SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
SPDX-FileNotice: All rights not expressly granted are reserved.

SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+
-->

Optimization Loops for the Common Optimization Interfaces
=========================================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

[CernML-COI][] defines common interfaces that facilitate using numerical
optimization and reinforcement learning (RL) on the same optimization problems.
This makes it possible to unify both approaches into a generic optimization
application in the CERN Control Center.

CernML-COI-Optimizers defines an abstract optimizer interface for all numerical
optimization algorithms. It also a registration mechanism to find all
available algorithms. Finally, it comes with wrappers for the following
third-party packages:

- [Scipy][],
- [Py-BOBYQA][],
- [CernML-Extremum-Seeking][],
- Bayesian optimization with [Scikit-Optimize][].

The optimizers are suitable to be hooked into the pre-implemented optimization
loop defined in [CernML-COI-Loops][].

This repository can be found online on CERN's [Gitlab][].

[Gitlab]: https://gitlab.cern.ch/geoff/cernml-coi-optimizers/
[CernML-COI]: https://gitlab.cern.ch/geoff/cernml-coi/
[CernML-COI-Loops]: https://gitlab.cern.ch/geoff/cernml-coi-loops/
[CernML-Extremum-Seeking]: https://gitlab.cern.ch/geoff/optimizers/cernml-extremum-seeking
[Scipy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[Py-BOBYQA]: https://numericalalgorithmsgroup.github.io/pybobyqa/
[Scikit-Optimize]: https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html

Table of Contents
-----------------

[[_TOC_]]

Motivation
----------

TODO

Quickstart
----------

TODO

Stability
---------

This package uses a variant of [Semantic Versioning](https://semver.org/) that
makes additional promises during the initial development (major version 0):
whenever breaking changes to the public API are published, the first non-zero
version number will increase. This means that code that uses COI version 0.6.0
will continue to work with version 0.6.1, but may break with version 0.7.0.

The exception to this are the contents of `cernml.coi.unstable`, which may
change in any given release.

Changelog
---------

TODO

Documentation
-------------

TODO

License
-------

Except as otherwise noted, this work is licensed under either of [GNU Public
License, Version 3.0 or later](LICENSES/GPL-3.0-or-later.txt), or [European
Union Public License, Version 1.2 or later](LICENSES/EUPL-1.2.txt), at your
option. See [COPYING](COPYING) for details.

Unless You explicitly state otherwise, any contribution intentionally submitted
by You for inclusion in this Work (the Covered Work) shall be dual-licensed as
above, without any additional terms or conditions.

For full authorship information, see the version control history.
