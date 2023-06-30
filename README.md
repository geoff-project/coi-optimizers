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

The [generic optimization framework & front-end][GeOFF] needs a way to start
optimization algorithms that is independent of the specific algorithm's API. It
also needs to configure them in a dynamic manner that cannot necessarily
anticipate the exact parameters supported by each algorithm.

This package helps solve this problem by providing a single interface that can
wrap around almost any single-objective minimization algorithm. The registry
system allows loading algorithms without hard-coding their location. The
[CernML-COI][] provide an API to configure these algorithms dynamically at
runtime.

Quickstart
----------

To write an optimizer, create a new package using [`acc-py init`][] and add
this package to the dependencies:

```python
REQUIREMENTS: dict = {
    'core': [
        'cernml-coi-optimizers ~= 1.0',
        ...
    ],
    ...
}
```

[`acc-py init`]: https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py#GettingstartedwithAccPy-StartinganewPythonproject

Then write a subclass of `Optimizer` and register it:

```python
import typing as t

import cernml.optimizers as opt
import numpy as np

class MyOptimizer(opt.Optimizer):
    def make_solve_func(
        self,
        bounds: opt.Bounds,
        constraints: t.Sequence[opt.Constraint],
    ) -> opt.SolveFunc:
        def solve(obj: opt.Objective, x0: np.ndarray) -> opt.OptimizeResult:
            ...

        return solve

register("MyOptimizer-v1", MyOptimizer)
```

Any [*host application*][GeOFF] may then import your package and instantiate
your optimizer to solve its optimization problem:

```python
import my_project
import cernml.optimizers as opt

def objective(x: np.ndarray) -> float:
    return np.sum(x**4 - x**2)

x0 = np.array([0.0, 0.0])

optimizer = opt.make("MyOptimizer-v1")
solve = optimizer.make_solve_func(bounds=(x0 - 2.0, x0 + 2.0), constraints=[])
results = solve(objective, x0)
```

[GeOFF]: https://gitlab.cern.ch/geoff/geoff-app

Documentation
-------------

Inside the CERN network, you can read the package documentation on the [Acc-Py
documentation server][acc-py-docs]. The API is also documented via extensive
Python docstrings.

[acc-py-docs]: https://acc-py.web.cern.ch/gitlab/geoff/cernml-coi-optimizers/

Changelog
---------

[See here](https://acc-py.web.cern.ch/gitlab/geoff/cernml-coi-optimizers/docs/stable/changelog.html).

Stability
---------

This package uses a variant of [Semantic Versioning](https://semver.org/) that
makes additional promises during the initial development (major version 0):
whenever breaking changes to the public API are published, the first non-zero
version number will increase. This means that code that uses COI version 0.6.0
will continue to work with version 0.6.1, but may break with version 0.7.0.

The exception to this are the contents of `cernml.coi.unstable`, which may
change in any given release.

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
