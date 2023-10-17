..
    SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
    SPDX-FileNotice: All rights not expressly granted are reserved.

    SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

User Guide
==========

.. currentmodule:: cernml.optimizers

This package defines `Optimizer`, a generic interface with which a
multitude of single-objective optimization algorithms can be wrapped.

It further provides wrappers for a number of third-party packages:

- :doc:`SciPy <scipy:index>`,
- :doc:`Scikit-Optimize <skopt:user_guide>`,
- :doc:`Py-BOBYQA <bobyqa:index>`,
- :doc:`CernML Extremum Seeking <ces:index>`,

Installation
------------

To install this package from the `Acc-Py Repository`_ together with all
third-party packages that are wrapped, run the following line while on the CERN
network:

.. _Acc-Py Repository:
   https://wikis.cern.ch/display/ACCPY/Getting+started+with+Acc-Py

.. code-block:: shell-session

    $ pip install cernml-coi-optimizers[all]

You can also install only individual third-party packages by specifying the
respecting *extra*. If no extra is given, no additional packages are installed
and you can only use those wrappers whose packages are already installed.

.. code-block:: shell-session

    $ pip install cernml-coi-optimizers[bobyqa]
    $ pip install cernml-coi-optimizers[cernml-es]
    $ pip install cernml-coi-optimizers[scipy]
    $ pip install cernml-coi-optimizers[skopt]
    $ pip install cernml-coi-optimizers

To use the source repository, you must first install it as well:

.. code-block:: shell-session

    $ git clone https://gitlab.cern.ch/geoff/cernml-coi-optimizers.git
    $ cd ./cernml-coi-optimizers/
    $ pip install .[all]

Using an Optimizer
------------------

The built-in wrappers are documented on the page
:doc:`/api/third_party_wrappers`. The easiest way to use them is through
:doc:`/api/registration` provided by this package:

    >>> from cernml.optimizers import make
    >>> make("BOBYQA")  # doctest: +ELLIPSIS
    <cernml.optimizers.bobyqa.Bobyqa object at ...>

But you can also instantiate it directly:

    >>> from cernml.optimizers.bobyqa import Bobyqa
    >>> Bobyqa()  # doctest: +ELLIPSIS
    <cernml.optimizers.bobyqa.Bobyqa object at ...>

To optimize an optimization problem like the following:

    >>> import numpy as np
    >>> def objective(x):
    ...     return np.sum(x**4 - x**2)
    >>> x0 = np.zeros(2)

you create a *solve function* and call it:

    >>> opt = make("BOBYQA")
    >>> solve = opt.make_solve_func(bounds=(x0-2, x0+2), constraints=[])
    >>> result = solve(objective, x0)

The resulting `OptimizeResult` contains various information about the
optimization procedure:

    >>> assert result.success
    >>> print(result.message)
    Success: rho has reached rhoend
    >>> result.x
    array([0.71053567, 0.7209604 ])
    >>> round(result.fun, 3)
    -0.5

Integration with the COI
------------------------

This package depends on and integrates with the :doc:`Common Optimization
Interfaces <coi:index>`.
If you have an object that subclasses `~cernml.coi.SingleOptimizable` or
`~cernml.coi.FunctionOptimizable` …

    >>> from cernml.coi import SingleOptimizable
    >>> from gym.spaces import Box
    ...
    >>> class ExampleProblem(SingleOptimizable):
    ...     optimization_space = Box(x0-2, x0+2)
    ...     def get_initial_params(self):
    ...         return x0
    ...     def compute_single_objective(self, params):
    ...         return objective(params)

… you can use the convenience functions
`cernml.optimizers.make_solve_func()` and `cernml.optimizers.solve()` and they
will extract the necessary information automatically:

    >>> # This overwrites the previous `solve`.
    ... from cernml.optimizers import solve
    ...
    >>> problem = ExampleProblem()
    >>> result = solve(opt, problem)
    >>> round(result.fun, 3)
    -0.5

Configuring an Optimizer
------------------------

Every optimizer can be configured in three equivalent ways: via initializer
arguments, via its attributes and via the `~cernml.coi.Configurable` API:

.. tab:: initializer arguments

   >>> opt = make("BOBYQA", maxfun=100, objfun_has_noise=True)

.. tab:: attributes

   >>> opt = make("BOBYQA")
   >>> opt.maxfun = 100
   >>> opt.objfun_has_noise = True

.. tab:: Configurable

   >>> opt = make("BOBYQA")
   >>> config = opt.get_config()
   >>> raw_values = config.get_field_values()
   >>> raw_values["maxfun"] = "100"
   >>> raw_values["objfun_has_noise"] = "True"
   >>> values = config.validate_all(raw_values)
   >>> opt.apply_config(values)

Registering Your Own Optimizer
------------------------------

There are two different ways to register your own `Optimizer` subclass so that
it is picked up by `make()`: you can do it statically via `entry points`_ or
dynamically via the `register()` function.

Entry points have the advantage that your optimizer will always be picked up as
long as your package is installed. The user need not import anything. To use
them, simply add an entry to your project metadata (typically in
:file:`pyproject.toml`, :file:`setup.cfg` or :file:`setup.py`) with the `object
reference`_ pointing at your subclass:

.. _entry points:
   https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins
.. _object reference:
   https://packaging.python.org/en/latest/specifications/entry-points/#data-model

.. tab:: pyproject.toml

    .. code-block:: toml

        [project.entry-points."cernml.optimizers"]
        MyOpt-v1 = "my_package.my_optimizer:MyOptimizer"

.. tab:: setup.cfg

    .. code-block:: ini

        [options.entry_points]
        cernml.optimizers =
            MyOpt-v1 = my_package.my_optimizer:MyOptimizer

.. tab:: setup.py

    .. code-block:: python

        setup(
            "my-distribution",
            # ...,
            entry_points={
                "cernml.optimizers": [
                    "MyOpt-v1 = my_package.my_optimizer:MyOptimizer",
                ],
            },
        )

The object reference has the format :samp:`{module}:{class}` or
:samp:`{package}.{module}:class` or even :samp:`{module}:{nested}.{class}`. The
entry point name (:samp:`MyOpt-v1` in the example above) should be unique among
all installed optimizers.

If entry points don't work for you, you can also register your optimizers
dynamically via `register()`. Typically you call this function in the global
scope of your module so that it is called upon import.

You can point the registry at your subclass either by passing a string `object
reference`_ (same syntax as for entry points) or by passing the class object
directly:

.. tab:: string reference

    .. code-block:: python

        # my_package/my_optimizer.py
        from cernml.optimizers import Optimizer

        class MyOptimizer(Optimizer):
            ...

    .. code-block:: python

        # my_package/__init__.py

        from cernml.optimizers import register

        register("MyOpt-v1", "my_package.my_optimizer:MyOptimizer")

.. tab:: class object

    .. code-block:: python

        # my_package/my_optimizer.py
        from cernml.optimizers import Optimizer, register

        class MyOptimizer(Optimizer):
            ...

        register("MyOpt-v1", MyOptimizer)

    .. code-block:: python

        # my_package/__init__.py
        from .my_optimizer import MyOptimizer

If you pass a string, the optimizer module is only imported when the user calls
`make()`. You still need to import the module that calls `register()`, however.
If you pass the optimizer class directly, it must be available at registration
time. No deferred import is possible in this case.

Note that statically and dynamically registered optimizers all share one
namespace. So again, the name that you pass to `register()` (:samp:`MyOpt-v1`
in this example) must not conflict with any previous name, or with any entry
point name in the ``cernml.optimizers`` group.
