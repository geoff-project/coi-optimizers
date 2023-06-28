# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Example for how to write an optimizer and register it."""

import typing as t

import click
import numpy as np
from gym.spaces import Box

from cernml.coi import Config, Configurable, ConfigValues, Constraint
from cernml.optimizers import (
    Bounds,
    Objective,
    Optimizer,
    OptimizeResult,
    SolveFunc,
    make,
    register,
)


class RandomSearchOptimizer(Optimizer, Configurable):
    """Example optimizer that simply performs a random search.

    This showcases how to typically implement an optimizer compatible
    with the CernML Common Optimization Interfaces. The following points
    are worth pointing out:

    1. the only job of the `Optimizer` subclass is to configure and
       build the *solve function*. It follows the Builder_ design
       pattern.

    2. All attributes can be configured in three ways:

       a) as a (keyword) argument to `~std:object.__init__()`;
       b) as an attribute of the class;
       c) via the `~cernml.coi.Configurable` API.

       You're encouraged to follow this pattern for maximal flexibility.

    3. Unless prohibitively expensive, you're encouraged to round-trip
       all settings through `get_config()` and `apply_config()` to test
       their consistency. This class does so in the
       `~std:object.__init__()` method.

    4. The solve function *can* but *need not* be a nested function
       inside `make_solve_func()`. It is nested in this example.

    5. Beware that all constraints and *objective* are allowed to raise
       exceptions. Your solve function should be well-behaved in this
       case and not swallow exceptions. In particular, the COI expect to
       be able to abort an optimization procedure by raising
       `cernml.coi.cancellation.CancelledError`.

    6. Taking the *constraints* into account is encouraged but optional.
       If your algorithm cannot handle constraints, it is acceptable to
       print a warning and completely ignore them.

    .. _Builder: https://en.wikipedia.org/wiki/Builder_pattern
    """

    def __init__(self, maxfun: int = 1000) -> None:
        self.maxfun = maxfun
        # Self-consistency check:
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    def get_config(self) -> Config:
        return Config().add("maxfun", self.maxfun, range=(0, 1_000_000))

    def apply_config(self, values: ConfigValues) -> None:
        self.maxfun = values.maxfun

    def make_solve_func(
        self,
        bounds: Bounds,
        constraints: t.Sequence[Constraint],
    ) -> SolveFunc:
        space = Box(*bounds, dtype=np.common_type(*bounds))

        def is_valid(params: np.ndarray) -> bool:
            return all(c(params) >= 0 for c in constraints)

        def solve_func(objective: Objective, initial: np.ndarray) -> OptimizeResult:
            best_x = initial
            best_o = objective(initial)
            for _ in range(1, self.maxfun):
                next_x = space.sample()
                valid = is_valid(next_x)
                next_o = objective(next_x)
                if valid and not next_o >= best_o:
                    best_x, best_o = next_x, next_o
            return OptimizeResult(
                x=best_x,
                fun=best_o,
                success=True,
                message="",
                nit=self.maxfun,
                nfev=self.maxfun,
            )

        return solve_func


# Typically, you would not call `register()` directly, but instead
# declare an entry point for the group `cernml.optimizers` in your
# setup.py, setup.cfg or pyproject.toml file.
register("RandomSearch", RandomSearchOptimizer)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-n", "--nfev", default=1000, help="number of iterations")
def main(nfev: int = 1000) -> None:
    """Run a custom random-search optimization algorithm."""
    opt = make("RandomSearch", maxfun=nfev)
    solve = opt.make_solve_func(Bounds(-np.ones(3), np.ones(3)), constraints=[])
    res = solve(np.linalg.norm, np.random.uniform(-1, 1, size=3))
    print(
        f"Best result after {res.nit} iterations of {opt.spec.name}:",
        f"{res.fun:.3g} at x={res.x}",
    )


if __name__ == "__main__":
    main()
