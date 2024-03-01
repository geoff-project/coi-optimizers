# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Defines the `Optimizer` interface."""

from __future__ import annotations

import typing as t
from functools import partial

from gym.spaces import Box, flatten_space

from cernml.coi import FunctionOptimizable, SingleOptimizable

from ._interface import Optimizer, OptimizeResult, Solve

__all__ = [
    "make_solve_func",
    "solve",
]


@t.overload
def make_solve_func(optimizer: Optimizer, problem: SingleOptimizable) -> Solve: ...


@t.overload
def make_solve_func(
    optimizer: Optimizer, problem: FunctionOptimizable, cycle_time: float
) -> Solve: ...


def make_solve_func(
    optimizer: Optimizer,
    problem: t.Union[SingleOptimizable, FunctionOptimizable],
    cycle_time: t.Optional[float] = None,
) -> Solve:
    """Create a new solve function.

    This is a convenience wrapper around `Optimizer.make_solve_func()`.
    It extracts the required arguments from the given *problem* and
    passes them on.

    Args:
        optimizer: The optimization algorithm to use. Generally
            instantiated via `make()`.
        problem: The optimization problem on which to run.
        cycle_time: Only passed if *problem* is
            a `~cernml.coi.FunctionOptimizable`. The *cycle time* in
            milliseconds at which the underlying function should be
            optimized.

    Returns:
        The `Solve` function.
    """
    if cycle_time is None:
        if not isinstance(problem, SingleOptimizable):
            raise TypeError(f"not a SingleOptimizable: {problem!r}")
        space: Box = flatten_space(problem.optimization_space)
        bounds = (space.low, space.high)
        constraints = problem.constraints
        return optimizer.make_solve_func(bounds, constraints)
    if not isinstance(problem, FunctionOptimizable):
        raise TypeError(
            f"passed a cycle time, but not a FunctionOptimizable: {problem!r}"
        )
    space = flatten_space(problem.get_optimization_space(cycle_time))
    bounds = (space.low, space.high)
    constraints = problem.constraints
    return optimizer.make_solve_func(bounds, constraints)


@t.overload
def solve(optimizer: Optimizer, problem: SingleOptimizable) -> OptimizeResult: ...


@t.overload
def solve(
    optimizer: Optimizer, problem: FunctionOptimizable, cycle_time: float
) -> OptimizeResult: ...


def solve(
    optimizer: Optimizer,
    problem: t.Union[SingleOptimizable, FunctionOptimizable],
    cycle_time: t.Optional[float] = None,
) -> OptimizeResult:
    """Solve an optimization problem with the given optimizer.

    This is a convenience function that creates a `solve function
    <cernml.optimizers.Solve>` and immediately calls it. All necessary
    information is extracted from the given *optimizer* and *problem*.

    Args:
        optimizer: The optimization algorithm to use. Generally
            instantiated via `make()`.
        problem: The optimization problem on which to run.
        cycle_time: Only passed if *problem* is
            a `~cernml.coi.FunctionOptimizable`. The *cycle time* in
            milliseconds at which the underlying function should be
            optimized.

    Returns:
        The `OptimizeResult` returned by the `Solve` function.
    """
    if cycle_time is None:
        # If this cast is wrong, `make_solve_func()` will raise
        # TypeError.
        problem = t.cast(SingleOptimizable, problem)
        solvefunc = make_solve_func(optimizer, problem)
        objective = problem.compute_single_objective
        initial_params = problem.get_initial_params()
        return solvefunc(objective, initial_params)
    # If this cast is wrong, `make_solve_func()` will raise
    # TypeError.
    problem = t.cast(FunctionOptimizable, problem)
    solvefunc = make_solve_func(optimizer, problem, cycle_time)
    objective = partial(problem.compute_function_objective, cycle_time)
    initial_params = problem.get_initial_params(cycle_time)
    return solvefunc(objective, initial_params)
