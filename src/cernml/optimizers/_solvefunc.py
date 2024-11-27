# SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Defines the `Optimizer` interface."""

from __future__ import annotations

import typing as t
from functools import partial

from cernml.coi import FunctionOptimizable, SingleOptimizable

from ._interface import Optimizer, OptimizeResult, Solve

if t.TYPE_CHECKING:
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces  # type: ignore[no-redef]


__all__ = (
    "make_solve_func",
    "solve",
)


@t.overload
def make_solve_func(optimizer: Optimizer, problem: SingleOptimizable) -> Solve: ...


@t.overload
def make_solve_func(
    optimizer: Optimizer, problem: FunctionOptimizable, cycle_time: float
) -> Solve: ...


def make_solve_func(
    optimizer: Optimizer,
    problem: SingleOptimizable | FunctionOptimizable,
    cycle_time: float | None = None,
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
        space = _flatten_space(problem.optimization_space)
        bounds = (space.low, space.high)
        constraints = problem.constraints
        return optimizer.make_solve_func(bounds, constraints)
    if not isinstance(problem, FunctionOptimizable):
        raise TypeError(
            f"passed a cycle time, but not a FunctionOptimizable: {problem!r}"
        )
    space = _flatten_space(problem.get_optimization_space(cycle_time))
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
    problem: SingleOptimizable | FunctionOptimizable,
    cycle_time: float | None = None,
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


def _flatten_space(space: spaces.Space) -> spaces.Box:
    """Wrapper around `gymnasium.spaces.flatten_space()`.

    This is a compatibility shim that detects spaces from the deprecated
    Gym package by name. (Their `__module__` must be ``gym.spaces.*``
    and their `__name__` must be title-case.) If such a space is
    detected, the matching overload of
    `gymnasium.spaces.flatten_space()` is looked up by name. Without
    this test, Gym spaces would always be dispatched to the default
    overload.

    For compatibility with the new Gymnasium package, we also detect if
    the result of `flatten_space()` is not a `Box` and raise
    a `TypeError` in such a case.
    """
    # pylint: disable = import-outside-toplevel
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces  # type: ignore[no-redef]

    if not getattr(space, "is_np_flattenable", True):
        raise TypeError(f"space cannot be flattened: {space!r}")
    space_type = type(space)
    # pylint: disable = unnecessary-dunder-call
    module_name: t.Any = vars(type)["__module__"].__get__(space_type)
    space_name: str = vars(type)["__name__"].__get__(space_type)
    if str.startswith(module_name, "gym.spaces.") and str.istitle(space_name):
        # Compatibility shim for old Gym package.
        gymnasium_class: type[spaces.Space] = getattr(spaces, space_name)
        flatten_space = spaces.flatten_space.dispatch(gymnasium_class)
    else:
        flatten_space = spaces.flatten_space
    return t.cast(spaces.Box, flatten_space(space))
