# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for the abstract base class."""

import typing as t
from unittest import mock

import pytest
from gymnasium.spaces import Box

from cernml.coi import Constraint, FunctionOptimizable, SingleOptimizable
from cernml.optimizers import Bounds, Objective, Optimizer, Solve, solve


def test_abstract_method_raises() -> None:
    class Impl(Optimizer):
        # pylint: disable = useless-parent-delegation
        def make_solve_func(
            self, bounds: Bounds, constraints: t.Sequence[Constraint]
        ) -> Solve:
            return super().make_solve_func(  # type: ignore[safe-super]
                bounds, constraints
            )

    with pytest.raises(NotImplementedError):
        Impl().make_solve_func(*t.cast(list[t.Any], [None, None]))


def test_solve_single_optimizable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given:
    def _solve_side_effect(objective: Objective, params: t.Any) -> t.Any:
        objective(params)
        return mock.DEFAULT

    flatten_space = mock.Mock(name="flatten-space")
    flattened = flatten_space.return_value
    flattened.mock_add_spec(Box(-1, 1, [1]))
    monkeypatch.setattr("gymnasium.spaces.flatten_space", flatten_space)
    problem = mock.Mock(SingleOptimizable)
    optimizer = mock.Mock(Optimizer)
    solvefunc = optimizer.make_solve_func.return_value
    solvefunc.side_effect = _solve_side_effect
    # When:
    result = solve(optimizer, problem)
    # Then:
    flatten_space.assert_called_once_with(problem.optimization_space)
    optimizer.make_solve_func.assert_called_once_with(
        (flattened.low, flattened.high), problem.constraints
    )
    problem.get_initial_params.assert_called_once_with()
    initial_params = problem.get_initial_params.return_value
    solvefunc.assert_called_once_with(problem.compute_single_objective, initial_params)
    problem.compute_single_objective.assert_called_once_with(initial_params)
    assert solvefunc.return_value == result


def test_solve_function_optimizable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given:
    def _solve_side_effect(objective: Objective, params: t.Any) -> t.Any:
        objective(params)
        return mock.DEFAULT

    flatten_space = mock.Mock(name="flatten-space")
    flattened = flatten_space.return_value
    flattened.mock_add_spec(Box(-1, 1, [1]))
    monkeypatch.setattr("gymnasium.spaces.flatten_space", flatten_space)
    problem = mock.Mock(FunctionOptimizable)
    optimizer = mock.Mock(Optimizer)
    solvefunc = optimizer.make_solve_func.return_value
    solvefunc.side_effect = _solve_side_effect
    cycle_time = mock.Mock(float, name="cycle_time")
    # When:
    result = solve(optimizer, problem, cycle_time)
    # Then:
    problem.get_optimization_space.assert_called_once_with(cycle_time)
    space = problem.get_optimization_space.return_value
    flatten_space.assert_called_once_with(space)
    optimizer.make_solve_func.assert_called_once_with(
        (flattened.low, flattened.high), problem.constraints
    )
    problem.get_initial_params.assert_called_once_with(cycle_time)
    initial_params = problem.get_initial_params.return_value
    solvefunc.assert_called_once_with(mock.ANY, initial_params)
    problem.compute_function_objective.assert_called_once_with(
        cycle_time, initial_params
    )
    assert solvefunc.return_value == result


def test_make_solve_func_type_errors() -> None:
    with pytest.raises(TypeError, match="not a SingleOptimizable"):
        solve(mock.Mock(), mock.Mock())
    with pytest.raises(TypeError, match="not a FunctionOptimizable"):
        solve(mock.Mock(), mock.Mock(), mock.Mock())
