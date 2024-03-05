# SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for the third-party wrappers."""

import inspect
import typing as t
import warnings
from unittest.mock import ANY, Mock, call

import numpy as np
import pytest
from gymnasium.spaces import Box, Sequence
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

from cernml import coi, optimizers


@pytest.fixture
def problem() -> coi.SingleOptimizable:
    class _Problem(coi.SingleOptimizable):
        optimization_space = Box(-1.0, 1.0, shape=[3], dtype=np.double)

        def get_initial_params(self) -> NDArray[np.double]:
            return np.array([0.1, 0.2, 0.0])

        def compute_single_objective(self, params: NDArray[np.double]) -> np.double:
            return np.linalg.norm(params)

    return _Problem()


def test_names() -> None:
    names = list(optimizers.registry.keys())
    assert names == [
        "BOBYQA",
        "COBYLA",
        "ExtremumSeeking",
        "NelderMeadSimplex",
        "Powell",
        "SkoptBayesian",
    ]


def configure_optimizer(optimizer: optimizers.Optimizer) -> optimizers.Optimizer:
    """Ensure that the configs are comparable."""
    if isinstance(optimizer, coi.Configurable):
        config: coi.Config = optimizer.get_config()
        overrides = {
            "verbose": False,
            "max_calls": 20,
            "n_calls": 15,
            "maxfun": 100,
            "check_convergence": True,
        }
        raw_values = {
            k: overrides.get(k, v) for k, v in config.get_field_values().items()
        }
        optimizer.apply_config(config.validate_all(t.cast(dict, raw_values)))
    return optimizer


@pytest.mark.parametrize(
    "name, nfev",
    {
        "BOBYQA": 15,
        "COBYLA": 20,
        "ExtremumSeeking": 20,
        "NelderMeadSimplex": 4,
        "Powell": 43,
        "SkoptBayesian": 15,
    }.items(),
)
def test_run_optimizer(problem: coi.SingleOptimizable, name: str, nfev: int) -> None:
    opt = configure_optimizer(optimizers.make(name))
    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = opt.make_solve_func((space.low, space.high), problem.constraints)
    res = solve(problem.compute_single_objective, problem.get_initial_params())
    assert res.success
    assert res.nfev == nfev


@pytest.mark.parametrize("name", list(optimizers.registry.keys()))
def test_warn_ignored_constraints(name: str) -> None:
    opt = optimizers.make(name)
    bounds = (-np.ones(3), np.ones(3))
    constraint = LinearConstraint(np.diag(np.ones(3)), -1.0, 1.0)
    if name == "COBYLA":
        # COBYLA knows constraints, so it should not raise a warning.
        warnings.simplefilter("error")
        opt.make_solve_func(bounds, [constraint])
    else:
        # None of the other optimizers know constraints, so they should
        # raise a warning.
        with pytest.warns(optimizers.IgnoredArgumentWarning):
            opt.make_solve_func(bounds, [constraint])


def test_bad_bobyqa_bounds(problem: coi.SingleOptimizable) -> None:
    from cernml.optimizers.bobyqa import BobyqaException

    opt = optimizers.make("BOBYQA")
    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = opt.make_solve_func((space.low, space.high), problem.constraints)
    with pytest.raises(BobyqaException):
        res = solve(problem.compute_single_objective, np.zeros(2))
        raise RuntimeError(res.message)


@pytest.mark.parametrize("field", ["gain", "oscillation_size", "decay_rate"])
def test_extremum_seeking_zero_configs(field: str) -> None:
    kwargs = {field: 0.0}
    with pytest.raises(coi.BadConfig, match=field):
        optimizers.make("ExtremumSeeking", **kwargs)


def test_skopt_bad_num_calls() -> None:
    with pytest.raises(
        coi.BadConfig, match="n_initial_points must be less than maxfun"
    ):
        optimizers.make("SkoptBayesian", n_initial_points=10, n_calls=9)


def test_gym_compatibility_shim(problem: coi.SingleOptimizable) -> None:
    import gym  # type: ignore[import-untyped]

    space = gym.spaces.Box(-12.0, 12.0, shape=[3, 2, 1], dtype=np.double)
    problem.optimization_space = t.cast(Box, space)
    opt = optimizers.make("BOBYQA")
    solve = optimizers.make_solve_func(opt, problem)
    variables = inspect.getclosurevars(solve)
    low, high = variables.nonlocals["bounds"]
    assert np.array_equal(low, space.low.flatten())
    assert np.array_equal(high, space.high.flatten())


def test_bad_opt_space_type(problem: coi.SingleOptimizable) -> None:
    space = Sequence(problem.optimization_space)
    problem.optimization_space = t.cast(Box, space)
    opt = optimizers.make("BOBYQA")
    with pytest.raises(TypeError, match="cannot be flattened"):
        optimizers.make_solve_func(opt, problem)


def test_gym_fallback(
    monkeypatch: pytest.MonkeyPatch, problem: coi.SingleOptimizable
) -> None:
    # Given:
    opt = optimizers.make("BOBYQA")
    gym = Mock(name="gym")
    original_import = __import__

    def mock_import(name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Force use of the Gym fallback."""
        if name == "gymnasium":
            raise ModuleNotFoundError(f"No module named {name!r}")
        if name == "gym":
            return gym
        return original_import(name, *args, **kwargs)

    imp = Mock(name="__import__", side_effect=mock_import)

    # When:
    with monkeypatch.context() as m:
        m.setattr("builtins.__import__", imp)
        solve = optimizers.make_solve_func(opt, problem)

    # Then:
    assert imp.call_args_list == [
        call("gymnasium", ANY, None, ("spaces",), 0),
        call("gym", ANY, None, ("spaces",), 0),
    ]
    box = gym.spaces.flatten_space.return_value
    variables = inspect.getclosurevars(solve)
    low, high = variables.nonlocals["bounds"]
    assert low == box.low
    assert high == box.high
