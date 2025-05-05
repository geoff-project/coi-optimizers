# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for the third-party wrappers."""

from __future__ import annotations

import contextlib
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

ALL_OPTIMIZERS: t.Final = tuple(optimizers.registry.keys())


@pytest.fixture
def nfev(request: pytest.FixtureRequest) -> int:
    """The maximum number of allowed iterations per optimizer.

    This reads the pytest marker `optimizer`.
    """
    mark: pytest.Mark | None = request.node.get_closest_marker("optimizer")
    if mark is None:
        request.raiseerror("missing marker: optimizer")
    name: str
    [name] = mark.args
    try:
        return {
            "BOBYQA": 15,
            "COBYLA": 20,
            "ExtremumSeeking": 20,
            "NelderMeadSimplex": 4,
            "Powell": 43,
            "SkoptBayesian": 15,
        }[name]
    except KeyError:
        request.raiseerror(f"invalid value for marker 'optimizer': {name!r}")


@pytest.fixture
def optimizer(request: pytest.FixtureRequest) -> optimizers.Optimizer:
    mark: pytest.Mark | None = request.node.get_closest_marker("optimizer")
    if mark is None:
        request.raiseerror("missing marker: optimizer")
    name: str
    [name] = mark.args
    opt = optimizers.make(name)
    if coi.is_configurable(opt):
        config = opt.get_config()
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
        opt.apply_config(config.validate_all(t.cast(dict, raw_values)))
    return opt


@pytest.fixture
def problem(nfev: int) -> coi.SingleOptimizable:
    class _Problem(coi.SingleOptimizable):
        optimization_space = Box(-1.0, 1.0, shape=[3], dtype=np.double)
        num_calls = 0

        def get_initial_params(
            self, *, seed: int | None = None, options: dict[str, t.Any] | None = None
        ) -> NDArray[np.double]:
            super().get_initial_params(seed=seed, options=options)
            return np.array([0.1, 0.2, 0.0])

        def compute_single_objective(self, params: NDArray[np.double]) -> np.double:
            self.num_calls += 1
            assert self.num_calls <= nfev, "too many calls for this optimizer"
            return np.linalg.norm(params)

    return _Problem()


def test_builtin_names() -> None:
    assert ALL_OPTIMIZERS == (
        "BOBYQA",
        "COBYLA",
        "ExtremumSeeking",
        "NelderMeadSimplex",
        "Powell",
        "SkoptBayesian",
    )


@pytest.mark.parametrize(
    (),
    [pytest.param(marks=pytest.mark.optimizer(name)) for name in ALL_OPTIMIZERS],
    ids=ALL_OPTIMIZERS,
)
def test_builtins_are_configurable(optimizer: optimizers.Optimizer) -> None:
    assert coi.is_configurable(optimizer)


@pytest.mark.parametrize(
    (),
    [pytest.param(marks=pytest.mark.optimizer(name)) for name in ALL_OPTIMIZERS],
    ids=ALL_OPTIMIZERS,
)
def test_run_optimizer(
    problem: coi.SingleOptimizable, optimizer: optimizers.Optimizer, nfev: int
) -> None:
    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = optimizer.make_solve_func((space.low, space.high), problem.constraints)
    res = solve(problem.compute_single_objective, problem.get_initial_params())
    assert res.success
    assert res.nfev == nfev


@pytest.mark.parametrize(
    (),
    [pytest.param(marks=pytest.mark.optimizer(name)) for name in ALL_OPTIMIZERS],
    ids=ALL_OPTIMIZERS,
)
def test_warn_ignored_constraints(optimizer: optimizers.Optimizer) -> None:
    bounds = (-np.ones(3), np.ones(3))
    constraint = LinearConstraint(np.diag(np.ones(3)), -1.0, 1.0)
    context = contextlib.ExitStack()
    assert optimizer.spec is not None
    if optimizer.spec.name == "COBYLA":
        # COBYLA knows constraints, so it should not raise a warning.
        context.enter_context(warnings.catch_warnings())
        warnings.simplefilter("error")
    else:
        # None of the other optimizers know constraints, so they should
        # raise a warning.
        context.enter_context(pytest.warns(optimizers.IgnoredArgumentWarning))
    with context:
        optimizer.make_solve_func(bounds, [constraint])


@pytest.mark.optimizer("BOBYQA")
def test_bad_bobyqa_bounds(
    problem: coi.SingleOptimizable, optimizer: optimizers.Optimizer
) -> None:
    from cernml.optimizers.bobyqa import BobyqaException

    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = optimizer.make_solve_func((space.low, space.high), problem.constraints)
    with pytest.raises(BobyqaException):
        solve(problem.compute_single_objective, np.zeros(2))


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


@pytest.mark.optimizer("BOBYQA")
def test_gym_compatibility_shim(
    problem: coi.SingleOptimizable, optimizer: optimizers.Optimizer
) -> None:
    try:
        import gym  # type: ignore[import-untyped]
    except AttributeError:
        pytest.skip("gym incompatible with installed version of importlib_metadata")

    space = gym.spaces.Box(-12.0, 12.0, shape=[3, 2, 1], dtype=np.double)
    problem.optimization_space = t.cast(Box, space)
    solve = optimizers.make_solve_func(optimizer, problem)
    variables = inspect.getclosurevars(solve)
    low, high = variables.nonlocals["bounds"]
    assert np.array_equal(low, space.low.flatten())
    assert np.array_equal(high, space.high.flatten())


@pytest.mark.optimizer("BOBYQA")
def test_bad_opt_space_type(
    problem: coi.SingleOptimizable, optimizer: optimizers.Optimizer
) -> None:
    space = Sequence(problem.optimization_space)
    problem.optimization_space = t.cast(Box, space)
    with pytest.raises(TypeError, match="cannot be flattened"):
        optimizers.make_solve_func(optimizer, problem)


@pytest.mark.optimizer("BOBYQA")
def test_gym_fallback(
    monkeypatch: pytest.MonkeyPatch,
    problem: coi.SingleOptimizable,
    optimizer: optimizers.Optimizer,
) -> None:
    # Given:
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
        solve = optimizers.make_solve_func(optimizer, problem)

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
