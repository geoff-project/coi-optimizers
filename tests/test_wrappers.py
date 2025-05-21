# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for the third-party wrappers."""

from __future__ import annotations

import contextlib
import inspect
import sys
import typing as t
import warnings
from unittest.mock import ANY, Mock, call

import numpy as np
import pytest
import xopt
from gymnasium.spaces import Box, Sequence
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, NonlinearConstraint

from cernml import coi, optimizers
from cernml.optimizers.bobyqa import BobyqaException
from cernml.optimizers.xopt import BayesianMethod, TurboController, XoptBayesian

if t.TYPE_CHECKING:
    from xopt import Xopt


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
            "XoptBayesian": 5,
            "XoptRcds": 5,
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
            "max_evaluations": 5,
            "random_evaluations": 3,
            "check_convergence": True,
        }
        raw_values = {
            k: overrides.get(k, v) for k, v in config.get_field_values().items()
        }
        opt.apply_config(config.validate_all(t.cast(dict, raw_values)))
    return opt


@pytest.fixture
def problem(request: pytest.FixtureRequest, nfev: int) -> coi.SingleOptimizable:
    mark: pytest.Mark | None = request.node.get_closest_marker("enable_constraints")

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

        if mark is not None:
            constraints = (LinearConstraint(np.identity(3, dtype=np.double), 0.0, 2.0),)

    return _Problem()


def test_builtin_names() -> None:
    assert ALL_OPTIMIZERS == (
        "BOBYQA",
        "COBYLA",
        "ExtremumSeeking",
        "NelderMeadSimplex",
        "Powell",
        "SkoptBayesian",
        "XoptBayesian",
        "XoptRcds",
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
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="xopt.generators.sequential.rcds",
        message="Conversion of an array with ndim > 0 to a scalar",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="botorch.optim.utils.numpy_utils",
        message="__array__ implementation doesn't accept a copy keyword",
    )
    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = optimizer.make_solve_func((space.low, space.high), problem.constraints)
    res = solve(problem.compute_single_objective, problem.get_initial_params())
    assert res.success
    assert res.nfev == nfev
    assert problem.get_wrapper_attr("num_calls") == nfev


@pytest.mark.enable_constraints
@pytest.mark.optimizer("XoptBayesian")
@pytest.mark.parametrize("method", BayesianMethod)
@pytest.mark.parametrize("turbo_controller", TurboController)
def test_run_bayesian(
    problem: coi.SingleOptimizable,
    optimizer: XoptBayesian,
    method: BayesianMethod,
    turbo_controller: TurboController,
    nfev: int,
) -> None:
    warnings.filterwarnings(
        "ignore",
        module="xopt.generators.bayesian.upper_confidence_bound",
        message="Using upper confidence bound with constraints",
    )
    warnings.filterwarnings(
        "ignore",
        module="xopt.generators.bayesian.custom_botorch.constrained_acquisition",
        message="The base acquisition function has negative values",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="botorch.optim.utils.numpy_utils",
        message="__array__ implementation doesn't accept a copy keyword",
    )
    optimizer.method = method
    optimizer.turbo_controller = turbo_controller
    solve = optimizers.make_solve_func(optimizer, problem)
    res = solve(problem.compute_single_objective, problem.get_initial_params())
    assert res.success
    assert res.nfev == nfev
    assert problem.get_wrapper_attr("num_calls") == nfev


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
    if optimizer.spec.name in ("COBYLA", "XoptBayesian"):
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
@pytest.mark.parametrize(
    ("low", "high"), [(-np.ones(3), np.ones(2)), (-np.ones(2), np.ones(3))]
)
def test_bad_bobyqa_bounds(
    optimizer: optimizers.Optimizer, low: NDArray[np.double], high: NDArray[np.double]
) -> None:
    solve = optimizer.make_solve_func((low, high), [])
    with pytest.raises(BobyqaException, match="must have same shape as x0"):
        solve(Mock(name="objective"), np.zeros(2))


@pytest.mark.optimizer("BOBYQA")
def test_bad_bobyqa_shapes(optimizer: optimizers.Optimizer) -> None:
    from cernml.optimizers.bobyqa import BobyqaException

    x0 = np.zeros((2, 2))
    low = x0 - 1.0
    high = x0 + 1.0
    solve = optimizer.make_solve_func((low, high), [])
    with pytest.raises(BobyqaException):
        solve(Mock(name="objective"), x0)


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


@pytest.mark.parametrize("name", ["XoptRcds", "XoptBayesian"])
def test_xopt_bad_max_evaluations(name: str) -> None:
    with pytest.raises(ValueError, match="max_evaluations"):
        optimizers.make(name, max_evaluations=0)


@pytest.mark.parametrize("name", ["XoptRcds", "XoptBayesian"])
def test_xopt_bad_max_evaluations_type(name: str) -> None:
    with pytest.raises(TypeError, match="max_evaluations"):
        optimizers.make(name, max_evaluations=t.cast(int, "1"))


def test_xopt_bad_random_evaluations_type() -> None:
    with pytest.raises(TypeError, match="random_evaluations"):
        optimizers.make("XoptBayesian", random_evaluations=t.cast(int, "1"))


@pytest.mark.parametrize("value", [-1, 10])
def test_xopt_bad_random_evaluations(value: int) -> None:
    with pytest.raises(ValueError, match="random_evaluations"):
        optimizers.make("XoptBayesian", max_evaluations=10, random_evaluations=value)


@pytest.mark.optimizer("XoptBayesian")
def test_xopt_bayesian_constraints(optimizer: XoptBayesian) -> None:
    warnings.filterwarnings(
        "ignore",
        module="xopt.generators.bayesian.upper_confidence_bound",
        message="Using upper confidence bound with constraints",
    )
    x0 = np.arange(3) * 0.1 - 0.1
    A = np.fliplr(np.diag(np.ones(3)))
    lb = np.arange(-5.0, -2.0)
    ub = np.arange(1.0, 4.0)
    constraints = [
        LinearConstraint(A, lb=lb),
        LinearConstraint(-A, ub=ub),
        LinearConstraint(np.ones((1, 3))),
        LinearConstraint(np.ones((1, 3)), lb=-1.0, ub=1.0),
        NonlinearConstraint(Mock(return_value=x0 + 1), lb=lb, ub=ub + np.inf),
        NonlinearConstraint(Mock(return_value=x0 - 1), lb=lb - np.inf, ub=ub),
        NonlinearConstraint(Mock(return_value=-0.5), lb=-np.inf, ub=np.inf),
        NonlinearConstraint(Mock(return_value=-0.5), lb=-2.0, ub=2.0),
    ]
    optimizer.random_evaluations = 0
    optimizer.max_evaluations = 1
    solve = optimizer.make_solve_func((x0 - 1.0, x0 + 1.0), constraints)
    xopt: Xopt | None = None

    def objective(params: NDArray[np.double]) -> float:
        nonlocal xopt
        frame = sys._getframe(1)
        while frame.f_code is not solve.__code__:
            assert frame.f_back is not None, "reached end of frame stack"
            frame = frame.f_back
        xopt = frame.f_locals.get("xopt_main")
        return -7.0

    solve(objective, x0)
    assert xopt is not None
    assert xopt.vocs.constraints == {
        "c1-lb1": ["GREATER_THAN", -5.0],
        "c1-lb2": ["GREATER_THAN", -4.0],
        "c1-lb3": ["GREATER_THAN", -3.0],
        "c2-ub1": ["LESS_THAN", 1.0],
        "c2-ub2": ["LESS_THAN", 2.0],
        "c2-ub3": ["LESS_THAN", 3.0],
        "c4-lb1": ["GREATER_THAN", -1.0],
        "c4-ub1": ["LESS_THAN", 1.0],
        "c5-lb1": ["GREATER_THAN", -5.0],
        "c5-lb2": ["GREATER_THAN", -4.0],
        "c5-lb3": ["GREATER_THAN", -3.0],
        "c6-ub1": ["LESS_THAN", 1.0],
        "c6-ub2": ["LESS_THAN", 2.0],
        "c6-ub3": ["LESS_THAN", 3.0],
        "c8-lb": ["GREATER_THAN", -2.0],
        "c8-ub": ["LESS_THAN", 2.0],
    }
    assert xopt.data.T.to_dict() == {
        0: {
            "x1": -0.1,
            "x2": 0.0,
            "x3": 0.1,
            "c1-lb1": 0.1,
            "c1-lb2": 0.0,
            "c1-lb3": -0.1,
            "c2-ub1": -0.1,
            "c2-ub2": 0.0,
            "c2-ub3": 0.1,
            "c4-lb1": 0.0,
            "c4-ub1": 0.0,
            "c5-lb1": 0.9,
            "c5-lb2": 1.0,
            "c5-lb3": 1.1,
            "c6-ub1": -1.1,
            "c6-ub2": -1.0,
            "c6-ub3": -0.9,
            "c8-lb": -0.5,
            "c8-ub": -0.5,
            "objective": -7.0,
            "xopt_runtime": ANY,
            "xopt_error": False,
        }
    }


@pytest.mark.optimizer("XoptBayesian")
def test_xopt_config_bad_random_evaluations(optimizer: XoptBayesian) -> None:
    config = optimizer.get_config()
    values = config.get_field_values()
    values["max_evaluations"] = 10
    values["random_evaluations"] = 20
    with pytest.raises(ValueError, match=r"^20 not in range \[0, 10\]$"):
        optimizer.apply_config(config.validate_all(values))


@pytest.mark.optimizer("XoptBayesian")
def test_xopt_config_bad_constraint_shape(optimizer: XoptBayesian) -> None:
    lb = -np.ones((3, 3))
    ub = +np.ones((3, 3))
    constraints = [NonlinearConstraint(Mock(name="fun"), lb, ub)]
    with pytest.raises(ValueError, match="constraints must be at most 1D, not 2D"):
        optimizer.make_solve_func((lb, ub), constraints)


@pytest.mark.optimizer("XoptBayesian")
def test_xopt_config_bad_constraint_value_shape(optimizer: XoptBayesian) -> None:
    warnings.filterwarnings(
        "ignore",
        module="xopt.generators.bayesian.upper_confidence_bound",
        message="Using upper confidence bound with constraints",
    )
    lb = -np.ones(3)
    ub = +np.ones(3)
    constraints: list[coi.Constraint] = []

    def constraint(x0: NDArray[np.double]) -> NDArray[np.double]:
        [self] = constraints
        self.lb = -np.ones((3, 3))
        self.lb = +np.ones((3, 3))
        return np.zeros((3, 3))

    constraints.append(NonlinearConstraint(constraint, lb, ub))
    solve = optimizer.make_solve_func((lb, ub), constraints)
    with pytest.raises(
        xopt.errors.XoptError,
        match="ValueError: constraints must be at most 1D, not 2D",
    ):
        solve(Mock(name="objective", return_value=0.0), np.zeros(3))


@pytest.mark.optimizer("XoptBayesian")
def test_xopt_config_bad_constraint_type(optimizer: XoptBayesian) -> None:
    warnings.filterwarnings(
        "ignore",
        module="xopt.generators.bayesian.upper_confidence_bound",
        message="Using upper confidence bound with constraints",
    )
    x0 = np.zeros(3)
    lb = x0 - 1.0
    ub = x0 + 1.0
    constraints = [Mock(name="constraint", spec=["lb", "ub"], lb=lb, ub=ub)]
    solve = optimizer.make_solve_func((lb, ub), constraints)
    with pytest.raises(
        xopt.errors.XoptError, match="TypeError: constraint of unknown type:"
    ):
        solve(Mock(name="objective", return_value=0.0), x0)


@pytest.mark.optimizer("BOBYQA")
def test_gym_compatibility_shim(
    problem: coi.SingleOptimizable, optimizer: optimizers.Optimizer
) -> None:
    class Box:
        __module__ = "gym.spaces.box"
        low: NDArray[np.double]
        high: NDArray[np.double]
        dtype = np.double

    space = Box()
    space.low = -12.0 * np.ones((3, 2, 1))
    space.high = -12.0 * np.ones((3, 2, 1))
    problem.optimization_space = t.cast(t.Any, space)
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
    assert call("gymnasium", ANY, None, ("spaces",), 0) in imp.call_args_list
    assert call("gym", ANY, None, ("spaces",), 0) in imp.call_args_list
    box = gym.spaces.flatten_space.return_value
    variables = inspect.getclosurevars(solve)
    low, high = variables.nonlocals["bounds"]
    assert low == box.low
    assert high == box.high
