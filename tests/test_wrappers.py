# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = import-outside-toplevel
# pylint: disable = missing-function-docstring
# pylint: disable = redefined-outer-name

"""Unit tests for the third-party wrappers."""

import typing as t

import numpy as np
import pytest
from gym.spaces import Box

from cernml import coi, optimizers


@pytest.fixture
def problem() -> coi.SingleOptimizable:
    class _Problem(coi.SingleOptimizable):
        optimization_space = Box(-1.0, 1.0, shape=(3,), dtype=float)

        def get_initial_params(self) -> np.ndarray:
            return np.array([0.1, 0.2, 0.0])

        def compute_single_objective(self, params: np.ndarray) -> float:
            return float(np.linalg.norm(params))

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
    solve = opt.make_solve_func(
        optimizers.Bounds(space.low, space.high), problem.constraints
    )
    res = solve(problem.compute_single_objective, problem.get_initial_params())
    assert res.success
    assert res.nfev == nfev


def test_bad_bobyqa_bounds(problem: coi.SingleOptimizable) -> None:
    from cernml.optimizers.bobyqa import BobyqaException

    opt = optimizers.make("BOBYQA")
    space = problem.optimization_space
    assert isinstance(space, Box)
    solve = opt.make_solve_func(
        optimizers.Bounds(space.low, space.high), problem.constraints
    )
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
