# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = missing-function-docstring

"""Unit tests for the third-party wrappers."""

import typing as t

import numpy as np
import pytest

from cernml import coi, optimizers


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
def test_run_optimizer(name: str, nfev: int) -> None:
    def objective(params: np.ndarray[t.Any, np.dtype[np.floating]]) -> float:
        return float(np.linalg.norm(params))

    opt = configure_optimizer(optimizers.make(name))
    bounds = optimizers.Bounds(-np.ones(3), np.ones(3))
    solve = opt.make_solve_func(bounds=bounds, constraints=[])
    x_init = np.array([0.1, 0.2, 0.0])
    res = solve(objective, x_init)
    assert res.success
    assert res.nfev == nfev
