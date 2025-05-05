# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Bayesian optimization as provided by scikit-opt."""

from __future__ import annotations

import sys
import typing as t
import warnings

import numpy as np
import skopt.optimizer
from numpy.typing import NDArray

from cernml import coi

from ._interface import (
    Bounds,
    IgnoredArgumentWarning,
    Objective,
    Optimizer,
    OptimizeResult,
    Solve,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


__all__ = ("SkoptBayesian",)


class SkoptBayesian(Optimizer, coi.Configurable):
    """Adapter for Bayesian optimization via scikit-optimize."""

    def __init__(
        self,
        *,
        verbose: bool = True,
        check_convergence: bool = False,
        min_objective: float = 0.0,
        n_calls: int = 100,
        n_initial_points: int = 10,
        acq_func: str = "LCB",
        kappa_param: float = 1.96,
        xi_param: float = 0.01,
    ) -> None:
        self.verbose = verbose
        self.check_convergence = check_convergence
        self.min_objective = min_objective
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.kappa_param = kappa_param
        self.xi_param = xi_param
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    @override
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        if constraints:
            warnings.warn(
                "SkoptBayesian ignores constraints",
                category=IgnoredArgumentWarning,
                stacklevel=2,
            )
        callback = (
            (lambda res: res.fun < self.min_objective)
            if self.check_convergence
            else None
        )

        def solve(objective: Objective, x_0: NDArray[np.floating]) -> OptimizeResult:
            res = skopt.optimizer.gp_minimize(
                objective,
                x0=list(x_0),
                dimensions=zip(*bounds),
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
                kappa=self.kappa_param,
                xi=self.xi_param,
                verbose=self.verbose,
                callback=callback,
            )
            return OptimizeResult(
                x=np.asarray(res.x, dtype=float),
                fun=float(res.fun),
                success=True,
                message="",
                nit=len(res.func_vals),
                nfev=len(res.func_vals),
            )

        return solve

    @override
    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "n_calls",
            self.n_calls,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "n_initial_points",
            self.n_initial_points,
            range=(0, np.inf),
            help="Number of function evaluations before approximating "
            "with base estimator",
        )
        config.add(
            "acq_func",
            self.acq_func,
            choices=["LCB", "EI", "PI", "EIps", "PIps"],
            help="Function to minimize over the Gaussian prior",
        )
        config.add(
            "kappa_param",
            self.kappa_param,
            range=(0, np.inf),
            help='Only used with "LCB". Controls how much of the '
            "variance in the predicted values should be taken into "
            "account. If set to be very high, then we are favouring "
            "exploration over exploitation and vice versa.",
        )
        config.add(
            "xi_param",
            self.xi_param,
            range=(0, np.inf),
            help='Only used with "EI", "PI" and variants. Controls '
            "how much improvement one wants over the previous best "
            "values.",
        )
        config.add(
            "verbose",
            self.verbose,
            help="If enabled, print progress to stdout",
        )
        config.add(
            "check_convergence",
            self.check_convergence,
            help="Enable convergence check at every iteration. "
            "Without this, the algorithm always evaluates the "
            "function the maximum number of times.",
        )
        config.add(
            "min_objective",
            self.min_objective,
            help="If convergence check is enabled, end optimization "
            "below this value of the objective function.",
        )
        return config

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        if values.n_initial_points > values.n_calls:
            raise coi.BadConfig("n_initial_points must be less than maxfun")
        self.n_calls = values.n_calls
        self.n_initial_points = values.n_initial_points
        self.acq_func = values.acq_func
        self.kappa_param = values.kappa_param
        self.xi_param = values.xi_param
        self.verbose = values.verbose
        self.check_convergence = values.check_convergence
        self.min_objective = values.min_objective
