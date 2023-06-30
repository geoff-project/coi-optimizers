# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Optimizers provided by the Scipy package."""

from __future__ import annotations

import typing as t
import warnings

import numpy as np
import scipy.optimize

from cernml import coi

from ._interface import (
    Bounds,
    IgnoredArgumentWarning,
    Objective,
    Optimizer,
    OptimizeResult,
    Solve,
)

__all__ = [
    "Cobyla",
    "NelderMeadSimplex",
    "Powell",
]


class Cobyla(Optimizer, coi.Configurable):
    """Adapter for the COBYLA algorithm."""

    def __init__(
        self,
        *,
        maxfun: int = 100,
        rhobeg: float = 1.0,
        rhoend: float = 0.05,
    ) -> None:
        self.maxfun = maxfun
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        lower, upper = bounds
        constraints = list(constraints)
        constraints.append(
            scipy.optimize.NonlinearConstraint(lambda x: x, lower, upper)
        )

        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="COBYLA",
                x0=x_0,
                constraints=constraints,
                options={
                    "maxiter": self.maxfun,
                    "rhobeg": self.rhobeg,
                    "tol": self.rhoend,
                },
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nfev,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "rhobeg",
            self.rhobeg,
            range=(0.0, 1.0),
            help="Reasonable initial changes to the variables",
        )
        config.add(
            "rhoend",
            self.rhoend,
            range=(0.0, 1.0),
            help="Step size below which the optimization is considered converged",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.rhobeg = values.rhobeg
        self.rhoend = values.rhoend


class NelderMeadSimplex(Optimizer, coi.Configurable):
    """Adapter for the Nelder–Mead algorithm."""

    DELTA_IF_ZERO: t.ClassVar[float] = 0.001
    DELTA_IF_NONZERO: t.ClassVar[float] = 0.05

    def __init__(
        self,
        *,
        maxfun: int = 100,
        adaptive: bool = False,
        tolerance: float = 0.05,
        delta_if_zero: float = DELTA_IF_ZERO,
        delta_if_nonzero: float = DELTA_IF_NONZERO,
    ) -> None:
        self.maxfun = maxfun
        self.adaptive = adaptive
        self.tolerance = tolerance
        self.delta_if_zero = delta_if_zero
        self.delta_if_nonzero = delta_if_nonzero
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    def make_solve_func(
        self,
        bounds: Bounds,
        constraints: t.Sequence[coi.Constraint],
    ) -> Solve:
        if constraints:
            warnings.warn(
                "NelderMeadSimplex ignores constraints", IgnoredArgumentWarning
            )

        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="Nelder-Mead",
                x0=x_0,
                tol=self.tolerance,
                bounds=scipy.optimize.Bounds(*bounds),
                options={
                    "maxfev": self.maxfun,
                    "adaptive": self.adaptive,
                    "initial_simplex": self._build_simplex(x_0),
                },
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nit,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "adaptive",
            self.adaptive,
            help="Adapt algorithm parameters to dimensionality of problem",
        )
        config.add(
            "tolerance",
            self.tolerance,
            range=(0.0, 1.0),
            help="Convergence tolerance",
        )
        config.add(
            "delta_if_nonzero",
            self.delta_if_nonzero,
            range=(-1.0, 1.0),
            default=type(self).DELTA_IF_NONZERO,
            help="Relative change to nonzero entries to get initial simplex",
        )
        config.add(
            "delta_if_zero",
            self.delta_if_zero,
            range=(-1.0, 1.0),
            default=type(self).DELTA_IF_ZERO,
            help="Absolute addition to zero entries to get initial simplex",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.adaptive = values.adaptive
        self.tolerance = values.tolerance
        self.delta_if_nonzero = values.delta_if_nonzero
        self.delta_if_zero = values.delta_if_zero

    def _build_simplex(self, x_0: np.ndarray) -> np.ndarray:
        """Build an initial simplex based on an initial point.

        This is identical to the simplex construction in Scipy, but
        makes the two scaling factors (``nonzdelt`` and ``zdelt``)
        configurable.

        See https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
        """
        dim = len(x_0)
        simplex = np.empty((dim + 1, dim), dtype=x_0.dtype)
        simplex[0] = x_0
        for i in range(dim):
            point = x_0.copy()
            if point[i] != 0.0:
                point[i] *= 1 + self.delta_if_nonzero
            else:
                point[i] = self.delta_if_zero
            simplex[i + 1] = point
        return simplex


class Powell(Optimizer, coi.Configurable):
    """Adapter for the Powell's conjugate-direction method."""

    def __init__(
        self,
        *,
        maxfun: int = 100,
        tolerance: float = 0.05,
        initial_step_size: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.maxfun = maxfun
        self.tolerance = tolerance
        self.initial_step_size = initial_step_size
        self.verbose = verbose
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    def make_solve_func(
        self,
        bounds: Bounds,
        constraints: t.Sequence[coi.Constraint],
    ) -> Solve:
        if constraints:
            warnings.warn("Powell ignores constraints", IgnoredArgumentWarning)

        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="Powell",
                x0=x_0,
                tol=self.tolerance,
                bounds=scipy.optimize.Bounds(*bounds),
                options={
                    "maxfev": self.maxfun,
                    "direc": self.initial_step_size * np.eye(len(x_0)),
                    "disp": self.verbose,
                },
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nit,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "tolerance",
            self.tolerance,
            range=(0.0, 1.0),
            help="Convergence tolerance",
        )
        config.add(
            "initial_step_size",
            self.initial_step_size,
            range=(1e-3, 1.0),
            help="Step size for the first iteration",
        )
        config.add(
            "verbose",
            self.verbose,
            help="If enabled, log convergence messages",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.tolerance = values.tolerance
        self.initial_step_size = values.initial_step_size
        self.verbose = values.verbose
