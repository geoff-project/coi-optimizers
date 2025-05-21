# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""The BOBYQA algorithm as provided by Py-BOBYQA."""

from __future__ import annotations

import sys
import typing as t
import warnings

import numpy as np
import pybobyqa
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


__all__ = (
    "Bobyqa",
    "BobyqaException",
)


class BobyqaException(Exception):
    """BOBYQA failed in an exceptional manner.

    Most importantly, this includes invalid parameter shapes and bounds.
    It does not cover divergent behavior. :class:`.OptimizeResult` is
    used in this case.
    """


class Bobyqa(Optimizer, coi.Configurable):
    """The BOBYQA algorithm as provided by Py-BOBYQA.

    The parameters are identical to those given in the `Py-BOBYQA user
    guide`_.

    .. _Py-BOBYQA user guide:
        https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html#optional-arguments
    """

    def __init__(
        self,
        *,
        maxfun: int = 100,
        rhobeg: float = 0.25,
        rhoend: float = 0.025,
        nsamples: int = 1,
        seek_global_minimum: bool = False,
        objfun_has_noise: bool = False,
        scaling_within_bounds: bool = True,
    ) -> None:
        self.maxfun = maxfun
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.nsamples = nsamples
        self.seek_global_minimum = seek_global_minimum
        self.objfun_has_noise = objfun_has_noise
        self.scaling_within_bounds = scaling_within_bounds
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    @override
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        if constraints:
            warnings.warn(
                "BOBYQA ignores constraints",
                category=IgnoredArgumentWarning,
                stacklevel=2,
            )

        def solve(objective: Objective, x_0: NDArray[np.floating]) -> OptimizeResult:
            _assert_bounds_shape(bounds, expected=np.shape(x_0))
            nsamples = self.nsamples
            res = pybobyqa.solve(
                objective,
                x0=np.asarray(x_0, dtype=float),
                bounds=bounds,
                rhobeg=self.rhobeg,
                rhoend=self.rhoend,
                maxfun=self.maxfun,
                seek_global_minimum=self.seek_global_minimum,
                objfun_has_noise=self.objfun_has_noise,
                nsamples=lambda *_: nsamples,
                scaling_within_bounds=self.scaling_within_bounds,
            )
            if res.flag < 0:
                raise BobyqaException(res.msg)
            return OptimizeResult(
                x=np.asarray(res.x, dtype=float),
                fun=float(res.f),
                success=res.flag == res.EXIT_SUCCESS,
                message=res.msg,
                nit=res.nx,
                nfev=res.nf,
            )

        return solve

    @override
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
            help="Initial size of the trust region",
        )
        config.add(
            "rhoend",
            self.rhoend,
            range=(0.0, 1.0),
            help="Step size below which the optimization is considered converged",
        )
        config.add(
            "nsamples",
            self.nsamples,
            range=(1, 100),
            help="Number of measurements which to average over in each iteration",
        )
        config.add(
            "seek_global_minimum",
            self.seek_global_minimum,
            help="Enable additional logic to avoid local minima",
        )
        config.add(
            "objfun_has_noise",
            self.objfun_has_noise,
            help="Enable additional logic to handle non-deterministic environments",
        )
        config.add(
            "scaling_within_bounds",
            self.scaling_within_bounds,
            help="Internally parameters into the space [0; 1]",
        )
        return config

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.rhobeg = values.rhobeg
        self.rhoend = values.rhoend
        self.nsamples = values.nsamples
        self.seek_global_minimum = values.seek_global_minimum
        self.objfun_has_noise = values.objfun_has_noise
        self.scaling_within_bounds = values.scaling_within_bounds


def _assert_bounds_shape(bounds: Bounds, expected: tuple[int, ...]) -> None:
    low, high = bounds
    try:
        np.broadcast_to(low, expected)
    except ValueError as exc:
        raise BobyqaException("lower bounds must have same shape as x0") from exc
    try:
        np.broadcast_to(high, expected)
    except ValueError as exc:
        raise BobyqaException("upper bounds must have same shape as x0") from exc
