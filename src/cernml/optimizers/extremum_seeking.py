# SPDX-FileCopyrightText: 2023-2024 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""The Extremum Seeking algorithm provided by :mod:`cernml.extremum_seeking`."""

from __future__ import annotations

import sys
import typing as t
import warnings

import numpy as np
from numpy.typing import NDArray

from cernml import coi, extremum_seeking

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


__all__ = ("ExtremumSeeking",)


class ExtremumSeeking(Optimizer, coi.Configurable):
    """Wrapper around an `~cernml.extremum_seeking.ExtremumSeeker`."""

    # pylint: disable = too-many-instance-attributes

    def __init__(
        self,
        *,
        check_convergence: bool = False,
        max_calls: int = 0,
        check_goal: bool = False,
        cost_goal: float = 0.0,
        gain: float = 0.2,
        oscillation_size: float = 0.1,
        oscillation_sampling: int = 10,
        decay_rate: float = 1.0,
    ) -> None:
        self.check_convergence = check_convergence
        self.max_calls = max_calls
        self.check_goal = check_goal
        self.cost_goal = cost_goal
        self.gain = gain
        self.oscillation_size = oscillation_size
        self.oscillation_sampling = oscillation_sampling
        self.decay_rate = decay_rate
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    @override
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        if constraints:
            warnings.warn(
                "ExtremumSeeking ignores constraints",
                category=IgnoredArgumentWarning,
                stacklevel=2,
            )

        def solve(objective: Objective, x_0: NDArray[np.floating]) -> OptimizeResult:
            res = extremum_seeking.optimize(
                objective,
                x0=x_0,
                max_calls=self.max_calls if self.max_calls else None,
                cost_goal=self.cost_goal if self.check_goal else None,
                bounds=bounds,
                gain=self.gain,
                oscillation_size=self.oscillation_size,
                oscillation_sampling=self.oscillation_sampling,
                decay_rate=self.decay_rate,
            )
            return OptimizeResult(
                x=np.asarray(res.params, dtype=float),
                fun=float(res.cost),
                success=True,
                message="",
                nit=res.nit,
                nfev=res.nit,
            )

        return solve

    @override
    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "max_calls",
            self.max_calls,
            range=(0, np.inf),
            help="Maximum number of function evaluations; if zero, there is no limit",
        )
        config.add(
            "check_goal",
            self.check_goal,
            help="If enabled, stop optimization when the objective "
            "function is below this value",
        )
        config.add(
            "cost_goal",
            self.cost_goal,
            help="If check_goal is enabled, end optimization when "
            "the objective goes below this value; if check_goal is "
            "disabled, this does nothing",
        )
        config.add(
            "gain",
            self.gain,
            range=(0.0, np.inf),
            help="Scaling factor applied to the objective function",
        )
        config.add(
            "oscillation_size",
            self.oscillation_size,
            range=(0.0, 1.0),
            help="Amplitude of the dithering oscillations; higher "
            "values make the parameters fluctuate stronger",
        )
        config.add(
            "oscillation_sampling",
            self.oscillation_sampling,
            range=(1, np.inf),
            help="Number of samples per dithering period; higher "
            "values make the parameters fluctuate slower",
        )
        config.add(
            "decay_rate",
            self.decay_rate,
            range=(0.0, 1.0),
            help="Decrease oscillation_size by this factor after every iteration",
        )
        return config

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        if values.gain == 0.0:
            raise coi.BadConfig("gain must not be zero")
        if values.oscillation_size == 0.0:
            raise coi.BadConfig("oscillation_size must not be zero")
        if values.decay_rate == 0.0:
            raise coi.BadConfig("decay_rate must not be zero")
        self.max_calls = values.max_calls
        self.check_goal = values.check_goal
        self.cost_goal = values.cost_goal
        self.gain = values.gain
        self.oscillation_size = values.oscillation_size
        self.oscillation_sampling = values.oscillation_sampling
        self.decay_rate = values.decay_rate
