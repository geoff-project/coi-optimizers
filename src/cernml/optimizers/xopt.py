# SPDX-FileCopyrightText: 2025 CERN
# SPDX-FileCopyrightText: 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Compatibility with the XOpt optimization framework."""

from __future__ import annotations

import enum
import sys
import typing as t
import warnings

import numpy as np
import xopt

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

if t.TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


__all__ = (
    "XoptBayesian",
    "XoptRcds",
)


class BaseXoptOptimizer(Optimizer, coi.Configurable):
    """Base class for Xopt optimizers.

    Parameters:
        max_evaluations: Maximum number of calls to the objective
            function before optimization terminates.
        random_evaluations: Number of candidates that are sampled from
            a uniform distribution to initialize the Gaussian Process.
    """

    def __init__(
        self,
        *,
        max_evaluations: int = 100,
        random_evaluations: int,
    ) -> None:
        if not isinstance(max_evaluations, int):
            raise TypeError(
                f"`max_evaluations` must be an integer: {max_evaluations!r}"
            )
        if not isinstance(random_evaluations, int):
            raise TypeError(
                f"`random_evaluations` must be an integer: {random_evaluations!r}"
            )
        if max_evaluations <= 0:
            raise ValueError(f"invalid `max_evaluations`: {max_evaluations!r}")
        if not 0 <= random_evaluations <= max_evaluations:
            raise ValueError(f"invalid `random_evaluations`: {random_evaluations!r}")
        self.max_evaluations = max_evaluations
        self.random_evaluations = random_evaluations

    def make_generator(self, vocs: xopt.VOCS) -> xopt.Generator:
        """Instantiate the underlying Xopt generator.

        This is a template method to be overridden by subclasses.
        """
        raise NotImplementedError

    @override
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        vocs = get_vocs(bounds, constraints)

        def solve(objective: Objective, x_0: NDArray[np.floating]) -> OptimizeResult:
            objective_wrapper = make_objective_wrapper(objective, constraints, vocs)
            generator = self.make_generator(vocs)
            evaluator = xopt.Evaluator(
                function=objective_wrapper, max_workers=1, vectorized=False
            )
            xopt_main = xopt.Xopt(
                vocs=vocs,
                generator=generator,
                evaluator=evaluator,
                strict=True,
                max_evaluations=self.max_evaluations,
            )
            xopt_main.evaluate_data(params_to_dict(x_0, vocs))
            if self.random_evaluations:
                xopt_main.random_evaluate(self.random_evaluations)
            xopt_main.run()
            _, best_fun, best_params = vocs.select_best(xopt_main.data)
            return OptimizeResult(
                x=dict_to_params(best_params, vocs),
                fun=best_fun.item(),
                success=True,
                message="optimization finished",
                nit=xopt_main.n_data,
                nfev=xopt_main.n_data,
            )

        return solve

    @override
    def get_config(self) -> coi.Config:
        return coi.Config().add(
            "max_evaluations",
            self.max_evaluations,
            range=(1, np.inf),
            label="Iterations",
            help="Maximum number of function evaluations",
        )

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        self.max_evaluations = values.max_evaluations


@enum.unique
class BayesianMethod(enum.Enum):
    """Possible choices for the algorithm used by `XoptBayesian`."""

    UPPER_CONFIDENCE_BOUND = "Upper confidence bound"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.upper_confidence_bound.UpperConfidenceBoundGenerator`."""

    EXPECTED_IMPROVEMENT = "Expected improvement"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.expected_improvement.ExpectedImprovementGenerator`."""

    BAYESIAN_EXPLORATION = "Bayesian exploration"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.bayesian_exploration.BayesianExplorationGenerator`."""

    def get_generator(self) -> type[xopt.Generator]:
        """Return the chosen generator class."""
        return xopt.generators.get_generator(self.name.lower())


@enum.unique
class TurboController(enum.Enum):
    """Possible choices for the TuRBO controller used by `XoptBayesian`."""

    NONE = "Disabled"
    """Disables TuRBO."""

    OPTIMIZE = "Basic"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.turbo.OptimizeTurboController`."""

    SAFETY = "Safety-constrained"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.turbo.SafetyTurboController`."""

    ENTROPY = "Entropy-based"
    """Corresponds to the Xopt class
    `~xopt.generators.bayesian.turbo.EntropyTurboController`."""

    def get_controller(
        self,
    ) -> type[xopt.generators.bayesian.turbo.TurboController] | None:
        """Return the chosen controller class."""
        if self is TurboController.NONE:
            return None
        from xopt.generators.bayesian import turbo

        return getattr(turbo, f"{self.name.title()}TurboController")


class XoptBayesian(BaseXoptOptimizer):
    """Bayesian optimization as provided by XOpt.

    Parameters:
        max_evaluations: Maximum number of calls to the objective
            function before optimization terminates.
        random_evaluations: Number of candidates that are sampled from
            a uniform distribution to initialize the Gaussian Process.
        method: Concrete subclass of
            `~xopt.generators.bayesian.bayesian_generator.BayesianGenerator`
            that should be instantiated.
        beta: Exploration factor. Only used if *method* is
            `.UPPER_CONFIDENCE_BOUND`.

    For an explanation of the remaining parameters, see
    `~xopt.generators.bayesian.bayesian_generator.BayesianGenerator`
    in the Xopt_ docs.

    .. _Xopt: https://xopt.xopt.org/
    """

    def __init__(
        self,
        *,
        max_evaluations: int = 100,
        random_evaluations: int = 10,
        method: BayesianMethod = BayesianMethod.UPPER_CONFIDENCE_BOUND,
        beta: float = 2.0,
        n_monte_carlo_samples: int = 128,
        use_cuda: bool = False,
        max_travel_distances: list[float] | float | None = None,
        n_interpolate_points: int | None = None,
        turbo_controller: TurboController = TurboController.NONE,
    ):
        super().__init__(
            max_evaluations=max_evaluations,
            random_evaluations=random_evaluations,
        )
        self.n_monte_carlo_samples = n_monte_carlo_samples
        self.use_cuda = use_cuda
        self.max_travel_distances = max_travel_distances
        self.n_interpolate_points = n_interpolate_points
        self.method = method
        self.turbo_controller = turbo_controller
        self.beta = beta

    @override
    def make_generator(self, vocs: xopt.VOCS) -> xopt.Generator:
        method = self.method.name.lower()
        gen_class = xopt.generators.get_generator(method)
        max_travel_distances = (
            None
            if self.max_travel_distances is None
            else (
                [self.max_travel_distances] * len(vocs.variables)
                if isinstance(self.max_travel_distances, float)
                else self.max_travel_distances
            )
        )
        turbo_class = self.turbo_controller.get_controller()
        turbo_controller = None if turbo_class is None else turbo_class(vocs=vocs)
        kwargs = (
            {"beta": self.beta}
            if self.method == BayesianMethod.UPPER_CONFIDENCE_BOUND
            else {}
        )
        return gen_class(
            vocs=vocs,
            n_monte_carlo_samples=self.n_monte_carlo_samples,
            use_cuda=self.use_cuda,
            max_travel_distances=max_travel_distances,
            n_candidates=1,
            n_interpolate_points=self.n_interpolate_points,
            turbo_controller=turbo_controller,
            **kwargs,
        )

    @override
    def get_config(self) -> coi.Config:
        config = super().get_config()
        config.add(
            "random_evaluations",
            self.random_evaluations,
            range=(0, np.inf),
            label="Initial random evaluations",
            help="Number of initial random evaluations",
        )
        config.add(
            "n_monte_carlo_samples",
            self.n_monte_carlo_samples,
            range=(1, np.inf),
            label="Monte-Carlo samples",
            help="The number of Monte Carlo samples to use in the "
            "optimization process.",
        )
        config.add(
            "use_cuda",
            self.use_cuda,
            label="Use CUDA",
            help="If enabled, use CUDA if available and fall back to "
            "the CPU otherwise; if disabled, always use the CPU.",
        )
        config.add(
            "max_travel_distances",
            self.max_travel_distances or 0.0,
            range=(0.0, 2.0),
            label="Maximum travel distance",
            help="The limit for travel distance between points "
            "in normalized space; set to zero for no limit.",
        )
        config.add(
            "n_interpolate_points",
            self.n_interpolate_points or 0,
            range=(0, np.inf),
            label="Interpolation points",
            help="Number of interpolation points to generate between "
            "last observation and next observation.",
        )
        config.add(
            "method",
            self.method.value,
            choices=[m.value for m in BayesianMethod],
            label="Algorithm",
            help="Bayesian algorithm to use",
        )
        config.add(
            "beta",
            self.beta,
            range=(0.0, 10.0),
            label="UCB beta",
            help="Beta parameter for Upper confidence bound "
            "optimization; controls the trade-off between exploration "
            "and exploitation. Higher values of beta prioritize exploration.",
        )
        config.add(
            "turbo_controller",
            self.turbo_controller.value,
            choices=[m.value for m in TurboController],
            label="TuRBO",
            help="Controller for Trust-Region-based Bayesian Optimization",
        )
        return config

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        super().apply_config(values)
        if values.random_evaluations > values.max_evaluations:
            raise ValueError(
                f"{values.random_evaluations} not in range "
                f"[0, {values.max_evaluations}]"
            )
        self.random_evaluations = values.random_evaluations
        self.n_monte_carlo_samples = values.n_monte_carlo_samples
        self.use_cuda = values.use_cuda
        self.max_travel_distances = values.max_travel_distances or None
        self.n_interpolate_points = values.n_interpolate_points or None
        self.method = BayesianMethod(values.method)
        self.beta = values.beta
        self.turbo_controller = TurboController(values.turbo_controller)


class XoptRcds(BaseXoptOptimizer):
    """The RCDS algorithm as provided by XOpt.

    Parameters:
        max_evaluations: Maximum number of calls to the objective
            function before optimization terminates.

    For an explanation of the remaining parameters, see
    `~xopt:xopt.generators.sequential.RCDSGenerator` in the Xopt_ docs.

    .. _Xopt: https://xopt.xopt.org/
    """

    def __init__(
        self,
        *,
        max_evaluations: int = 100,
        init_mat: NDArray[np.floating] | None = None,
        noise: float = 0.00001,
        step: float = 0.01,
    ) -> None:
        super().__init__(max_evaluations=max_evaluations, random_evaluations=0)
        self.init_mat = init_mat
        self.noise = noise
        self.step = step
        # Quick and dirty validation of the arguments.
        config = self.get_config()
        self.apply_config(config.validate_all(config.get_field_values()))

    @override
    def make_generator(self, vocs: xopt.VOCS) -> xopt.Generator:
        gen_class: type[xopt.generators.sequential.rcds.RCDSGenerator] = (
            xopt.generators.get_generator("rcds")
        )
        return gen_class(
            vocs=vocs,
            init_mat=self.init_mat,
            noise=self.noise,
            step=self.step,
        )

    @override
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[coi.Constraint]
    ) -> Solve:
        if constraints:
            warnings.warn(
                "RCDS ignores constraints",
                category=IgnoredArgumentWarning,
                stacklevel=2,
            )
        return super().make_solve_func(bounds, constraints)

    @override
    def get_config(self) -> coi.Config:
        config = super().get_config()
        config.add(
            "noise",
            self.noise,
            label="Noise level",
            help="Estimated noise level of the objective function",
        )
        config.add(
            "step",
            self.step,
            label="Step size",
            help="Step size for the optimization",
        )
        return config

    @override
    def apply_config(self, values: coi.ConfigValues) -> None:
        super().apply_config(values)
        self.noise = values.noise
        self.step = values.step


def get_vocs(bounds: Bounds, constraints: t.Sequence[coi.Constraint], /) -> xopt.VOCS:
    """Convert bounds and constraints to an Xopt-compatible format.

    In particular, this generates names for all quantities:

    - the variables are named :code:`x1` through :code:`x{N}`;
    - the constraints (which in COI always have a lower and an upper
      bound) are split into a series :code:`c1-lb` through
      :code:`c{N}-lb` for the lower bound and :code:`c1-ub` through
      :code:`c{N}-ub` for the upper bound, skipping any trivial bounds
      that compare the constrained variable to infinity;
    - the objective is always called :code:`objective`.
    """
    return xopt.VOCS(
        variables={
            f"x{i}": [low, high] for i, (low, high) in enumerate(zip(*bounds), 1)
        },
        constraints=_translate_constraints(constraints),
        objectives={"objective": xopt.vocs.ObjectiveEnum.MINIMIZE},
    )


def params_to_dict(params: NDArray[np.double], vocs: xopt.VOCS) -> dict[str, float]:
    """Convert inputs from a flat 1D-array as used by COI to an Xopt dict."""
    # Use vocs.variables instead of vocs.variable_names; the latter is
    # sorted alphabetically, whereas we want the same sorting that we
    # used for creating the VOCS in the first place.
    return {name: value.item() for name, value in zip(vocs.variables, params.flat)}


def dict_to_params(
    input_dict: pd.Series[float] | t.Mapping[str, float | NDArray[np.double]],
    vocs: xopt.VOCS,
) -> NDArray[np.double]:
    """Convert inputs from Xopt to a flat 1D-array as used by COI."""
    # Some generators give us a dict containing `float` values; RCDS
    # gives us a dict containing 1D-arrays with a size of 1. This
    # function concatenates all of them into a simple 1D-array with as
    # many entries as we have variables.
    return np.fromiter(
        # Use vocs.variables instead of vocs.variable_names; the latter
        # is sorted alphabetically, whereas we want the same sorting
        # that we used for creating the VOCS in the first place.
        # Also, don't pass the shape as named argument; the argument
        # name changed between NumPy 1 and 2.
        (np.reshape(input_dict[name], ()) for name in vocs.variables),
        dtype=np.double,
        count=len(vocs.variables),
    )


def _translate_constraints(
    constraints: t.Sequence[coi.Constraint], /
) -> dict[str, tuple[xopt.vocs.ConstraintEnum, float]]:
    """Helper for `get_vocs()`."""
    ConstraintEnum = xopt.vocs.ConstraintEnum
    result = {}
    for i, c in enumerate(constraints, 1):
        try:
            lb = np.unique(c.lb).item()
        except ValueError:
            raise ValueError(f"Xopt requires scalar lower bounds: {c.lb!r}") from None
        if not np.isneginf(lb):
            result[f"c{i}-lb"] = (ConstraintEnum.GREATER_THAN, lb)
        try:
            ub = np.unique(c.ub).item()
        except ValueError:
            raise ValueError(f"Xopt requires scalar upper bounds: {c.ub!r}") from None
        if not np.isposinf(ub):
            result[f"c{i}-ub"] = (ConstraintEnum.LESS_THAN, ub)
    return result


def eval_constraints(
    constraints: t.Sequence[coi.Constraint], params: NDArray[np.double]
) -> dict[str, float]:
    """Evaluate constraints and pack results Xopt-compatibly.

    Note that if constraint *i* is two-sided, its value appears twice in
    the returned dict: once as :code:`c{i}-lb` and once as
    :code:`c{i}-ub`. Nonetheless the constraint is evaluated only once.
    """
    # The code here should always have the same structure as
    # `_translate_constraints()`.
    result = {}
    for i, c in enumerate(constraints, 1):
        value = _eval_constraint(c, params)
        if not np.isneginf(c.lb):
            result[f"c{i}-lb"] = value
        if not np.isposinf(c.ub):
            result[f"c{i}-ub"] = value
    return result


def _eval_constraint(constraint: coi.Constraint, params: NDArray[np.double]) -> float:
    """Helper for `eval_constraints()`."""
    # To support duck-typing and reduce dependency on SciPy, we check
    # for attributes here instead of instance checks. This is
    # technically superfluous because Xopt depends on SciPy.
    if callable(fun := getattr(constraint, "fun", None)):
        return float(fun(params))
    if (A := getattr(constraint, "A", None)) is not None:
        return float(A @ params)
    raise ValueError(f"constraint of unknown type: {constraint!r}")


def make_objective_wrapper(
    objective: Objective,
    constraints: t.Sequence[coi.Constraint],
    vocs: xopt.VOCS,
) -> t.Callable[[dict[str, NDArray[np.double]]], dict[str, float]]:
    """Wrap *objective* and *constraints* Xopt-compatibly.

    Every call to the returned wrapper evaluates *objective* and each of
    the *constraints* exactly once and returns the results in a dict as
    Xopt would expect.
    """

    def objective_wrapper(
        input_dict: dict[str, NDArray[np.double]], /
    ) -> dict[str, float]:
        params = dict_to_params(input_dict, vocs)
        result = float(objective(params))
        constraint_values = eval_constraints(constraints, params)
        constraint_values["objective"] = result
        return constraint_values

    return objective_wrapper
