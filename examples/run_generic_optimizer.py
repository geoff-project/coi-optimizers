# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Example of how to configure and run an optimizer generically."""

from __future__ import annotations

import dataclasses
import sys
import typing as t
import warnings

import click
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from cernml import coi, optimizers

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class ConvexToyProblem(coi.SingleOptimizable):
    """Minimal toy problem for the optimizers.

    This is the 3D function $f(x, y, z) = √(x² + y² + z²)$. The goal is
    to find its minimum from the constant initial point
    $(x, y, z) = (0.1, 0.2, 0.0)$.
    """

    optimization_space = Box(-1.0, 1.0, shape=[3], dtype=np.double)

    @override
    def get_initial_params(
        self, *, seed: int | None = None, options: dict[str, t.Any] | None = None
    ) -> NDArray:
        super().get_initial_params(seed=seed, options=options)
        return np.array([0.1, 0.2, 0.0])

    @override
    def compute_single_objective(self, params: NDArray) -> float:
        return float(np.linalg.norm(params))


def apply_config(
    optimizer: optimizers.AnyOptimizer,
    nfev: int | None = None,
    extra_config: dict[str, str] | None = None,
) -> optimizers.AnyOptimizer:
    """Apply all config options to the given optimizer.

    Args:
        optimizer: The optimizer to configure.
        nfev: Maximum number of function evaluations. This is
            translated to the config name appropriate for *optimizer*.
            (This is necessary because the name of this parameter is
            inconsistent between optimizers.)
        extra_config: Any additional fields to configure on the
            *optimizer*. If the name is not known, it is ignored and a
            warning is issued.

    Returns:
        *optimizer* after configuration.
    """
    # Not every optimizer is configurable. Check that!
    if not coi.is_configurable(optimizer):
        warnings.warn(f"Not configurable: {optimizer!r}", stacklevel=2)
        return optimizer
    # These are all the values we want to change by default. Not all of
    # them are known to all optimizers.
    overrides = {
        "verbose": False,
        "max_calls": nfev or 20,
        "n_calls": nfev or 15,
        "maxfun": nfev or 100,
        "check_convergence": True,
        "max_evaluations": nfev or 100,
    }
    # Acquire the optimizer's presets. If we have an override for any of
    # them, use it instead.
    config: coi.Config = optimizer.get_config()
    raw_values: dict[str, t.Any] = {
        name: overrides.get(name, preset_value)
        for name, preset_value in config.get_field_values().items()
    }
    # Now apply all the extra config values. Ignore those that are
    # unknown to the optimizer.
    for name, value in (extra_config or {}).items():
        if name not in raw_values:
            warnings.warn(f"ignoring unknown config: {name}", stacklevel=2)
            continue
        raw_values[name] = value if value.lower() != "false" else ""
    optimizer.apply_config(config.validate_all(t.cast(dict, raw_values)))
    return t.cast(optimizers.AnyOptimizer, optimizer)


def _get_opt_name(opt: optimizers.Optimizer) -> str:
    return repr(getattr(opt.spec, "name", opt))


def show_opt_config(opt: optimizers.OptimizerWithSpec) -> None:
    """Show the config of a given optimizer."""
    click.echo(f"Configs for optimizer {click.style(opt.spec.name, fg='blue')}:")
    if not coi.is_configurable(opt):
        click.echo("    <not configurable>")
        return
    for field in opt.get_config().fields():
        label = field.label
        value = repr(field.value)
        # Show additional fields *if* they have non-default values.
        extra_info = [
            f"{attr!s}={getattr(field, attr)!r}"
            for attr in ["range", "choices", "default"]
            if getattr(field, attr) is not None
        ]
        # Format as " (name=value, name=value, ...)" or nothing.
        extra_string = ", ".join(extra_info).join([" (", ")"]) if extra_info else ""
        click.echo(f"    {label}: {click.style(value, fg='blue')}{extra_string}")


def show_opt_result(
    opt: optimizers.OptimizerWithSpec, res: optimizers.OptimizeResult
) -> None:
    """Show the results of an optimization run."""
    click.echo(f"Results for optimizer {click.style(opt.spec.name, fg='blue')}:")
    for label, value in dataclasses.asdict(res).items():
        click.echo(f"    {label}: {click.style(repr(value), fg='blue')}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-o",
    "--optimizer",
    help="The algorithm to use",
    default="BOBYQA",
    type=click.Choice(list(optimizers.registry.keys()), case_sensitive=False),
)
@click.option(
    "-n",
    "--nfev",
    help="number of iterations; default depends on optimizer",
    type=click.IntRange(min=1),
)
@click.option(
    "--no-run",
    help="don't run the optimizer, just show the config",
    is_flag=True,
)
@click.argument("extra_config", metavar="CONFIG=VALUE...", nargs=-1)
def main(
    optimizer: str = "BOBYQA",
    nfev: int | None = None,
    no_run: bool = False,
    extra_config: tuple[str, ...] = (),
) -> None:
    """Run a generic optimizer on a minimal optimization problem.

    Any optimizer can be configured arbitrarily by passing as many pairs
    of config name and value as desired.
    """
    opt = apply_config(
        optimizers.make(optimizer),
        nfev,
        extra_config=dict(extra.split("=", 1) for extra in extra_config),
    )
    show_opt_config(opt)
    if no_run:
        return
    env = ConvexToyProblem()
    res = optimizers.solve(opt, env)
    show_opt_result(opt, res)


if __name__ == "__main__":
    main()
