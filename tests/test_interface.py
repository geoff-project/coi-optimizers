# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for the abstract base class."""

import typing as t

import pytest

from cernml.coi import Constraint
from cernml.optimizers import Bounds, Optimizer, Solve


def test_abstract_method_raises() -> None:
    class Impl(Optimizer):
        # pylint: disable = useless-parent-delegation
        def make_solve_func(
            self, bounds: Bounds, constraints: t.Sequence[Constraint]
        ) -> Solve:
            return super().make_solve_func(  # type: ignore[safe-super]
                bounds, constraints
            )

    with pytest.raises(NotImplementedError):
        Impl().make_solve_func(*t.cast(t.List[t.Any], [None, None]))
