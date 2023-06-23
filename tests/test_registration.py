# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = missing-function-docstring
# pylint: disable = redefined-outer-name

"""Unit tests for the third-party wrappers."""

import typing as t
from unittest.mock import MagicMock, Mock, NonCallableMock, patch

import pytest

from cernml.optimizers import _registration as _reg

metadata = _reg.metadata


@pytest.fixture
def opt_class() -> t.Type[_reg.Optimizer]:
    return type("MockOptimizer", (Mock, _reg.Optimizer), {"make_solve_func": Mock()})


@pytest.fixture
def entry_point(opt_class: t.Type[_reg.Optimizer]) -> metadata.EntryPoint:
    # None of the attributes that we pass to `EntryPoint` will be used
    # in `res`. We only pass real strings to avoid assertion errors
    # while `Mock` iterates over `vars(spec_ep)`.
    spec_ep = metadata.EntryPoint("name", "value", "group")
    res = Mock(spec=spec_ep)
    res.load.return_value = opt_class
    return res


def test_spec_from_entry_point(entry_point: metadata.EntryPoint) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    assert spec.name == entry_point.name
    assert spec.value == entry_point.value
    assert spec.dist == entry_point.dist


def test_spec_from_optimizer() -> None:
    name = Mock()
    opt = MagicMock()
    opt.__qualname__ = "<locals>.MockOptimizer"
    spec = _reg.OptimizerSpec.from_optimizer(name, opt)
    assert spec.name == name
    assert spec.value == "unittest.mock:<locals>.MockOptimizer"
    assert spec.dist is None


def test_spec_from_optimizer_with_dist() -> None:
    name = Mock()
    opt = MagicMock(__qualname__="<locals>.MockOptimizer")
    dist = Mock()
    spec = _reg.OptimizerSpec.from_optimizer(name, opt, dist)
    assert spec.dist is dist


@pytest.mark.parametrize(
    "do_dist, do_load, string",
    [
        (False, False, "<OptimizerSpec('name', 'package:Class')>"),
        (False, True, "<OptimizerSpec('name', \"<class 'abc.MockOptimizer'>\")>"),
        (True, False, "<OptimizerSpec('dist/name', 'package:Class')>"),
        (True, True, "<OptimizerSpec('dist/name', \"<class 'abc.MockOptimizer'>\")>"),
    ],
)
def test_spec_str(
    entry_point: metadata.EntryPoint, do_dist: bool, do_load: bool, string: str
) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    t.cast(Mock, entry_point).name = "name"
    t.cast(Mock, entry_point).value = "package:Class"
    if do_dist:
        assert entry_point.dist
        t.cast(Mock, entry_point.dist).name = "dist"
    else:
        t.cast(Mock, entry_point).dist = None
    if do_load:
        spec.load()
    assert str(spec) == string


def test_spec_load(entry_point: metadata.EntryPoint) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    opt = spec.load()
    assert opt is t.cast(Mock, entry_point.load).return_value


def test_spec_load_caches(entry_point: metadata.EntryPoint) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    opt1 = spec.load()
    opt2 = spec.load()
    t.cast(Mock, entry_point.load).assert_called_once()
    assert opt1 is opt2


def test_spec_load_is_noop_with_class(opt_class: t.Type[_reg.Optimizer]) -> None:
    # pylint: disable = protected-access
    name = Mock(spec=str)
    spec = _reg.OptimizerSpec.from_optimizer(name, opt_class)
    spec._ep = Mock()
    load_result = spec.load()
    assert load_result is opt_class
    t.cast(Mock, spec._ep).load.assert_not_called()


def test_spec_load_warns_if_not_class(entry_point: metadata.EntryPoint) -> None:
    t.cast(Mock, entry_point.load).return_value = Mock()
    spec = _reg.OptimizerSpec(entry_point)
    with pytest.warns(_reg.TypeWarning, match="not a type"):
        spec.load()


def test_spec_load_requires_optimizer(entry_point: metadata.EntryPoint) -> None:
    t.cast(Mock, entry_point.load).return_value = type("NotOptimizer", (Mock,), {})
    spec = _reg.OptimizerSpec(entry_point)
    with pytest.warns(_reg.TypeWarning, match="not a subclass"):
        spec.load()


def test_spec_make(entry_point: metadata.EntryPoint) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    res = spec.make()
    opt_class = t.cast(Mock, entry_point.load).return_value
    assert isinstance(res, opt_class)


def test_spec_make_adds_spec_to_opt(entry_point: metadata.EntryPoint) -> None:
    spec = _reg.OptimizerSpec(entry_point)
    res = spec.make()
    assert res.spec is spec


def test_func_spec() -> None:
    name = Mock()
    with patch("cernml.optimizers._registration.registry") as registry:
        _reg.spec(name)
    registry.__getitem__.assert_called_with(name)


def test_func_spec_with_bad_name() -> None:
    with patch("cernml.optimizers._registration.registry", {}):
        with pytest.raises(_reg.OptimizerNotFound):
            _reg.spec(Mock())


def test_func_make() -> None:
    name = Mock()
    with patch("cernml.optimizers._registration.spec") as spec:
        opt = _reg.make(name)
    spec.assert_called_with(name)
    assert opt == spec.return_value.make.return_value


def test_register_with_str() -> None:
    name = Mock(spec=str)
    value = Mock(spec=str)
    dist = Mock(spec=metadata.Distribution)
    registry: t.Dict[str, _reg.OptimizerSpec] = {}
    with patch("cernml.optimizers._registration.registry", registry):
        _reg.register(name, value, dist)
    reg_name, spec = registry.popitem()
    assert spec.name == name == reg_name
    assert spec.value == value
    assert spec.dist == dist


def test_register_with_class(opt_class: t.Type[_reg.Optimizer]) -> None:
    name = Mock(spec=str)
    dist = Mock(spec=metadata.Distribution)
    registry: t.Dict[str, _reg.OptimizerSpec] = {}
    with patch("cernml.optimizers._registration.registry", registry):
        _reg.register(name, opt_class, dist)
    reg_name, spec = registry.popitem()
    assert spec.name == name == reg_name
    assert spec.load() == opt_class
    assert spec.dist == dist


def test_register_with_bad_arg() -> None:
    name = Mock(spec=str)
    opt_class = NonCallableMock()
    with pytest.raises(TypeError, match="must be a string or an Optimizer subclass"):
        _reg.register(name, opt_class)


def test_register_twice() -> None:
    name = Mock(spec=str)
    old_value = Mock(spec=str)
    new_value = Mock(spec=str)
    registry: t.Dict[str, _reg.OptimizerSpec] = {}
    with patch("cernml.optimizers._registration.registry", registry):
        _reg.register(name, old_value)
        with pytest.warns(_reg.DuplicateOptimizerWarning, match=str(old_value)):
            _reg.register(name, new_value)
    reg_name, spec = registry.popitem()
    assert not registry
    assert reg_name == name
    assert spec.value == new_value
