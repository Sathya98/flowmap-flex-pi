"""Core tests run without FlexPi: importing ``flexpi`` fails inside this suite.

The block is process-wide, so it is installed only when pytest runs nothing but
``flowmap_core`` (``pytest flowmap_core/tests``: the FlexPi-free gate). Invoked together
with FlexPi's tests, the core tests still run, without the import check.
"""
import importlib.abc
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]


class _BlockFlexPi(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flexpi" or name.startswith("flexpi."):
            raise ImportError(f"flowmap_core tests must not import {name}")
        return None


def _core_only(config):
    paths = [Path(str(a).split("::")[0]).resolve() for a in config.args] or [Path.cwd()]
    return all(p == CORE or CORE in p.parents for p in paths)


def pytest_configure(config):
    config._flowmap_core_only = _core_only(config)
    if config._flowmap_core_only:
        sys.meta_path.insert(0, _BlockFlexPi())


@pytest.fixture(autouse=True)
def no_flexpi(request):
    yield
    if request.config._flowmap_core_only:
        assert not any(n == "flexpi" or n.startswith("flexpi.") for n in sys.modules), "flexpi was imported"
