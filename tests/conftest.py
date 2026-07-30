from __future__ import annotations

import importlib.util
import pathlib
import site
import sys
import types

import pytest


@pytest.fixture(scope="session")
def upstream_webrtcvad():
    package_resources = types.ModuleType("pkg_resources")
    package_resources.get_distribution = lambda _: types.SimpleNamespace(
        version="2.0.10"
    )
    sys.modules.setdefault("pkg_resources", package_resources)
    candidates = [
        pathlib.Path(directory, "webrtcvad.py") for directory in site.getsitepackages()
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        pytest.skip("upstream webrtcvad is not installed")
    spec = importlib.util.spec_from_file_location("upstream_webrtcvad", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
