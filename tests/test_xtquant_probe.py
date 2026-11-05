from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import ModuleType

import pytest


def _probe_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "xtquant_probe.py"
    spec = importlib.util.spec_from_file_location("xtquant_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_call_with_timeout_returns_value() -> None:
    probe = _probe_module()

    value, error = probe.call_with_timeout(lambda: 42, timeout=2.0)

    assert value == 42
    assert error == ""


def test_call_with_timeout_captures_exception() -> None:
    probe = _probe_module()

    def boom() -> None:
        raise RuntimeError("no client")

    value, error = probe.call_with_timeout(boom, timeout=2.0)

    assert value is None
    assert "RuntimeError" in error and "no client" in error


def test_call_with_timeout_does_not_hang() -> None:
    """A vendor call with no client answering blocks forever; the guard must fire."""
    probe = _probe_module()

    def slow() -> str:
        time.sleep(30)
        return "never"

    started = time.monotonic()
    value, error = probe.call_with_timeout(slow, timeout=0.2)
    elapsed = time.monotonic() - started

    assert value is None
    assert "超时" in error
    assert elapsed < 5


def test_explicit_path_without_xtquant_reports_not_installed(tmp_path: Path) -> None:
    probe = _probe_module()

    module, location, notes = probe.import_xtquant(str(tmp_path))

    assert module is None
    assert location == ""
    assert any("xtquant" in note for note in notes)


@pytest.mark.parametrize("name", ["candidate_parents", "probe_data", "import_xtquant"])
def test_public_helpers_exist(name: str) -> None:
    probe = _probe_module()
    assert callable(getattr(probe, name))
