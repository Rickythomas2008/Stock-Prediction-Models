from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stockpred.errors import MissingRunConfig
from stockpred.runs import CONFIG_FILENAME, RunRecorder, config_sha256, load_run


def test_run_directory_has_hashed_config(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    info = recorder.start({"model": "lstm", "window": 20}, tags={"sprint": 1})

    assert info.run_dir.is_dir()
    assert (info.run_dir / CONFIG_FILENAME).is_file()
    payload = load_run(info.run_dir)
    assert payload["config"] == {"model": "lstm", "window": 20}
    assert payload["config_sha256"] == config_sha256({"window": 20, "model": "lstm"})
    assert payload["tags"] == {"sprint": 1}


def test_metrics_are_recorded(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    info = recorder.start({"model": "nhits"}, now=datetime(2024, 1, 8, tzinfo=UTC))
    path = recorder.record_metrics(info, {"sharpe": 1.2, "max_drawdown": -0.11})

    assert json.loads(path.read_text(encoding="utf-8")) == {"sharpe": 1.2, "max_drawdown": -0.11}


def test_missing_config_is_refused(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    info = recorder.start({"model": "lstm"})
    (info.run_dir / CONFIG_FILENAME).unlink()

    with pytest.raises(MissingRunConfig, match="no config.json"):
        load_run(info.run_dir)


def test_tampered_config_is_refused(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    info = recorder.start({"model": "lstm", "window": 20})
    path = info.run_dir / CONFIG_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["window"] = 60
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MissingRunConfig, match="modified after creation"):
        load_run(path.parent)


def test_run_id_is_stable_for_same_config_and_time(tmp_path: Path) -> None:
    stamp = datetime(2024, 1, 8, 9, 30, tzinfo=UTC)
    first = RunRecorder(tmp_path / "a").start({"model": "x"}, now=stamp)
    second = RunRecorder(tmp_path / "b").start({"model": "x"}, now=stamp)

    assert first.run_id == second.run_id
    with suppress(FileNotFoundError):
        (tmp_path / "unused").unlink()
