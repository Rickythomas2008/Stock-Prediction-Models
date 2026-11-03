from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpred.monitor import (
    GENESIS,
    append_record,
    detect_drift,
    hit_rate,
    last_payload,
    population_stability_index,
    read_records,
    verify_chain,
)


def test_empty_chain_is_reported_missing(tmp_path: Path) -> None:
    status = verify_chain(tmp_path / "absent.jsonl")
    assert not status.ok
    assert "does not exist" in (status.reason or "")


def test_append_builds_a_valid_chain(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    for index in range(3):
        append_record(log, {"as_of": f"2024-03-0{index + 1}", "n": index})

    status = verify_chain(log)
    assert status.ok
    assert status.records == 3

    records = list(read_records(log))
    assert records[0]["prev"] == GENESIS
    assert records[1]["prev"] == records[0]["hash"]
    assert records[2]["seq"] == 2


def test_edited_payload_breaks_the_chain(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    append_record(log, {"n": 0})
    append_record(log, {"n": 1})

    lines = log.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["payload"]["n"] = 99
    lines[1] = json.dumps(record, sort_keys=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = verify_chain(log)
    assert not status.ok
    assert status.broken_at == 1
    assert "hash" in (status.reason or "")


def test_deleted_record_breaks_the_chain(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    for index in range(3):
        append_record(log, {"n": index})

    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    status = verify_chain(log)
    assert not status.ok
    assert status.broken_at == 1


def test_corrupt_line_is_reported(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    append_record(log, {"n": 0})
    log.write_text(log.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")

    status = verify_chain(log)
    assert not status.ok
    assert "invalid json" in (status.reason or "")


def test_last_payload_returns_tail(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    append_record(log, {"n": 1})
    append_record(log, {"n": 2})
    assert last_payload(log)["n"] == 2
    assert last_payload(tmp_path / "absent.jsonl") is None


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 300)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_psi_flags_a_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 300)
    shifted = rng.normal(3, 1, 300)
    assert population_stability_index(base, shifted) > 0.25


def test_psi_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        population_stability_index([1, 2, 3], [1, 2, 3, 4, 5, 6])


def test_drift_report_is_clean_when_nothing_moved() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(5, 1, 200)})
    report = detect_drift(frame, frame.copy())
    assert report.ok
    assert report.summary() == "no drift detected"


def test_drift_flags_feature_shift() -> None:
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"f1": rng.normal(0, 1, 300), "f2": rng.normal(5, 1, 300)})
    current = base.copy()
    current["f2"] = current["f2"] + 3
    report = detect_drift(base, current)

    assert not report.ok
    assert "f2" in report.psi
    assert any(alert.kind == "feature_psi" and alert.subject == "f2" for alert in report.alerts)


def test_drift_flags_prediction_shift() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"f1": rng.normal(0, 1, 200)})
    base_pred = pd.Series(rng.normal(0, 1, 200))
    cur_pred = pd.Series(rng.normal(2.5, 1, 50))
    report = detect_drift(frame, frame.copy(), baseline_predictions=base_pred, current_predictions=cur_pred)

    assert any(alert.kind == "prediction_shift" for alert in report.alerts)


def test_drift_flags_hit_rate_floor() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"f1": rng.normal(0, 1, 200)})
    report = detect_drift(
        frame, frame.copy(), hit_rate_baseline=0.55, hit_rate_recent=0.41
    )
    assert any(alert.kind == "hit_rate" and alert.severity == "major" for alert in report.alerts)


def test_drift_flags_hit_rate_degradation() -> None:
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"f1": rng.normal(0, 1, 200)})
    report = detect_drift(
        frame, frame.copy(), hit_rate_baseline=0.60, hit_rate_recent=0.48
    )
    assert any(alert.kind == "hit_rate" and alert.severity == "moderate" for alert in report.alerts)


def test_drift_requires_shared_columns() -> None:
    left = pd.DataFrame({"a": np.arange(50, dtype=float)})
    right = pd.DataFrame({"b": np.arange(50, dtype=float)})
    with pytest.raises(ValueError, match="share no feature columns"):
        detect_drift(left, right)


def test_drift_report_serialises() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"f1": rng.normal(0, 1, 200)})
    payload = detect_drift(frame, frame.copy()).as_dict()
    assert payload["ok"] is True
    assert payload["psi"]["f1"] == pytest.approx(0.0, abs=1e-5)


def test_hit_rate_matches_signs() -> None:
    assert hit_rate([1, -1, 1, 1], [0.02, -0.01, -0.03, 0.01]) == pytest.approx(0.75)
    assert hit_rate([1, 1], [1, 1]) == pytest.approx(1.0)
    assert hit_rate([0, 0], [1, -1]) == pytest.approx(0.0)
