from __future__ import annotations

import pandas as pd
import pytest

from stockpred.data.crosscheck import compare_bars


def _frame(scale: float = 1.0, *, closes: tuple[float, float] = (10.0, 10.5)) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "date": dates,
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": list(closes),
            "volume": [1_000_000.0 * scale, 1_200_000.0 * scale],
            "amount": [1.0e7, 1.26e7],
        }
    )


def test_agreeing_providers_pass() -> None:
    result = compare_bars(_frame(), _frame(), name_left="a", name_right="b")

    assert result.ok
    assert result.sessions == 2


def test_volume_unit_mismatch_fails() -> None:
    # 100x is the lots-vs-shares slip that reached a live run once.
    result = compare_bars(_frame(1.0), _frame(100.0), name_left="a", name_right="b")

    assert not result.ok
    assert any("unit mismatch" in finding for finding in result.findings)


def test_amount_rounding_within_tolerance_passes() -> None:
    right = _frame()
    right.loc[0, "amount"] = right.loc[0, "amount"] + 0.46

    result = compare_bars(_frame(), right, name_left="a", name_right="b")

    assert result.ok


def test_price_difference_beyond_tolerance_fails() -> None:
    result = compare_bars(_frame(), _frame(closes=(10.02, 10.5)), name_left="a", name_right="b")

    assert not result.ok
    assert any(finding.startswith("close") and "FAIL" in finding for finding in result.findings)


def test_no_overlap_fails() -> None:
    later = _frame()
    later["date"] = pd.to_datetime(["2024-02-02", "2024-02-05"])

    result = compare_bars(_frame(), later, name_left="a", name_right="b")

    assert not result.ok
    assert result.sessions == 0


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "amount", "volume"])
def test_each_compared_column_is_reported(column: str) -> None:
    result = compare_bars(_frame(), _frame(), name_left="a", name_right="b")

    assert any(finding.startswith(column) for finding in result.findings)
