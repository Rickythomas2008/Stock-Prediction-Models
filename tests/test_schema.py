from __future__ import annotations

import pandas as pd
import pytest

from stockpred.data.schema import validate_bars
from stockpred.errors import DataQualityError


def test_valid_bars_pass(bars: pd.DataFrame) -> None:
    out = validate_bars(bars, symbol="600519")
    assert len(out) == len(bars)
    assert list(out.columns)[:6] == ["date", "open", "high", "low", "close", "volume"]


def test_negative_price_is_rejected(bars: pd.DataFrame) -> None:
    broken = bars.copy()
    broken.loc[3, "close"] = -1.0
    with pytest.raises(DataQualityError, match="schema violation"):
        validate_bars(broken)


def test_high_below_close_is_rejected(bars: pd.DataFrame) -> None:
    broken = bars.copy()
    broken.loc[2, "high"] = broken.loc[2, "close"] - 0.5
    with pytest.raises(DataQualityError, match="schema violation"):
        validate_bars(broken)


def test_duplicate_dates_are_rejected(bars: pd.DataFrame) -> None:
    broken = pd.concat([bars, bars.iloc[[5]]], ignore_index=True).sort_values("date")
    with pytest.raises(DataQualityError, match="schema violation"):
        validate_bars(broken)


def test_unsorted_dates_are_rejected(bars: pd.DataFrame) -> None:
    broken = bars.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataQualityError, match="schema violation"):
        validate_bars(broken)


def test_missing_column_is_rejected(bars: pd.DataFrame) -> None:
    with pytest.raises(DataQualityError, match="missing required columns"):
        validate_bars(bars.drop(columns=["volume"]))


def test_nan_price_is_rejected(bars: pd.DataFrame) -> None:
    broken = bars.copy()
    broken.loc[1, "open"] = None
    with pytest.raises(DataQualityError, match="schema violation"):
        validate_bars(broken)


def test_empty_frame_is_rejected(bars: pd.DataFrame) -> None:
    with pytest.raises(DataQualityError, match="no rows"):
        validate_bars(bars.iloc[0:0])
