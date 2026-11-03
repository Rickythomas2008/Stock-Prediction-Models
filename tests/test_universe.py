from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stockpred.universe import (
    UniverseFilter,
    build_universe,
    explanation_table,
    tradable_symbols,
)

AS_OF = date(2024, 3, 10)


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("600519", False, False, 9e7, 1700.0),
        ("000001", False, False, 5e8, 12.0),
        ("300750", True, False, 3e7, 180.0),
        ("600000", False, True, 1e7, 8.0),
        ("000002", False, False, 1e6, 9.0),
        ("688981", False, False, 6e7, 55.0),
    ]
    for symbol, is_st, suspended, amount, price in specs:
        for day in pd.date_range("2023-06-01", periods=200, freq="B"):
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "close": price,
                    "amount": amount,
                    "is_st": is_st,
                    "is_suspended": suspended,
                }
            )
    return pd.DataFrame(rows)


def test_only_tradable_symbols_kept(frame: pd.DataFrame) -> None:
    assert tradable_symbols(frame, as_of=AS_OF) == ["000001", "688981"]


def test_each_exclusion_has_a_reason(frame: pd.DataFrame) -> None:
    decisions = {d.symbol: d for d in build_universe(frame, as_of=AS_OF)}

    assert "st" in decisions["300750"].reasons
    assert "suspended" in decisions["600000"].reasons
    assert "illiquid" in decisions["000002"].reasons
    assert "price_too_high_for_lot" in decisions["600519"].reasons


def test_price_cap_is_configurable(frame: pd.DataFrame) -> None:
    rules = UniverseFilter(max_price=None, exclude_st=False, exclude_suspended=False)
    included = tradable_symbols(frame, as_of=AS_OF, rules=rules)
    assert "600519" in included
    assert "300750" in included


def test_short_history_is_excluded(frame: pd.DataFrame) -> None:
    rules = UniverseFilter(min_history_days=500)
    assert tradable_symbols(frame, as_of=AS_OF, rules=rules) == []


def test_delisted_flag_excludes(frame: pd.DataFrame) -> None:
    marked = frame.copy()
    marked.loc[marked["symbol"] == "000001", "is_delisted"] = True
    assert "000001" not in tradable_symbols(marked, as_of=AS_OF)


def test_universe_ignores_rows_after_as_of(frame: pd.DataFrame) -> None:
    """A symbol that only appears later must not be selectable today."""
    future = frame[frame["symbol"] == "000001"].copy()
    future["symbol"] = "NEWIPO"
    future["date"] = pd.to_datetime(future["date"]) + pd.Timedelta(days=400)
    combined = pd.concat([frame, future], ignore_index=True)

    assert "NEWIPO" not in tradable_symbols(combined, as_of=AS_OF)
    assert "NEWIPO" in tradable_symbols(combined, as_of=date(2025, 8, 1))


def test_delisting_after_as_of_does_not_leak(frame: pd.DataFrame) -> None:
    """The sample must not 'know' about a delisting that happens later."""
    marked = frame.copy()
    late = pd.to_datetime(marked["date"]) > pd.Timestamp(AS_OF)
    marked.loc[(marked["symbol"] == "000001") & late, "is_delisted"] = True

    assert "000001" in tradable_symbols(marked, as_of=AS_OF)


def test_missing_columns_raise(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_universe(frame.drop(columns=["amount"]), as_of=AS_OF)


def test_no_history_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows on or before"):
        build_universe(frame, as_of=date(2020, 1, 1))


def test_explanation_table_shape(frame: pd.DataFrame) -> None:
    table = explanation_table(build_universe(frame, as_of=AS_OF))
    assert list(table.columns) == ["symbol", "included", "reasons"]
    assert table["included"].sum() == 2
