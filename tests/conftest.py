from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from stockpred.data.calendar import TradingCalendar

HOLIDAY = date(2024, 2, 12)


@pytest.fixture(scope="session")
def trading_days() -> list[date]:
    days: list[date] = []
    cursor = date(2024, 1, 1)
    while cursor <= date(2024, 3, 29):
        if cursor.weekday() < 5 and cursor != HOLIDAY:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


@pytest.fixture
def calendar(trading_days: list[date]) -> TradingCalendar:
    return TradingCalendar(trading_days)


@pytest.fixture
def bars(trading_days: list[date]) -> pd.DataFrame:
    rows = []
    price = 10.0
    for i, day in enumerate(trading_days[:20]):
        price = round(price * (1 + (0.01 if i % 3 else -0.005)), 4)
        rows.append(
            {
                "date": pd.Timestamp(day),
                "open": price,
                "high": round(price * 1.01, 4),
                "low": round(price * 0.99, 4),
                "close": price,
                "volume": 1_000_000.0 + i,
                "amount": (1_000_000.0 + i) * price,
            }
        )
    return pd.DataFrame(rows)


class FakeProvider:
    """Deterministic in-memory provider; satisfies ``MarketDataProvider``."""

    name = "fake"

    def __init__(self, frame: pd.DataFrame, days: Sequence[date]) -> None:
        self._frame = frame
        self._days = list(days)
        self.calls: list[tuple[str, date, date, str]] = []

    def fetch_daily(
        self, symbol: str, start: date, end: date, *, adjust: str = ""
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end, adjust))
        mask = (self._frame["date"] >= pd.Timestamp(start)) & (self._frame["date"] <= pd.Timestamp(end))
        return self._frame.loc[mask].reset_index(drop=True)

    def trading_calendar(self) -> Sequence[date]:
        return list(self._days)


@pytest.fixture
def fake_provider(bars: pd.DataFrame, trading_days: list[date]) -> FakeProvider:
    return FakeProvider(bars, trading_days)
