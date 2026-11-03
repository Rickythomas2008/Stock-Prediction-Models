from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from stockpred.data.calendar import TradingCalendar
from stockpred.errors import DataQualityError

MONDAY = date(2024, 1, 8)
HOLIDAY = date(2024, 2, 12)


def test_weekend_is_not_trading_day(calendar: TradingCalendar) -> None:
    assert calendar.is_trading_day(MONDAY) is True
    assert calendar.is_trading_day(MONDAY + timedelta(days=5)) is False


def test_holiday_is_not_trading_day(calendar: TradingCalendar) -> None:
    assert HOLIDAY.weekday() < 5
    assert calendar.is_trading_day(HOLIDAY) is False


def test_last_settled_uses_previous_day_before_close(calendar: TradingCalendar) -> None:
    assert calendar.last_settled_trading_day(datetime(2024, 1, 8, 10, 0)) == date(2024, 1, 5)
    assert calendar.last_settled_trading_day(datetime(2024, 1, 8, 16, 0)) == MONDAY


def test_last_settled_on_saturday_returns_friday(calendar: TradingCalendar) -> None:
    assert calendar.last_settled_trading_day(datetime(2024, 1, 13, 12, 0)) == date(2024, 1, 12)


def test_next_and_previous_skip_weekend(calendar: TradingCalendar) -> None:
    assert calendar.next_trading_day(date(2024, 1, 12)) == date(2024, 1, 15)
    assert calendar.last_trading_day_on_or_before(date(2024, 1, 14)) == date(2024, 1, 12)


def test_trading_days_between_is_inclusive(calendar: TradingCalendar) -> None:
    days = calendar.trading_days_between(date(2024, 1, 8), date(2024, 1, 12))
    assert len(days) == 5
    assert days[0] == MONDAY


def test_empty_calendar_is_rejected() -> None:
    with pytest.raises(DataQualityError, match="empty"):
        TradingCalendar([])


def test_calendar_from_provider(fake_provider) -> None:
    calendar = TradingCalendar.from_provider(fake_provider)
    assert calendar.is_trading_day(date(2024, 1, 8))
