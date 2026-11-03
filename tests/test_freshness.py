from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from stockpred.data.calendar import TradingCalendar
from stockpred.data.freshness import check_freshness
from stockpred.errors import DataQualityError, LookaheadError, StaleDataError

MONDAY = date(2024, 1, 8)
FRIDAY = date(2024, 1, 12)
HOLIDAY = date(2024, 2, 12)


def test_fresh_bar_passes(calendar: TradingCalendar) -> None:
    report = check_freshness(MONDAY, as_of=datetime(2024, 1, 8, 16, 0), calendar=calendar)
    assert report.ok
    assert report.stale_trading_days == 0
    assert report.expected_date == MONDAY


def test_stale_bar_is_rejected(calendar: TradingCalendar) -> None:
    with pytest.raises(StaleDataError, match="stale data"):
        check_freshness(FRIDAY, as_of=datetime(2024, 1, 15, 16, 0), calendar=calendar)


def test_stale_bar_allowed_within_budget(calendar: TradingCalendar) -> None:
    report = check_freshness(
        FRIDAY,
        as_of=datetime(2024, 1, 15, 16, 0),
        calendar=calendar,
        max_stale_trading_days=1,
    )
    assert report.stale_trading_days == 1
    assert report.ok


def test_multi_day_gap_counts_trading_days(calendar: TradingCalendar) -> None:
    with pytest.raises(StaleDataError) as excinfo:
        check_freshness(date(2024, 1, 4), as_of=datetime(2024, 1, 9, 16, 0), calendar=calendar)
    assert "3 trading days behind" in str(excinfo.value)


def test_future_bar_is_rejected(calendar: TradingCalendar) -> None:
    with pytest.raises(LookaheadError, match="ahead of settled trading day"):
        check_freshness(date(2024, 1, 9), as_of=datetime(2024, 1, 8, 16, 0), calendar=calendar)


def test_today_bar_before_close_is_lookahead(calendar: TradingCalendar) -> None:
    with pytest.raises(LookaheadError):
        check_freshness(MONDAY, as_of=datetime(2024, 1, 8, 10, 0), calendar=calendar)


def test_non_trading_day_bar_is_rejected(calendar: TradingCalendar) -> None:
    with pytest.raises(DataQualityError, match="not a trading day"):
        check_freshness(HOLIDAY, as_of=datetime(2024, 2, 12, 16, 0), calendar=calendar)


def test_weekend_as_of_uses_friday(calendar: TradingCalendar) -> None:
    report = check_freshness(
        FRIDAY, as_of=datetime(2024, 1, 13, 12, 0), calendar=calendar
    )
    assert report.ok
    assert report.expected_date == FRIDAY


def test_datetime_bar_date_is_accepted(calendar: TradingCalendar) -> None:
    report = check_freshness(
        datetime(2024, 1, 8, 15, 0), as_of=datetime(2024, 1, 8, 16, 0), calendar=calendar
    )
    assert report.bar_date == MONDAY
    assert report.ok is True
    assert report.max_stale_trading_days == 0
    assert isinstance(MONDAY + timedelta(days=0), date)
