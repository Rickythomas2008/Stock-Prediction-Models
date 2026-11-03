"""Freshness and lookahead guards.

Two failure modes kill live decisions: trading on a bar that is older than it
should be, and trading on a bar that has not happened yet. Both raise instead
of returning a "probably fine" flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

from stockpred.data.calendar import DEFAULT_MARKET_CLOSE, TradingCalendar
from stockpred.errors import DataQualityError, LookaheadError, StaleDataError


@dataclass(frozen=True)
class FreshnessReport:
    bar_date: date
    expected_date: date
    stale_trading_days: int
    max_stale_trading_days: int

    @property
    def ok(self) -> bool:
        return self.stale_trading_days <= self.max_stale_trading_days


def check_freshness(
    bar_date: date | datetime,
    *,
    as_of: datetime,
    calendar: TradingCalendar,
    market_close: time = DEFAULT_MARKET_CLOSE,
    max_stale_trading_days: int = 0,
) -> FreshnessReport:
    """Validate that ``bar_date`` is the freshest settled trading day."""
    bar_day = bar_date.date() if isinstance(bar_date, datetime) else bar_date
    if bar_day not in calendar:
        raise DataQualityError(f"bar date {bar_day} is not a trading day")

    expected = calendar.last_settled_trading_day(as_of, market_close=market_close)

    if bar_day > expected:
        raise LookaheadError(
            f"bar dated {bar_day} is ahead of settled trading day {expected} at {as_of.isoformat()}"
        )

    stale_days = len(calendar.trading_days_between(bar_day, expected)) - 1
    report = FreshnessReport(
        bar_date=bar_day,
        expected_date=expected,
        stale_trading_days=max(stale_days, 0),
        max_stale_trading_days=max_stale_trading_days,
    )
    if not report.ok:
        raise StaleDataError(
            f"stale data: latest bar {bar_day}, expected {expected} "
            f"({report.stale_trading_days} trading days behind, limit {max_stale_trading_days})"
        )
    return report
