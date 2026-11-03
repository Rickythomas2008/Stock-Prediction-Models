"""Trading calendar.

The calendar is never guessed. If the provider cannot supply it we fail loudly,
because "assume weekdays are trading days" is exactly how a holiday becomes a
stale-data incident.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time

from stockpred.errors import DataQualityError, ProviderUnavailable

DEFAULT_MARKET_CLOSE = time(15, 0)


class TradingCalendar:
    def __init__(self, days: Iterable[date]) -> None:
        ordered = sorted({_as_date(d) for d in days})
        if not ordered:
            raise DataQualityError("trading calendar is empty")
        self._days: tuple[date, ...] = tuple(ordered)

    @classmethod
    def from_provider(cls, provider: object) -> TradingCalendar:
        fetch = getattr(provider, "trading_calendar", None)
        if fetch is None:
            raise ProviderUnavailable("provider does not expose trading_calendar()")
        return cls(fetch())

    @property
    def days(self) -> Sequence[date]:
        return self._days

    def __len__(self) -> int:
        return len(self._days)

    def __contains__(self, day: object) -> bool:
        try:
            return _as_date(day) in self._index()
        except (TypeError, ValueError):
            return False

    def is_trading_day(self, day: date | datetime) -> bool:
        return _as_date(day) in self._index()

    def last_trading_day_on_or_before(self, day: date | datetime) -> date:
        target = _as_date(day)
        pos = bisect.bisect_right(self._days, target)
        if pos == 0:
            raise DataQualityError(f"no trading day on or before {target}")
        return self._days[pos - 1]

    def next_trading_day(self, day: date | datetime) -> date:
        target = _as_date(day)
        pos = bisect.bisect_right(self._days, target)
        if pos >= len(self._days):
            raise DataQualityError(f"no trading day after {target}")
        return self._days[pos]

    def trading_days_between(self, start: date | datetime, end: date | datetime) -> list[date]:
        lo, hi = sorted((_as_date(start), _as_date(end)))
        left = bisect.bisect_left(self._days, lo)
        right = bisect.bisect_right(self._days, hi)
        return list(self._days[left:right])

    def last_settled_trading_day(
        self,
        as_of: datetime,
        *,
        market_close: time = DEFAULT_MARKET_CLOSE,
    ) -> date:
        """Latest trading day whose bar should already be final at ``as_of``."""
        day = as_of.date() if isinstance(as_of, datetime) else _as_date(as_of)
        if isinstance(as_of, datetime) and as_of.time() < market_close:
            day = _previous_day(day)
        return self.last_trading_day_on_or_before(day)

    def _index(self) -> frozenset[date]:
        cached = getattr(self, "_index_cache", None)
        if cached is None:
            cached = frozenset(self._days)
            object.__setattr__(self, "_index_cache", cached)
        return cached


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"not a date: {value!r}")


def _previous_day(day: date) -> date:
    from datetime import timedelta

    return day - timedelta(days=1)
