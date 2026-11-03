"""Data layer: providers, validation, caching, calendar and freshness guards."""

from __future__ import annotations

from stockpred.data.cache import ParquetCache, content_hash
from stockpred.data.calendar import TradingCalendar
from stockpred.data.crosscheck import CrossCheck, compare_bars
from stockpred.data.freshness import FreshnessReport, check_freshness
from stockpred.data.loader import LoadResult, load_daily_bars, read_cached_bars
from stockpred.data.schema import REQUIRED_COLUMNS, validate_bars

__all__ = [
    "REQUIRED_COLUMNS",
    "CrossCheck",
    "FreshnessReport",
    "LoadResult",
    "ParquetCache",
    "TradingCalendar",
    "check_freshness",
    "compare_bars",
    "content_hash",
    "load_daily_bars",
    "read_cached_bars",
    "validate_bars",
]
