"""Bar loading: fetch -> validate -> cache.

The loader is the only place that turns a provider response into something the
rest of the pipeline may use, so validation cannot be skipped by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stockpred.data.cache import CacheWrite, ParquetCache
from stockpred.data.providers.base import MarketDataProvider
from stockpred.data.schema import validate_bars


@dataclass(frozen=True)
class LoadResult:
    symbol: str
    adjust: str
    bars: pd.DataFrame
    cache_write: CacheWrite | None = None

    @property
    def last_bar_date(self) -> date:
        return pd.Timestamp(self.bars["date"].max()).date()


def cache_key(symbol: str, adjust: str) -> str:
    return f"daily_{symbol}_{adjust or 'raw'}"


def load_daily_bars(
    symbol: str,
    start: date,
    end: date,
    *,
    provider: MarketDataProvider,
    cache: ParquetCache | None = None,
    adjust: str = "",
) -> LoadResult:
    """Fetch daily bars, validate them against the schema, and cache a snapshot."""
    raw = provider.fetch_daily(symbol, start, end, adjust=adjust)
    bars = validate_bars(raw, symbol=symbol)
    write = None
    if cache is not None:
        write = cache.store(cache_key(symbol, adjust), bars, source=provider.name)
    return LoadResult(symbol=symbol, adjust=adjust, bars=bars, cache_write=write)


def read_cached_bars(cache: ParquetCache, symbol: str, adjust: str = "") -> LoadResult:
    bars, manifest = cache.load(cache_key(symbol, adjust))
    return LoadResult(
        symbol=symbol,
        adjust=adjust,
        bars=validate_bars(bars, symbol=symbol),
        cache_write=CacheWrite(
            path=cache.root / cache_key(symbol, adjust) / manifest["file"],
            content_hash=manifest["content_hash"],
            written=False,
            rows=int(manifest["rows"]),
        ),
    )
