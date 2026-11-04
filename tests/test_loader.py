from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stockpred.data.cache import ParquetCache
from stockpred.data.loader import cache_key, load_daily_bars, read_cached_bars
from stockpred.data.providers.base import MarketDataProvider
from stockpred.errors import DataQualityError


class BadProvider:
    name = "bad"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch_daily(self, symbol: str, start: date, end: date, *, adjust: str = "") -> pd.DataFrame:
        return self._frame

    def trading_calendar(self):
        return []


def test_cached_read_refuses_another_vendors_snapshot(
    tmp_path: Path, fake_provider, bars: pd.DataFrame
) -> None:
    """Cache keys are provider-agnostic, so the manifest source has to be checked."""
    cache = ParquetCache(tmp_path)
    window = (bars["date"].min().date(), bars["date"].max().date())
    load_daily_bars("600519", *window, provider=fake_provider, cache=cache)

    with pytest.raises(DataQualityError, match="not 'baostock'"):
        read_cached_bars(cache, "600519", source="baostock")
    assert len(read_cached_bars(cache, "600519", source="fake").bars) == len(bars)


def test_fake_provider_satisfies_contract(fake_provider) -> None:
    assert isinstance(fake_provider, MarketDataProvider)


def test_load_validates_and_caches(tmp_path: Path, fake_provider, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    result = load_daily_bars(
        "600519",
        bars["date"].min().date(),
        bars["date"].max().date(),
        provider=fake_provider,
        cache=cache,
    )

    assert result.cache_write is not None
    assert result.cache_write.written is True
    assert result.last_bar_date == bars["date"].max().date()
    assert cache.manifest(cache_key("600519", ""))["rows"] == len(bars)


def test_cached_read_returns_same_rows(tmp_path: Path, fake_provider, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    load_daily_bars(
        "600519",
        bars["date"].min().date(),
        bars["date"].max().date(),
        provider=fake_provider,
        cache=cache,
    )

    cached = read_cached_bars(cache, "600519")

    assert len(cached.bars) == len(bars)
    assert cached.cache_write is not None
    assert cached.cache_write.written is False


def test_provider_with_bad_data_is_rejected(tmp_path: Path, bars: pd.DataFrame) -> None:
    broken = bars.copy()
    broken.loc[0, "low"] = broken.loc[0, "high"] + 1.0
    provider = BadProvider(broken)

    with pytest.raises(DataQualityError, match="schema violation"):
        load_daily_bars(
            "600519",
            bars["date"].min().date(),
            bars["date"].max().date(),
            provider=provider,
            cache=ParquetCache(tmp_path),
        )


def test_adjust_variant_uses_separate_cache_key(tmp_path: Path, fake_provider, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    window = (bars["date"].min().date(), bars["date"].max().date())

    raw = load_daily_bars("600519", *window, provider=fake_provider, cache=cache)
    hfq = load_daily_bars("600519", *window, provider=fake_provider, cache=cache, adjust="hfq")

    assert raw.cache_write.path.parent.name == "daily_600519_raw"
    assert hfq.cache_write.path.parent.name == "daily_600519_hfq"
    assert fake_provider.calls[-1][3] == "hfq"
