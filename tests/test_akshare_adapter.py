from __future__ import annotations

import sys
import types
from datetime import date

import pandas as pd
import pytest

from stockpred.data.providers.akshare_provider import AkshareProvider
from stockpred.errors import ProviderUnavailable


def test_missing_akshare_raises_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "akshare", None)
    with pytest.raises(ProviderUnavailable, match="not installed"):
        AkshareProvider().fetch_daily("600519", date(2024, 1, 2), date(2024, 1, 5))


def test_fetch_daily_normalises_chinese_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    module = types.ModuleType("akshare")

    def stock_zh_a_hist(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "日期": ["2024-01-03", "2024-01-02"],
                "股票代码": ["600519", "600519"],
                "开盘": ["10.5", "10.0"],
                "收盘": ["11.0", "10.4"],
                "最高": ["11.2", "10.6"],
                "最低": ["10.4", "9.9"],
                "成交量": ["120000", "100000"],
                "成交额": ["1320000", "1040000"],
                "换手率": ["0.1", "0.09"],
            }
        )

    module.stock_zh_a_hist = stock_zh_a_hist  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akshare", module)

    frame = AkshareProvider().fetch_daily("600519", date(2024, 1, 2), date(2024, 1, 5))

    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert frame["date"].is_monotonic_increasing
    assert frame["open"].dtype.kind == "f"
    # 成交量 comes in lots (手); canonical volume is shares, so x100 after sorting.
    assert frame["volume"].tolist() == [10_000_000.0, 12_000_000.0]
    assert captured["symbol"] == "600519"
    assert captured["start_date"] == "20240102"
    assert captured["end_date"] == "20240105"
    assert captured["adjust"] == ""


def test_trading_calendar_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("akshare")
    module.tool_trade_date_hist_sina = lambda: pd.DataFrame(  # type: ignore[attr-defined]
        {"trade_date": [date(2024, 1, 2), date(2024, 1, 3)]}
    )
    monkeypatch.setitem(sys.modules, "akshare", module)

    days = AkshareProvider().trading_calendar()

    assert days == [date(2024, 1, 2), date(2024, 1, 3)]


def test_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("akshare")
    module.stock_zh_a_hist = lambda **_: pd.DataFrame()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akshare", module)

    with pytest.raises(ProviderUnavailable, match="no rows"):
        AkshareProvider().fetch_daily("600519", date(2024, 1, 2), date(2024, 1, 5))
