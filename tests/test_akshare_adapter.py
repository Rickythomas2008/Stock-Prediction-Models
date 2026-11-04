from __future__ import annotations

import sys
import types
from datetime import date

import pandas as pd
import pytest

from stockpred.data.loader import cache_key
from stockpred.data.providers import to_baostock_code
from stockpred.data.providers.akshare_provider import AkshareIndexProvider, AkshareProvider
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


def test_index_provider_uses_one_request_and_keeps_share_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Sina endpoint already reports shares: scaling it by 100 would be wrong."""
    captured: dict[str, object] = {}
    module = types.ModuleType("akshare")

    def stock_zh_index_daily(symbol: str) -> pd.DataFrame:
        captured["symbol"] = symbol
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "open": [3000.0, 3010.0, 3020.0],
                "high": [3010.0, 3020.0, 3030.0],
                "low": [2990.0, 3000.0, 3010.0],
                "close": [3005.0, 3015.0, 3025.0],
                "volume": [24_000_000_000.0, 24_500_000_000.0, 25_000_000_000.0],
            }
        )

    module.stock_zh_index_daily = stock_zh_index_daily  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akshare", module)

    frame = AkshareIndexProvider().fetch_daily("000300", date(2024, 1, 3), date(2024, 1, 4))

    assert captured["symbol"] == "sh000300"
    assert frame["volume"].tolist() == [24_500_000_000.0, 25_000_000_000.0]
    assert frame["date"].iloc[0].date() == date(2024, 1, 3)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("000300", "sh.000300"), ("399006", "sz.399006"), ("sh.000905", "sh.000905")],
)
def test_baostock_index_code_mapping(symbol: str, expected: str) -> None:
    assert to_baostock_code(symbol, index=True) == expected


def test_baostock_index_namespace_differs_from_stock_namespace() -> None:
    """``000300`` is a Shenzhen stock code but a Shanghai index code."""
    assert to_baostock_code("000300") == "sz.000300"
    assert to_baostock_code("000300", index=True) == "sh.000300"
    with pytest.raises(ValueError, match="index code"):
        to_baostock_code("600519", index=True)


def test_cache_key_is_provider_agnostic() -> None:
    """The key stays stable across vendors; the manifest source carries provenance."""
    assert cache_key("600519", "qfq") == "daily_600519_qfq"
