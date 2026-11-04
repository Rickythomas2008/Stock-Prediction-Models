"""akshare provider (primary source, MIT licensed).

``akshare`` is an optional dependency: importing this module is free, calling a
method without the package installed raises ``ProviderUnavailable``.

Unit note: ``stock_zh_a_hist`` reports 成交量 in lots (手, 1 手 = 100 股) while the
canonical schema wants shares, so volume is scaled by ``_LOT_SIZE`` here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from stockpred.errors import ProviderUnavailable

_LOT_SIZE = 100

_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


class AkshareProvider:
    name = "akshare"

    def __init__(self) -> None:
        self._ak = None

    def _module(self):
        if self._ak is None:
            try:
                import akshare as ak  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ProviderUnavailable(
                    "akshare is not installed; install with: pip install -e '.[data]'"
                ) from exc
            self._ak = ak
        return self._ak

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        raw = self._module().stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        return self._normalize(raw)

    def trading_calendar(self) -> Sequence[date]:
        raw = self._module().tool_trade_date_hist_sina()
        column = "trade_date" if "trade_date" in raw.columns else raw.columns[0]
        return [pd.Timestamp(value).date() for value in raw[column]]

    @staticmethod
    def _normalize(raw: pd.DataFrame, *, scale_lots: bool = True) -> pd.DataFrame:
        if raw is None or raw.empty:
            raise ProviderUnavailable("akshare returned no rows")
        frame = raw.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in raw.columns})
        columns = [
            c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in frame
        ]
        frame = frame[columns].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if scale_lots and "volume" in frame.columns:
            frame["volume"] = frame["volume"] * _LOT_SIZE
        return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


class AkshareIndexProvider:
    """Index bars (e.g. 沪深300 ``000300``) in the same canonical units.

    Uses the Sina endpoint (one request, full history, 成交量 already in shares)
    rather than ``index_zh_a_hist``, which pages 17 sub-requests per call and is
    the first thing to time out under load.
    """

    name = "akshare_index"

    def __init__(self) -> None:
        self._ak = None

    def _module(self):
        if self._ak is None:
            try:
                import akshare as ak  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ProviderUnavailable(
                    "akshare is not installed; install with: pip install -e '.[data]'"
                ) from exc
            self._ak = ak
        return self._ak

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        raw = self._module().stock_zh_index_daily(symbol=_sina_index_code(symbol))
        frame = AkshareProvider._normalize(raw, scale_lots=False)
        mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
        return frame.loc[mask].reset_index(drop=True)

    def trading_calendar(self) -> Sequence[date]:
        return AkshareProvider().trading_calendar()


def _sina_index_code(symbol: str) -> str:
    """``000300`` -> ``sh000300``; ``399006`` -> ``sz399006``; ``sh.000300`` passes through."""
    text = symbol.strip().lower().replace(".", "")
    if text.startswith(("sh", "sz")):
        return text
    if text.startswith("000"):
        return f"sh{text}"
    if text.startswith("399"):
        return f"sz{text}"
    raise ValueError(f"unsupported A-share index code: {symbol!r}")
