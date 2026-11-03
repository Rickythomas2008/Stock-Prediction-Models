"""baostock provider (secondary/cross-check source).

License note: baostock's license has not been independently verified yet
(see ``tools/license_allowlist.toml``). Keep it in the optional ``data`` extra
and confirm before shipping a distribution that bundles it.

Unit note: baostock reports volume in shares (股), which already matches the
canonical schema, so no conversion is applied here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from stockpred.errors import ProviderUnavailable

_FIELDS = "date,open,high,low,close,volume,amount"


def to_baostock_code(symbol: str) -> str:
    """Map ``600519`` / ``600519.SH`` / ``sh.600519`` to baostock's ``sh.600519``."""
    text = symbol.strip().lower()
    if "." in text:
        head, tail = text.split(".", 1)
        if head in {"sh", "sz"}:
            return f"{head}.{tail}"
        if tail in {"sh", "sz"}:
            return f"{tail}.{head}"
        raise ValueError(f"cannot map symbol to a baostock code: {symbol!r}")
    if text.startswith("6"):
        return f"sh.{text}"
    if text.startswith(("0", "3")):
        return f"sz.{text}"
    raise ValueError(f"unsupported A-share symbol: {symbol!r}")


class BaostockProvider:
    name = "baostock"

    def __init__(self) -> None:
        self._bs = None

    def _module(self):
        if self._bs is None:
            try:
                import baostock as bs  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ProviderUnavailable(
                    "baostock is not installed; install with: pip install -e '.[data]'"
                ) from exc
            self._bs = bs
        return self._bs

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        bs = self._module()
        login = bs.login()
        if getattr(login, "error_code", "0") != "0":
            raise ProviderUnavailable(f"baostock login failed: {getattr(login, 'error_msg', '')}")
        try:
            result = bs.query_history_k_data_plus(
                to_baostock_code(symbol),
                _FIELDS,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                frequency="d",
                adjustflag=_adjust_flag(adjust),
            )
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
        finally:
            bs.logout()
        if not rows:
            raise ProviderUnavailable(f"baostock returned no rows for {symbol}")
        frame = pd.DataFrame(rows, columns=_FIELDS.split(","))
        frame["date"] = pd.to_datetime(frame["date"])
        for col in ("open", "high", "low", "close", "volume", "amount"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.sort_values("date", kind="mergesort").reset_index(drop=True)

    def trading_calendar(self) -> Sequence[date]:
        bs = self._module()
        login = bs.login()
        if getattr(login, "error_code", "0") != "0":
            raise ProviderUnavailable("baostock login failed")
        try:
            result = bs.query_trade_dates(start_date="1990-01-01", end_date="2035-12-31")
            days = []
            while result.error_code == "0" and result.next():
                row = dict(zip(result.fields, result.get_row_data(), strict=False))
                if str(row.get("is_trading_day", "0")) == "1":
                    days.append(pd.Timestamp(row["calendar_date"]).date())
            return days
        finally:
            bs.logout()


def _adjust_flag(adjust: str) -> str:
    """baostock: 1=后复权, 2=前复权, 3=不复权."""
    return {"": "3", "none": "3", "hfq": "1", "qfq": "2"}.get(adjust.lower() if adjust else "", "3")
