"""Dividend / split records, normalised from the vendors into ``CorporateAction``.

akshare's ``stock_history_dividend_detail`` reports per **10 shares** (送股、转增、
派息), baostock reports per **share** — a 10x slip here would corrupt every
adjusted return, so both parsers convert to "per share" in one place and are
tested against a fixture each.

Records are cached as JSON per (symbol, source) because the endpoint is one request
per symbol and the history only changes a few times a year.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from stockpred.data.adjust import CorporateAction
from stockpred.errors import ProviderUnavailable

FORWARD_FILL_DAYS = 0


def parse_akshare_detail(frame: pd.DataFrame) -> list[CorporateAction]:
    """``stock_history_dividend_detail`` -> actions. Columns are per 10 shares."""
    if frame is None or frame.empty:
        return []
    columns = {str(name): name for name in frame.columns}
    ex_col = columns.get("除权除息日")
    cash_col = columns.get("派息(税前)(元)") or columns.get("派息")
    stock_col = columns.get("送股(股)")
    reserve_col = columns.get("转增(股)")
    if ex_col is None:
        return []

    actions: list[CorporateAction] = []
    for _, row in frame.iterrows():
        raw_date = row[ex_col]
        if pd.isna(raw_date) or str(raw_date).strip() in {"", "--"}:
            continue
        cash = _number(row.get(cash_col)) / 10.0 if cash_col else 0.0
        shares = _number(row.get(stock_col)) / 10.0 if stock_col else 0.0
        reserve = _number(row.get(reserve_col)) / 10.0 if reserve_col else 0.0
        if cash <= 0 and shares + reserve <= 0:
            continue
        actions.append(
            CorporateAction(
                ex_date=pd.Timestamp(raw_date).date(),
                cash_per_share=cash,
                share_ratio=shares + reserve,
            )
        )
    return sorted(actions, key=lambda a: a.ex_date)


def parse_baostock_rows(rows: list[dict[str, str]]) -> list[CorporateAction]:
    """``query_dividend_data`` rows -> actions. Columns are already per share."""
    actions: list[CorporateAction] = []
    for row in rows:
        raw_date = row.get("dividOperateDate") or ""
        if not raw_date:
            continue
        cash = _number(row.get("dividCashPsBeforeTax"))
        shares = _number(row.get("dividStocksPs"))
        reserve = _number(row.get("dividReserveToStockPs"))
        if cash <= 0 and shares + reserve <= 0:
            continue
        actions.append(
            CorporateAction(
                ex_date=pd.Timestamp(raw_date).date(),
                cash_per_share=cash,
                share_ratio=shares + reserve,
            )
        )
    return sorted(actions, key=lambda a: a.ex_date)


def _number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "nan", "None"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _cache_path(cache_dir: Path, symbol: str, source: str) -> Path:
    return cache_dir / f"dividends_{symbol}_{source}.json"


def read_cached(cache_dir: Path, symbol: str, source: str) -> list[CorporateAction] | None:
    path = _cache_path(cache_dir, symbol, source)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        CorporateAction(
            ex_date=pd.Timestamp(item["ex_date"]).date(),
            cash_per_share=float(item["cash_per_share"]),
            share_ratio=float(item["share_ratio"]),
        )
        for item in payload
    ]


def write_cache(cache_dir: Path, symbol: str, source: str, actions: list[CorporateAction]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = [{**asdict(action), "ex_date": action.ex_date.isoformat()} for action in actions]
    _cache_path(cache_dir, symbol, source).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_akshare_actions(symbol: str, *, cache_dir: Path | None = None) -> list[CorporateAction]:
    """Dividend history from akshare (one request per symbol), disk-cached."""
    if cache_dir is not None:
        cached = read_cached(cache_dir, symbol, "akshare")
        if cached is not None:
            return cached
    try:
        import akshare as ak  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ProviderUnavailable("akshare is not installed; pip install -e '.[data]'") from exc

    try:
        frame = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
    except Exception as exc:  # noqa: BLE001 - vendor endpoints flake
        raise ProviderUnavailable(f"dividend history failed for {symbol}: {exc}") from exc
    actions = parse_akshare_detail(frame)
    if cache_dir is not None:
        write_cache(cache_dir, symbol, "akshare", actions)
    return actions
