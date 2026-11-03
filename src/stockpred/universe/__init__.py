"""Point-in-time tradable universe.

The only safe way to build a universe is from what was knowable on the decision
day. A universe built from today's instrument list is survivorship-biased: the
delisted names are missing, so the backtest never experiences the losses that
made them disappear.

Therefore every filter here takes an explicit ``as_of`` and reads only rows dated
on or before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

MIN_TURNOVER_YUAN = 5_000_000.0


@dataclass(frozen=True)
class TradabilityRow:
    symbol: str
    date: date
    is_st: bool
    is_suspended: bool
    is_delisted: bool
    amount: float
    close: float


@dataclass(frozen=True)
class UniverseFilter:
    """Rules for exclusion. Defaults target a 100k CNY account."""

    min_amount: float = MIN_TURNOVER_YUAN
    max_price: float | None = 200.0
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_limit_locked: bool = True
    min_history_days: int = 60


@dataclass(frozen=True)
class UniverseDecision:
    symbol: str
    included: bool
    reasons: tuple[str, ...]


def build_universe(
    frame: pd.DataFrame,
    *,
    as_of: date,
    rules: UniverseFilter | None = None,
) -> list[UniverseDecision]:
    """Return one decision per symbol using only rows dated on or before ``as_of``.

    ``frame`` needs columns: symbol, date, close, amount, and optionally
    is_st / is_suspended / is_delisted / limit_locked (booleans, default False).
    """
    contract = UniverseFilter() if rules is None else rules
    required = {"symbol", "date", "close", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"universe frame missing columns: {sorted(missing)}")

    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    history = dated[dated["date"] <= pd.Timestamp(as_of)]
    if history.empty:
        raise ValueError(f"no rows on or before {as_of}")

    decisions: list[UniverseDecision] = []
    for symbol, rows in history.groupby("symbol", sort=True):
        latest = rows.sort_values("date").iloc[-1]
        reasons: list[str] = []

        if _flag(latest, "is_delisted"):
            reasons.append("delisted")
        if contract.exclude_st and _flag(latest, "is_st"):
            reasons.append("st")
        if contract.exclude_suspended and _flag(latest, "is_suspended"):
            reasons.append("suspended")
        if contract.exclude_limit_locked and _flag(latest, "limit_locked"):
            reasons.append("limit_locked")
        if float(latest["amount"]) < contract.min_amount:
            reasons.append("illiquid")
        if contract.max_price is not None and float(latest["close"]) > contract.max_price:
            reasons.append("price_too_high_for_lot")
        if len(rows) < contract.min_history_days:
            reasons.append("short_history")

        decisions.append(
            UniverseDecision(
                symbol=str(symbol),
                included=not reasons,
                reasons=tuple(reasons),
            )
        )
    return decisions


def tradable_symbols(
    frame: pd.DataFrame,
    *,
    as_of: date,
    rules: UniverseFilter | None = None,
) -> list[str]:
    return [
        decision.symbol
        for decision in build_universe(frame, as_of=as_of, rules=rules)
        if decision.included
    ]


def explanation_table(decisions: list[UniverseDecision]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": d.symbol, "included": d.included, "reasons": ",".join(d.reasons)}
            for d in decisions
        ]
    )


def _flag(row: pd.Series, column: str) -> bool:
    if column not in row.index:
        return False
    value = row[column]
    return bool(value) if pd.notna(value) else False
