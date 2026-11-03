"""Paper broker: fill orders against real bars, using A-share rules and costs.

Purpose is narrow and deliberate: validate the *execution path* (order lifecycle,
lot sizing, rejection reasons, cash accounting) without touching money. It is not
a return simulator -- the backtest engine already does that -- so the only things
it models are the frictions that decide whether an order happens at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from stockpred.backtest.costs import CostModel
from stockpred.backtest.rules import LOT_SIZE, T1Ledger, can_buy, can_sell
from stockpred.errors import DataQualityError


@dataclass(frozen=True)
class Fill:
    date: date
    symbol: str
    side: str
    shares: int
    price: float
    notional: float
    fees: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "shares": self.shares,
            "price": round(self.price, 4),
            "notional": round(self.notional, 2),
            "fees": round(self.fees, 4),
        }


@dataclass(frozen=True)
class Rejection:
    date: date
    symbol: str
    side: str
    shares: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "shares": self.shares,
            "reason": self.reason,
        }


@dataclass
class PaperBroker:
    bars: dict[str, pd.DataFrame]
    cash: float = 100_000.0
    cost_model: CostModel = field(default_factory=CostModel)
    st_symbols: frozenset[str] = frozenset()
    suspended: dict[str, frozenset[date]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ledgers: dict[str, T1Ledger] = {}
        self._positions: dict[str, int] = {}
        self.fills: list[Fill] = []
        self.rejections: list[Rejection] = []

    # ------------------------------------------------------------------ queries
    def position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    def positions(self) -> dict[str, int]:
        return {symbol: shares for symbol, shares in self._positions.items() if shares}

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(shares * prices.get(symbol, 0.0) for symbol, shares in self.positions().items())

    # ------------------------------------------------------------------- orders
    def submit(self, *, day: date, symbol: str, side: str, shares: int, limit_price: float) -> Fill | Rejection:
        """Submit one limit order for ``day``; returns the fill or the refusal."""
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if shares <= 0 or shares % LOT_SIZE != 0:
            raise ValueError(f"shares must be a positive multiple of {LOT_SIZE}, got {shares}")

        bar = self._bar(symbol, day)
        if bar is None or day in self.suspended.get(symbol, frozenset()):
            return self._reject(day, symbol, side, shares, "suspended")

        prev_close = self._prev_close(symbol, day)
        if prev_close is not None:
            gate = can_buy if side == "buy" else can_sell
            allowed = gate(
                prev_close=prev_close,
                open_price=bar["open"],
                high=bar["high"],
                low=bar["low"],
                symbol=symbol,
                is_st=symbol in self.st_symbols,
            )
            if not allowed:
                reason = "limit_up" if side == "buy" else "limit_down"
                return self._reject(day, symbol, side, shares, reason)

        if side == "buy":
            return self._fill_buy(day, symbol, shares, bar["open"], limit_price)
        return self._fill_sell(day, symbol, shares, bar["open"], limit_price)

    # ------------------------------------------------------------------ internals
    def _fill_buy(self, day: date, symbol: str, shares: int, open_price: float, limit: float) -> Fill | Rejection:
        fill_price = self.cost_model.buy_fill_price(open_price)
        if fill_price > limit:
            return self._reject(day, symbol, "buy", shares, "limit_price_not_reached")
        affordable = self._affordable(day, symbol, fill_price, shares)
        if affordable <= 0:
            return self._reject(day, symbol, "buy", shares, "insufficient_cash")
        notional = affordable * fill_price
        fees = self.cost_model.buy_cost(notional)
        self.cash -= notional + fees
        self._ledger(symbol).buy(day, affordable)
        self._positions[symbol] = self.position(symbol) + affordable
        fill = Fill(day, symbol, "buy", affordable, fill_price, notional, fees)
        self.fills.append(fill)
        return fill

    def _fill_sell(self, day: date, symbol: str, shares: int, open_price: float, limit: float) -> Fill | Rejection:
        fill_price = self.cost_model.sell_fill_price(open_price)
        if fill_price < limit:
            return self._reject(day, symbol, "sell", shares, "limit_price_not_reached")
        sellable = self._ledger(symbol).sellable(day)
        quantity = min(shares, sellable)
        if quantity <= 0:
            return self._reject(day, symbol, "sell", shares, "t_plus_1")
        notional = quantity * fill_price
        fees = self.cost_model.sell_cost(notional)
        self.cash += notional - fees
        self._ledger(symbol).sell(day, quantity)
        self._positions[symbol] = self.position(symbol) - quantity
        fill = Fill(day, symbol, "sell", quantity, fill_price, notional, fees)
        self.fills.append(fill)
        return fill

    def _affordable(self, day: date, symbol: str, price: float, wanted: int) -> int:
        lots = wanted // LOT_SIZE
        while lots > 0:
            shares = lots * LOT_SIZE
            notional = shares * price
            if notional + self.cost_model.buy_cost(notional) <= self.cash + 1e-9:
                return shares
            lots -= 1
        return 0

    def _ledger(self, symbol: str) -> T1Ledger:
        if symbol not in self._ledgers:
            self._ledgers[symbol] = T1Ledger()
        return self._ledgers[symbol]

    def _reject(self, day: date, symbol: str, side: str, shares: int, reason: str) -> Rejection:
        rejection = Rejection(day, symbol, side, shares, reason)
        self.rejections.append(rejection)
        return rejection

    def _bar(self, symbol: str, day: date) -> dict[str, float] | None:
        frame = self.bars.get(symbol)
        if frame is None:
            return None
        dates = pd.to_datetime(frame["date"])
        match = frame.loc[dates == pd.Timestamp(day)]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

    def _prev_close(self, symbol: str, day: date) -> float | None:
        frame = self.bars.get(symbol)
        if frame is None:
            return None
        dates = pd.to_datetime(frame["date"])
        earlier = frame.loc[dates < pd.Timestamp(day)].sort_values("date")
        if earlier.empty:
            return None
        return float(earlier["close"].iloc[-1])


def replay_sheet(
    broker: PaperBroker,
    *,
    day: date,
    orders: list[tuple[str, str, int, float]],
) -> tuple[list[Fill], list[Rejection]]:
    """Replay ``(symbol, side, shares, limit)`` orders in sell-then-buy order."""
    fills: list[Fill] = []
    rejections: list[Rejection] = []
    ordered = [order for order in orders if order[1] == "sell"] + [
        order for order in orders if order[1] == "buy"
    ]
    for symbol, side, shares, limit in ordered:
        result = broker.submit(day=day, symbol=symbol, side=side, shares=shares, limit_price=limit)
        if isinstance(result, Fill):
            fills.append(result)
        else:
            rejections.append(result)
    if not broker.bars:
        raise DataQualityError("paper broker has no bars to fill against")
    return fills, rejections
