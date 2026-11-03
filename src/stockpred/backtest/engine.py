"""Daily-bar matching engine with A-share frictions.

Decisions are taken after the previous close, so an order dated ``T`` fills on
the ``T`` bar's open. Sells are matched before buys within a day (same-day cash
reuse is real), which also makes a same-day round trip fail T+1: the sell is
examined before the buy it would have depended on.

Rejections are recorded, not swallowed: every refused order leaves a
``{date, symbol, side, shares, reason}`` row so a backtest can be audited for
"why did the strategy not trade here".
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from stockpred.backtest.costs import CostModel
from stockpred.backtest.rules import LOT_SIZE, T1Ledger, can_buy, can_sell, round_lot
from stockpred.errors import StockpredError

_SIDES = frozenset({"buy", "sell"})
_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")


@dataclass(frozen=True)
class Order:
    date: date
    symbol: str
    side: str
    shares: int

    def __post_init__(self) -> None:
        if self.side not in _SIDES:
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.shares <= 0:
            raise ValueError(f"shares must be positive, got {self.shares}")
        if self.shares % LOT_SIZE != 0:
            raise ValueError(f"shares must be a multiple of {LOT_SIZE}, got {self.shares}")


@dataclass
class Trade:
    date: date
    symbol: str
    side: str
    shares: int
    price: float
    notional: float
    fees: float
    slippage_cost: float


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[Trade]
    initial_cash: float
    final_equity: float
    rejected: list[dict[str, Any]]


@dataclass(frozen=True)
class _Bar:
    open: float
    high: float
    low: float
    close: float


class _History:
    """Sorted per-symbol bars with the lookups the engine needs."""

    __slots__ = ("_bars", "_days")

    def __init__(self, symbol: str, frame: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"[{symbol}] missing required columns: {missing}")
        by_day: dict[date, _Bar] = {}
        for row in frame.itertuples(index=False):
            by_day[pd.Timestamp(row.date).date()] = _Bar(
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
            )
        self._days: list[date] = sorted(by_day)
        self._bars: list[_Bar] = [by_day[day] for day in self._days]

    @property
    def days(self) -> list[date]:
        return self._days

    def bar_on(self, day: date) -> _Bar | None:
        idx = bisect_left(self._days, day)
        if idx < len(self._days) and self._days[idx] == day:
            return self._bars[idx]
        return None

    def close_before(self, day: date) -> float | None:
        """Close of the last bar strictly before ``day`` (the limit-band reference)."""
        idx = bisect_left(self._days, day)
        return self._bars[idx - 1].close if idx > 0 else None

    def close_on_or_before(self, day: date) -> float | None:
        """Last known close up to ``day``; a suspended day marks to the old price."""
        idx = bisect_right(self._days, day)
        return self._bars[idx - 1].close if idx > 0 else None


def run_backtest(
    bars: dict[str, pd.DataFrame],
    orders: list[Order],
    *,
    initial_cash: float,
    cost_model: CostModel,
    st_symbols: set[str] | None = None,
    suspended: dict[str, set[date]] | None = None,
) -> BacktestResult:
    """Match ``orders`` against ``bars`` day by day and return the equity curve.

    Parameters mirror the tech-lead contract. ``suspended`` is the explicit
    halt calendar; a missing bar for a symbol on a session has the same effect,
    which keeps a data gap from being read as a tradeable session.
    """
    if initial_cash < 0:
        raise ValueError(f"initial_cash must be non-negative, got {initial_cash!r}")

    histories = {symbol: _History(symbol, frame) for symbol, frame in bars.items()}
    st = frozenset(st_symbols or ())
    halted = {symbol: frozenset(days) for symbol, days in (suspended or {}).items()}
    sessions = sorted({day for history in histories.values() for day in history.days})

    orders_by_day: dict[date, list[Order]] = defaultdict(list)
    for order in orders:
        orders_by_day[order.date].append(order)

    cash = float(initial_cash)
    ledgers: dict[str, T1Ledger] = defaultdict(T1Ledger)
    trades: list[Trade] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[date, str, str]] = set()
    rows: list[dict[str, Any]] = []

    for session in sessions:
        todays = orders_by_day.get(session, [])
        for order in [o for o in todays if o.side == "sell"] + [o for o in todays if o.side == "buy"]:
            cash = _match_order(
                order,
                session=session,
                histories=histories,
                halted=halted,
                st=st,
                ledgers=ledgers,
                trades=trades,
                rejected=rejected,
                seen=seen,
                cash=cash,
                cost_model=cost_model,
            )

        market_value = 0.0
        for symbol, ledger in ledgers.items():
            held = ledger.total_shares
            if held == 0:
                continue
            close = histories[symbol].close_on_or_before(session)
            market_value += held * (close if close is not None else 0.0)
        position_shares = sum(ledger.total_shares for ledger in ledgers.values())
        rows.append(
            {
                "date": pd.Timestamp(session),
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "position_shares": position_shares,
            }
        )

    # Orders dated outside the session union never reach the day loop; they are
    # refused rather than dropped, since silence here would look like "no signal".
    trade_days = set(sessions)
    for order_day in sorted(orders_by_day):
        if order_day in trade_days:
            continue
        rejected.extend(_rejection(order, "suspended") for order in orders_by_day[order_day])

    curve = pd.DataFrame(
        rows, columns=["date", "cash", "market_value", "equity", "position_shares"]
    )
    final_equity = float(curve["equity"].iloc[-1]) if len(curve) else float(initial_cash)
    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        initial_cash=float(initial_cash),
        final_equity=final_equity,
        rejected=rejected,
    )


def _match_order(
    order: Order,
    *,
    session: date,
    histories: dict[str, _History],
    halted: dict[str, frozenset[date]],
    st: frozenset[str],
    ledgers: dict[str, T1Ledger],
    trades: list[Trade],
    rejected: list[dict[str, Any]],
    seen: set[tuple[date, str, str]],
    cash: float,
    cost_model: CostModel,
) -> float:
    """Apply one order; returns the new cash balance."""
    key = (order.date, order.symbol, order.side)
    if key in seen:
        # Booked on first sighting even if that attempt was itself refused: the
        # order's identity is (session, symbol, side), not its outcome.
        rejected.append(_rejection(order, "duplicate_order"))
        return cash
    seen.add(key)

    history = histories.get(order.symbol)
    bar = history.bar_on(session) if history is not None else None
    if bar is None or session in halted.get(order.symbol, frozenset()):
        rejected.append(_rejection(order, "suspended"))
        return cash

    prev_close = history.close_before(session) if history is not None else None
    if prev_close is not None:
        gate = can_buy if order.side == "buy" else can_sell
        if not gate(
            prev_close=prev_close,
            open_price=bar.open,
            high=bar.high,
            low=bar.low,
            symbol=order.symbol,
            is_st=order.symbol in st,
        ):
            rejected.append(_rejection(order, "limit_up" if order.side == "buy" else "limit_down"))
            return cash

    ledger = ledgers[order.symbol]
    wanted = round_lot(order.shares)

    if order.side == "buy":
        fill_price = cost_model.buy_fill_price(bar.open)
        fill_shares = min(wanted, _max_affordable_shares(cash, fill_price, cost_model))
        if fill_shares <= 0:
            rejected.append(_rejection(order, "insufficient_cash"))
            return cash
        notional = fill_shares * fill_price
        fees = cost_model.buy_cost(notional)
        cash = cash - notional - fees
        if -1e-9 < cash < 0.0:  # exactly-funded lot: leave no negative residue
            cash = 0.0
        ledger.buy(session, fill_shares)
        trades.append(
            Trade(
                date=session,
                symbol=order.symbol,
                side="buy",
                shares=fill_shares,
                price=fill_price,
                notional=notional,
                fees=fees,
                slippage_cost=(fill_price - bar.open) * fill_shares,
            )
        )
    else:
        fill_shares = min(wanted, ledger.sellable(session))
        if fill_shares <= 0:
            rejected.append(_rejection(order, "t_plus_1"))
            return cash
        fill_price = cost_model.sell_fill_price(bar.open)
        notional = fill_shares * fill_price
        fees = cost_model.sell_cost(notional)
        cash = cash + notional - fees
        ledger.sell(session, fill_shares)
        trades.append(
            Trade(
                date=session,
                symbol=order.symbol,
                side="sell",
                shares=fill_shares,
                price=fill_price,
                notional=notional,
                fees=fees,
                slippage_cost=(bar.open - fill_price) * fill_shares,
            )
        )

    if cash < -1e-9 or ledger.total_shares < 0:  # pragma: no cover - invariant guard
        raise StockpredError(
            f"engine invariant broken on {session} {order.symbol} {order.side}: "
            f"cash={cash!r} shares={ledger.total_shares}"
        )
    return cash


def _max_affordable_shares(cash: float, price: float, cost_model: CostModel) -> int:
    """Largest whole-lot size whose notional plus buy fees still fits in ``cash``.

    Buy fees are the larger of two branch costs -- ``notional*(1+transfer) + 5``
    (commission at its floor) and ``notional*(1+commission+transfer)`` (commission
    above it) -- so the affordable size is bounded by the smaller of the two
    branch solutions. Float noise is then trimmed by walking down whole lots.
    """
    if cash <= 0 or price <= 0:
        return 0
    at_floor = (cash - cost_model.min_commission) / (price * (1.0 + cost_model.transfer_fee_rate))
    above_floor = cash / (price * (1.0 + cost_model.commission_rate + cost_model.transfer_fee_rate))
    upper = min(at_floor, above_floor)
    if upper <= 0:
        return 0
    shares = round_lot(upper)
    while shares > 0:
        notional = shares * price
        if notional + cost_model.buy_cost(notional) <= cash:
            break
        shares -= LOT_SIZE
    return shares


def _rejection(order: Order, reason: str) -> dict[str, Any]:
    return {
        "date": order.date,
        "symbol": order.symbol,
        "side": order.side,
        "shares": order.shares,
        "reason": reason,
    }
