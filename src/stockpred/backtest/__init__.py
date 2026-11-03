"""Rule-aware A-share backtest core: costs, trading rules and a daily matcher."""

from __future__ import annotations

from stockpred.backtest.costs import CostModel
from stockpred.backtest.engine import BacktestResult, Order, Trade, run_backtest
from stockpred.backtest.rules import (
    LOT_SIZE,
    T1Ledger,
    can_buy,
    can_sell,
    limit_prices,
    price_limit_ratio,
    round_lot,
)

__all__ = [
    "LOT_SIZE",
    "BacktestResult",
    "CostModel",
    "Order",
    "T1Ledger",
    "Trade",
    "can_buy",
    "can_sell",
    "limit_prices",
    "price_limit_ratio",
    "round_lot",
    "run_backtest",
]
