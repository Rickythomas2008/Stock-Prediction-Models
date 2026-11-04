"""Acceptance tests for the rule-aware A-share backtest core.

Each case targets a single rule that a naive backtest gets wrong, so a failure
names the broken rule rather than "the equity number moved".
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stockpred.backtest import (
    LOT_SIZE,
    CostModel,
    Order,
    T1Ledger,
    can_buy,
    can_sell,
    limit_prices,
    price_limit_ratio,
    round_lot,
    run_backtest,
)

SYMBOL = "600519"
START = date(2024, 3, 4)


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _bars(
    days: list[date], opens: list[float], closes: list[float] | None = None
) -> pd.DataFrame:
    """Deterministic bars: open/close given, high/low bracket them by +-1%."""
    resolved = list(opens) if closes is None else list(closes)
    rows = []
    for day, open_price, close in zip(days, opens, resolved, strict=True):
        rows.append(
            {
                "date": pd.Timestamp(day),
                "open": open_price,
                "high": round(max(open_price, close) * 1.01, 4),
                "low": round(min(open_price, close) * 0.99, 4),
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def _scenario(opens: list[float], closes: list[float] | None = None) -> tuple[list[date], dict]:
    days = _weekdays(START, len(opens))
    return days, {SYMBOL: _bars(days, opens, closes)}


# --------------------------------------------------------------------------- costs


def test_buy_cost_one_wan() -> None:
    model = CostModel()
    assert model.buy_cost(10_000.0) == pytest.approx(5.1)  # 5 佣金 + 0.1 过户费
    assert model.sell_cost(10_000.0) == pytest.approx(10.1)  # + 5 印花税


def test_min_commission_dominates_small_trade() -> None:
    model = CostModel()
    assert model.buy_cost(1_000.0) == pytest.approx(5.01)  # 0.25 佣金被抬到 5


@pytest.mark.parametrize("notional", [0.0, 1.0, 500.0, 10_000.0, 1_000_000.0])
def test_costs_are_non_negative(notional: float) -> None:
    model = CostModel()
    assert model.buy_cost(notional) >= 0.0
    assert model.sell_cost(notional) >= 0.0
    assert model.sell_cost(notional) >= model.buy_cost(notional)


# ---------------------------------------------------------------------------- rules


def test_price_limits_by_board() -> None:
    assert price_limit_ratio("600519") == 0.10
    assert price_limit_ratio("300750") == 0.20
    assert price_limit_ratio("301001") == 0.20
    assert price_limit_ratio("688981") == 0.20
    assert price_limit_ratio("689009") == 0.20
    assert price_limit_ratio("830799") == 0.20  # 北交所 8
    assert price_limit_ratio("430047") == 0.20  # 北交所 4
    assert price_limit_ratio("300750", is_st=True) == 0.05
    assert limit_prices(10.0, "600519") == (9.0, 11.0)
    assert limit_prices(10.0, "300750") == (8.0, 12.0)
    assert limit_prices(10.0, "600519", is_st=True) == (9.5, 10.5)


def test_limit_up_blocks_buy() -> None:
    assert (
        can_buy(prev_close=10.0, open_price=11.0, high=11.0, low=11.0, symbol="600519") is False
    )
    assert can_buy(prev_close=10.0, open_price=10.5, high=11.1, low=10.4, symbol="600519") is True
    assert (
        can_buy(
            prev_close=10.0,
            open_price=10.5,
            high=11.1,
            low=10.4,
            symbol="600519",
            suspended=True,
        )
        is False
    )


def test_limit_down_blocks_sell() -> None:
    assert (
        can_sell(prev_close=10.0, open_price=9.0, high=9.0, low=9.0, symbol="600519") is False
    )
    assert can_sell(prev_close=10.0, open_price=9.5, high=9.6, low=9.2, symbol="600519") is True


def test_round_lot() -> None:
    assert round_lot(150) == 100
    assert round_lot(99) == 0
    assert round_lot(100) == 100
    assert round_lot(1_000) == 1_000
    assert round_lot(299.9) == 200
    assert round_lot(0) == 0


def test_t1_ledger_blocks_same_day_sell() -> None:
    day0, day1, day2 = date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)
    ledger = T1Ledger()
    ledger.buy(day0, 300)

    assert ledger.sellable(day0) == 0
    assert ledger.sellable(day1) == 300

    with pytest.raises(ValueError):
        ledger.sell(day0, 100)

    ledger.sell(day1, 200)
    assert ledger.total_shares == 100
    assert ledger.sellable(day2) == 100
    with pytest.raises(ValueError):
        ledger.sell(day2, 200)


def test_t1_ledger_consumes_fifo() -> None:
    day0, day3 = date(2024, 3, 4), date(2024, 3, 7)
    ledger = T1Ledger()
    ledger.buy(day0, 100)
    ledger.buy(day3, 100)

    ledger.sell(day3, 100)  # only the day-0 lot is sellable here, and it goes first
    assert ledger.total_shares == 100
    assert ledger.sellable(day3) == 0  # the remainder is the day-3 lot
    assert ledger.sellable(day3 + timedelta(days=1)) == 100


# --------------------------------------------------------------------------- engine


def test_engine_buys_on_next_open() -> None:
    days, bars = _scenario([10.0, 10.5, 11.0], [10.2, 10.7, 11.1])
    model = CostModel(slippage_bps=10.0)
    result = run_backtest(
        bars,
        [Order(days[1], SYMBOL, "buy", 200)],
        initial_cash=100_000.0,
        cost_model=model,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == pytest.approx(10.5 * (1 + 10.0 / 10_000.0))
    assert trade.notional == pytest.approx(200 * trade.price)
    assert trade.fees == pytest.approx(model.buy_cost(trade.notional))

    curve = result.equity_curve
    assert curve.loc[0, "cash"] == pytest.approx(100_000.0)  # nothing happened on the decision day
    assert curve.loc[1, "cash"] == pytest.approx(100_000.0 - trade.notional - trade.fees)
    assert curve.loc[1, "position_shares"] == 200
    assert result.rejected == []


def test_engine_t1_rejection() -> None:
    days, bars = _scenario([10.0, 10.4, 10.8])
    result = run_backtest(
        bars,
        [Order(days[0], SYMBOL, "buy", 100), Order(days[0], SYMBOL, "sell", 100)],
        initial_cash=100_000.0,
        cost_model=CostModel(),
    )

    assert [t.side for t in result.trades] == ["buy"]
    assert result.rejected[0]["reason"] == "t_plus_1"
    assert result.rejected[0]["date"] == days[0]
    assert result.equity_curve.loc[0, "position_shares"] == 100


def test_engine_insufficient_cash_partial_fill() -> None:
    days, bars = _scenario([10.0, 10.0, 10.0])
    result = run_backtest(
        bars,
        [Order(days[0], SYMBOL, "buy", 500)],
        initial_cash=1_010.0,
        cost_model=CostModel(),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.shares == 100  # 1000 notional + 5.01 fees fits; 200 shares does not
    assert trade.notional == pytest.approx(1_000.0)
    assert result.equity_curve.loc[0, "cash"] == pytest.approx(1_010.0 - 1_000.0 - 5.01)
    assert result.rejected == []


def test_engine_rejects_when_cash_below_one_lot() -> None:
    days, bars = _scenario([10.0, 10.0])
    result = run_backtest(
        bars,
        [Order(days[0], SYMBOL, "buy", 100)],
        initial_cash=500.0,
        cost_model=CostModel(),
    )

    assert result.trades == []
    assert result.rejected[0]["reason"] == "insufficient_cash"
    assert result.final_equity == pytest.approx(500.0)


def test_engine_fills_exactly_affordable_lot() -> None:
    """The CNY 5 commission floor must not make an exactly-funded lot look unaffordable."""
    days, bars = _scenario([10.0, 10.0])
    model = CostModel()
    exact = 1_000.0 + model.buy_cost(1_000.0)  # 1000 notional + 5.01 fees

    filled = run_backtest(
        bars, [Order(days[0], SYMBOL, "buy", 300)], initial_cash=exact, cost_model=model
    )
    assert [t.shares for t in filled.trades] == [100]
    assert filled.equity_curve.loc[0, "cash"] == pytest.approx(0.0, abs=1e-9)
    assert (filled.equity_curve["cash"] >= 0.0).all()

    short = run_backtest(
        bars, [Order(days[0], SYMBOL, "buy", 300)], initial_cash=exact - 0.01, cost_model=model
    )
    assert short.trades == []
    assert [r["reason"] for r in short.rejected] == ["insufficient_cash"]


def test_engine_duplicate_order_is_idempotent() -> None:
    days, bars = _scenario([10.0, 10.2, 10.4])
    duplicate = Order(days[0], SYMBOL, "buy", 200)
    result = run_backtest(
        bars,
        [duplicate, Order(days[0], SYMBOL, "buy", 200)],
        initial_cash=100_000.0,
        cost_model=CostModel(),
    )

    assert len(result.trades) == 1
    assert result.trades[0].shares == 200
    assert [r["reason"] for r in result.rejected] == ["duplicate_order"]
    assert result.equity_curve.loc[0, "position_shares"] == 200


def test_slippage_monotonic() -> None:
    days, bars = _scenario([10.0, 10.2, 12.0, 12.2])
    orders = [Order(days[0], SYMBOL, "buy", 1_000), Order(days[2], SYMBOL, "sell", 1_000)]

    equity = []
    for bps in (0.0, 10.0, 50.0):
        result = run_backtest(
            bars, list(orders), initial_cash=100_000.0, cost_model=CostModel(slippage_bps=bps)
        )
        assert len(result.trades) == 2
        buy, sell = result.trades
        assert buy.price == pytest.approx(10.0 * (1 + bps / 10_000.0))
        assert sell.price == pytest.approx(12.0 * (1 - bps / 10_000.0))
        equity.append(result.final_equity)

    assert equity[0] > equity[1] > equity[2], equity


def test_engine_equity_curve_matches_cash_plus_position() -> None:
    days, bars = _scenario([10.0, 10.6, 11.2, 10.9, 11.5])
    result = run_backtest(
        bars,
        [
            Order(days[0], SYMBOL, "buy", 300),
            Order(days[2], SYMBOL, "sell", 100),
            Order(days[3], SYMBOL, "buy", 200),
        ],
        initial_cash=50_000.0,
        cost_model=CostModel(slippage_bps=5.0),
    )

    curve = result.equity_curve
    assert list(curve.columns) == ["date", "cash", "market_value", "equity", "position_shares"]
    assert len(curve) == len(days)
    residual = (curve["equity"] - (curve["cash"] + curve["market_value"])).abs().max()
    assert residual <= 1e-6
    assert result.final_equity == pytest.approx(curve["equity"].iloc[-1])
    assert (curve["cash"] >= 0.0).all()
    assert (curve["position_shares"] >= 0).all()


def test_suspended_day_rejected() -> None:
    days, bars = _scenario([10.0, 10.2, 10.4])
    result = run_backtest(
        bars,
        [Order(days[1], SYMBOL, "buy", 100)],
        initial_cash=100_000.0,
        cost_model=CostModel(),
        suspended={SYMBOL: {days[1]}},
    )

    assert result.trades == []
    assert result.rejected == [
        {"date": days[1], "symbol": SYMBOL, "side": "buy", "shares": 100, "reason": "suspended"}
    ]
    assert result.final_equity == pytest.approx(100_000.0)


def test_engine_rejects_order_on_non_session_date() -> None:
    days, bars = _scenario([10.0, 10.2, 10.4])
    weekend = date(2024, 3, 9)
    assert weekend not in days

    result = run_backtest(
        bars,
        [Order(weekend, SYMBOL, "buy", 100)],
        initial_cash=100_000.0,
        cost_model=CostModel(),
    )

    # Dropping the order silently would read as "the signal fired and did nothing".
    assert result.trades == []
    assert [r["reason"] for r in result.rejected] == ["suspended"]
    assert len(result.equity_curve) == len(days)
    assert result.final_equity == pytest.approx(100_000.0)


def test_engine_rejects_buy_on_sealed_limit_up() -> None:
    days = _weekdays(START, 3)
    bars = pd.DataFrame(
        [
            {"date": pd.Timestamp(days[0]), "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0},
            {"date": pd.Timestamp(days[1]), "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0},
            {"date": pd.Timestamp(days[2]), "open": 11.5, "high": 11.8, "low": 11.2, "close": 11.6},
        ]
    )
    result = run_backtest(
        {SYMBOL: bars},
        [Order(days[1], SYMBOL, "buy", 100)],
        initial_cash=100_000.0,
        cost_model=CostModel(),
    )

    assert result.trades == []
    assert [r["reason"] for r in result.rejected] == ["limit_up"]


def test_order_rejects_odd_lot() -> None:
    with pytest.raises(ValueError):
        Order(date(2024, 3, 4), SYMBOL, "buy", 150)
    with pytest.raises(ValueError):
        Order(date(2024, 3, 4), SYMBOL, "hold", LOT_SIZE)


def test_initial_positions_are_valued_and_sellable_on_day_one() -> None:
    """A continued account starts with holdings, not just cash."""
    days, bars = _scenario([10.0, 10.5, 11.0], [10.2, 10.7, 11.1])
    held = run_backtest(
        bars,
        [],
        initial_cash=5_000.0,
        cost_model=CostModel(),
        initial_positions={SYMBOL: 300},
    )
    assert held.equity_curve.loc[0, "position_shares"] == 300
    assert held.equity_curve.loc[0, "market_value"] == pytest.approx(300 * 10.2)
    assert held.equity_curve.loc[0, "equity"] == pytest.approx(5_000.0 + 300 * 10.2)

    sold = run_backtest(
        bars,
        [Order(days[0], SYMBOL, "sell", 300)],
        initial_cash=5_000.0,
        cost_model=CostModel(),
        initial_positions={SYMBOL: 300},
    )
    assert sold.rejected == []  # settled before the window => T+1 does not block
    assert len(sold.trades) == 1
    assert sold.trades[0].side == "sell"
    assert sold.equity_curve.loc[0, "position_shares"] == 0


def test_initial_position_must_be_a_whole_lot() -> None:
    days, bars = _scenario([10.0, 10.5])
    with pytest.raises(ValueError, match="whole lot|positive multiple"):
        run_backtest(
            bars,
            [],
            initial_cash=1_000.0,
            cost_model=CostModel(),
            initial_positions={SYMBOL: 150},
        )


def test_initial_position_needs_bars() -> None:
    days, bars = _scenario([10.0, 10.5])
    with pytest.raises(ValueError, match="no bars"):
        run_backtest(
            bars,
            [],
            initial_cash=1_000.0,
            cost_model=CostModel(),
            initial_positions={"000001": 100},
        )
