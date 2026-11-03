from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stockpred.errors import BrokerNotConfirmed
from stockpred.execution import (
    OrderSheet,
    PaperBroker,
    SheetRow,
    build_order_sheet,
    reconcile,
    replay_sheet,
    require_broker_confirmation,
)
from stockpred.pipeline import DecisionPackage, ExecutionPlan
from stockpred.risk import GateResult, Proposal

DAY0 = date(2024, 3, 4)
DAY1 = date(2024, 3, 5)


@pytest.fixture
def bars() -> pd.DataFrame:
    days = [DAY0, DAY1, date(2024, 3, 6)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(days),
            "open": [10.0, 10.4, 10.2],
            "high": [10.3, 10.5, 10.6],
            "low": [9.9, 10.1, 10.0],
            "close": [10.2, 10.3, 10.5],
        }
    )


def _package(**overrides) -> DecisionPackage:
    payload = {
        "as_of": DAY0,
        "bar_date": DAY0,
        "signal": {"600519": 0.01},
        "target_shares": {"600519": 1000},
        "risk": {},
        "rationale": "test",
        "cash_after": 90_000.0,
        "target_invested": 10_000.0,
        "executed_notional": 10_000.0,
        "leftover_cash": 0.0,
    }
    payload.update(overrides)
    return DecisionPackage(**payload)


def _plan(*specs: tuple[str, str, int, float, bool]) -> ExecutionPlan:
    proposals = []
    verdicts = {}
    for symbol, side, shares, price, approved in specs:
        proposals.append(Proposal(DAY0, symbol, side, shares, price))
        verdicts[symbol] = GateResult(approved=approved, violations=() if approved else ("position_limit",))
    return ExecutionPlan(proposals=tuple(proposals), verdicts=verdicts)


def test_sheet_contains_only_executable_orders() -> None:
    package = _package()
    plan = _plan(
        ("600519", "buy", 1000, 10.0, True),
        ("000001", "buy", 500, 12.0, False),
    )
    sheet = build_order_sheet(package, plan)

    assert len(sheet.approved_rows()) == 1
    assert len(sheet.refused_rows()) == 1
    assert "refused" in sheet.refused_rows()[0].note


def test_sheet_csv_has_stable_columns() -> None:
    sheet = build_order_sheet(_package(), _plan(("600519", "buy", 1000, 10.0, True)))
    header = sheet.to_csv().splitlines()[0]
    assert header == "as_of,symbol,side,shares,limit_price,notional,valid_until,risk_status,note"


def test_sheet_text_is_readable() -> None:
    sheet = build_order_sheet(_package(), _plan(("600519", "buy", 1000, 10.0, True)))
    text = sheet.to_text()
    assert "共 1 笔可执行" in text
    assert "600519" in text
    assert "1000" in text


def test_sheet_validity_window() -> None:
    sheet = build_order_sheet(_package(), _plan(("600519", "buy", 1000, 10.0, True)), valid_days=2)
    assert sheet.rows[0].valid_until == DAY0 + timedelta(days=2)


def test_blank_sheet_is_allowed() -> None:
    sheet = OrderSheet(rows=(), as_of=DAY0)
    assert sheet.approved_rows() == ()
    assert "共 0 笔可执行" in sheet.to_text()


def test_require_broker_confirmation_blocks_when_missing() -> None:
    with pytest.raises(BrokerNotConfirmed, match="report first"):
        require_broker_confirmation(None)


def test_require_broker_confirmation_passes_when_present() -> None:
    require_broker_confirmation("2026-11-10T09:00:00+08:00")


def test_paper_buy_fills_at_open_with_costs(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    result = broker.submit(day=DAY0, symbol="600519", side="buy", shares=1000, limit_price=11.0)

    assert result.shares == 1000
    assert result.price == pytest.approx(10.0)
    assert result.fees == pytest.approx(5.1)
    assert broker.cash == pytest.approx(100_000.0 - 10_000.0 - 5.1)
    assert broker.position("600519") == 1000


def test_paper_t_plus_one_blocks_same_day_sell(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    broker.submit(day=DAY0, symbol="600519", side="buy", shares=1000, limit_price=11.0)
    result = broker.submit(day=DAY0, symbol="600519", side="sell", shares=1000, limit_price=9.0)

    assert result.reason == "t_plus_1"
    assert broker.position("600519") == 1000


def test_paper_sell_next_day_releases_position(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    broker.submit(day=DAY0, symbol="600519", side="buy", shares=1000, limit_price=11.0)
    result = broker.submit(day=DAY1, symbol="600519", side="sell", shares=1000, limit_price=9.0)

    assert result.side == "sell"
    assert result.price == pytest.approx(10.4)
    assert result.fees == pytest.approx(10.304)
    assert broker.position("600519") == 0
    assert broker.cash == pytest.approx(100_384.596)


def test_paper_suspension_blocks_fill(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0, suspended={"600519": frozenset({DAY0})})
    result = broker.submit(day=DAY0, symbol="600519", side="buy", shares=100, limit_price=11.0)
    assert result.reason == "suspended"


def test_paper_sealed_limit_up_blocks_buy(bars: pd.DataFrame) -> None:
    locked = bars.copy()
    locked.loc[locked["date"] == pd.Timestamp(DAY1), ["open", "low", "high", "close"]] = 11.22
    broker = PaperBroker({"600519": locked}, cash=100_000.0)
    result = broker.submit(day=DAY1, symbol="600519", side="buy", shares=100, limit_price=12.0)
    assert result.reason == "limit_up"


def test_paper_limit_price_guard(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    result = broker.submit(day=DAY0, symbol="600519", side="buy", shares=100, limit_price=9.5)
    assert result.reason == "limit_price_not_reached"


def test_paper_partial_fill_on_insufficient_cash(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=3_000.0)
    result = broker.submit(day=DAY0, symbol="600519", side="buy", shares=1000, limit_price=11.0)
    assert result.shares == 200  # 2000 + 5.1 <= 3000, 300 shares would not fit
    assert broker.cash >= 0


def test_paper_rejects_bad_order_shape(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=10_000.0)
    with pytest.raises(ValueError, match="multiple of"):
        broker.submit(day=DAY0, symbol="600519", side="buy", shares=150, limit_price=11.0)
    with pytest.raises(ValueError, match="side"):
        broker.submit(day=DAY0, symbol="600519", side="hold", shares=100, limit_price=11.0)


def test_replay_sheet_sells_before_buys(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    broker.submit(day=DAY0, symbol="600519", side="buy", shares=500, limit_price=11.0)
    fills, rejections = replay_sheet(
        broker, day=DAY1, orders=[("600519", "buy", 500, 11.0), ("600519", "sell", 500, 9.0)]
    )
    assert fills[0].side == "sell"
    assert rejections == []


def test_reconcile_detects_missing_fill() -> None:
    report = reconcile([{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}], [])
    assert not report.ok
    assert report.by_kind() == {"missing_fill": 1}


def test_reconcile_detects_unexpected_fill() -> None:
    report = reconcile([], [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}])
    assert report.by_kind() == {"unexpected_fill": 1}


def test_reconcile_detects_quantity_and_price_mismatch() -> None:
    expected = [
        {"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0},
        {"date": DAY1, "symbol": "000001", "side": "buy", "shares": 100, "price": 12.0},
    ]
    actual = [
        {"date": DAY0, "symbol": "600519", "side": "buy", "shares": 200, "price": 10.0},
        {"date": DAY1, "symbol": "000001", "side": "buy", "shares": 100, "price": 12.5},
    ]
    report = reconcile(expected, actual)
    assert report.by_kind() == {"quantity_mismatch": 1, "price_mismatch": 1}
    assert report.matched == 0


def test_reconcile_tolerances() -> None:
    expected = [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}]
    actual = [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.02}]
    assert not reconcile(expected, actual).ok
    assert reconcile(expected, actual, price_tolerance=0.05).ok


def test_reconcile_duplicate_keys_rejected() -> None:
    row = {"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}
    with pytest.raises(ValueError, match="duplicate"):
        reconcile([row, row], [])


def test_reconcile_report_is_serialisable() -> None:
    report = reconcile(
        [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}], []
    )
    payload = report.as_dict()
    assert payload["ok"] is False
    assert payload["discrepancies"][0]["kind"] == "missing_fill"


def test_accumulator_tracks_worst_result() -> None:
    from stockpred.execution.reconcile import ReconcileAccumulator

    acc = ReconcileAccumulator()
    ok = reconcile(
        [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}],
        [{"date": DAY0, "symbol": "600519", "side": "buy", "shares": 100, "price": 10.0}],
    )
    bad = reconcile([], [{"date": DAY1, "symbol": "600519", "side": "sell", "shares": 100, "price": 10.0}])
    acc.add(ok)
    acc.add(bad)

    assert acc.days == 2
    assert not acc.ok
    assert acc.failing_kinds() == {"unexpected_fill": 1}


def test_sheet_row_notional() -> None:
    row = SheetRow(DAY0, "600519", "buy", 200, 10.5, DAY1, "approved", "ok")
    assert row.notional == pytest.approx(2_100.0)
    assert row.as_dict()["notional"] == pytest.approx(2_100.0)


def test_paper_broker_equity(bars: pd.DataFrame) -> None:
    broker = PaperBroker({"600519": bars}, cash=100_000.0)
    broker.submit(day=DAY0, symbol="600519", side="buy", shares=1000, limit_price=11.0)
    equity = broker.equity({"600519": 10.5})
    assert equity == pytest.approx(broker.cash + 10_500.0)
    assert Path(".").exists()
