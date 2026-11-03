from __future__ import annotations

from datetime import date

import pytest

from stockpred.backtest import CostModel
from stockpred.risk import (
    Position,
    Proposal,
    RiskGate,
    RiskLimits,
    RiskState,
    mark_to_market,
)
from stockpred.risk.turnover import (
    TurnoverBudget,
    annual_cost_drag,
    max_round_trips_within_drag,
    round_trip_cost_rate,
)

DAY = date(2024, 3, 11)


def _state(cash: float = 100_000.0, **positions: tuple[int, float]) -> RiskState:
    state = RiskState(cash=cash)
    for symbol, (shares, price) in positions.items():
        state.positions[symbol] = Position(symbol, shares, price)
    state.peak_equity = state.equity()
    return state


def test_simple_buy_is_approved() -> None:
    gate = RiskGate(RiskLimits(max_position_pct=0.35, max_single_order_pct=0.35))
    verdict = gate.evaluate(Proposal(DAY, "600519", "buy", 100, 100.0), _state())
    assert verdict.approved


def test_position_limit_blocks_concentration() -> None:
    gate = RiskGate(RiskLimits(max_position_pct=0.2, max_single_order_pct=0.5))
    verdict = gate.evaluate(Proposal(DAY, "600519", "buy", 300, 100.0), _state())
    assert verdict.refused
    assert "position_limit" in verdict.violations


def test_single_order_limit_blocks_large_ticket() -> None:
    gate = RiskGate(RiskLimits(max_single_order_pct=0.1, max_position_pct=0.9))
    verdict = gate.evaluate(Proposal(DAY, "600519", "buy", 200, 100.0), _state())
    assert "single_order_too_large" in verdict.violations


def test_cash_buffer_is_respected() -> None:
    gate = RiskGate(RiskLimits(max_single_order_pct=0.99, max_position_pct=0.99, min_cash_buffer_pct=0.10))
    verdict = gate.evaluate(Proposal(DAY, "600519", "buy", 1000, 100.0), _state())
    assert "insufficient_cash" in verdict.violations


def test_gross_exposure_limit() -> None:
    gate = RiskGate(RiskLimits(max_gross_exposure_pct=0.5, max_position_pct=0.5, max_single_order_pct=0.5))
    state = _state(cash=50_000.0, **{"000001": (4000, 12.5)})
    verdict = gate.evaluate(Proposal(DAY, "600519", "buy", 100, 100.0), state)
    assert "gross_exposure_limit" in verdict.violations


def test_sell_without_position_is_refused() -> None:
    gate = RiskGate()
    verdict = gate.evaluate(Proposal(DAY, "600519", "sell", 100, 100.0), _state())
    assert "no_position_to_sell" in verdict.violations


def test_sell_more_than_held_is_refused() -> None:
    gate = RiskGate()
    state = _state(cash=90_000.0, **{"600519": (100, 100.0)})
    verdict = gate.evaluate(Proposal(DAY, "600519", "sell", 200, 100.0), state)
    assert "no_position_to_sell" in verdict.violations


def test_drawdown_breaker_stops_new_buys() -> None:
    gate = RiskGate(RiskLimits(max_drawdown_stop_pct=0.10, max_position_pct=0.5, max_single_order_pct=0.5))
    state = _state(cash=10_000.0, **{"600519": (880, 100.0)})
    state.peak_equity = 200_000.0
    verdict = gate.evaluate(Proposal(DAY, "000001", "buy", 100, 10.0), state)
    assert "drawdown_circuit_breaker" in verdict.violations


def test_orders_per_day_limit() -> None:
    gate = RiskGate(RiskLimits(max_orders_per_day=1))
    state = _state()
    state.orders_today[DAY] = 1
    verdict = gate.evaluate(Proposal(DAY, "000001", "buy", 100, 10.0), state)
    assert "max_orders_per_day" in verdict.violations


def test_turnover_budget_refuses_excess() -> None:
    limits = RiskLimits(max_turnover_per_month_pct=10.0, max_position_pct=0.9, max_single_order_pct=0.9)
    gate = RiskGate(limits)
    state = _state()
    state.turnover_month["2024-03"] = 9_000.0
    verdict = gate.evaluate(Proposal(DAY, "000001", "buy", 200, 10.0), state)
    assert "turnover_budget_exceeded" in verdict.violations


def test_apply_records_turnover_and_orders() -> None:
    gate = RiskGate()
    state = _state()
    proposal = Proposal(DAY, "000001", "buy", 100, 10.0)
    verdict = gate.evaluate(proposal, state)
    gate.apply(proposal, state, verdict)

    assert state.orders_today[DAY] == 1
    assert state.turnover_month["2024-03"] == pytest.approx(1_000.0)
    assert state.positions["000001"].shares == 100
    assert state.cash == pytest.approx(99_000.0)


def test_refused_proposal_does_not_change_holdings() -> None:
    gate = RiskGate(RiskLimits(max_single_order_pct=0.005))
    state = _state()
    proposal = Proposal(DAY, "000001", "buy", 100, 10.0)
    verdict = gate.evaluate(proposal, state)
    gate.apply(proposal, state, verdict)

    assert state.positions == {}
    assert state.cash == pytest.approx(100_000.0)
    assert state.orders_today[DAY] == 1


def test_sell_clears_position_when_fully_exited() -> None:
    gate = RiskGate()
    state = _state(cash=90_000.0, **{"600519": (100, 100.0)})
    proposal = Proposal(DAY, "600519", "sell", 100, 100.0)
    gate.apply(proposal, state, gate.evaluate(proposal, state))
    assert "600519" not in state.positions
    assert state.cash == pytest.approx(100_000.0)


def test_invalid_limits_rejected() -> None:
    with pytest.raises(ValueError, match="max_position_pct"):
        RiskLimits(max_position_pct=1.5)
    with pytest.raises(ValueError, match="max_orders_per_day"):
        RiskLimits(max_orders_per_day=0)


def test_invalid_proposals_rejected() -> None:
    with pytest.raises(ValueError, match="side"):
        Proposal(DAY, "600519", "hold", 100, 10.0)
    with pytest.raises(ValueError, match="multiple of"):
        Proposal(DAY, "600519", "buy", 150, 10.0)
    with pytest.raises(ValueError, match="positive"):
        Proposal(DAY, "600519", "buy", 100, 0.0)


def test_mark_to_market_updates_peak() -> None:
    state = _state(cash=0.0, **{"600519": (100, 100.0)})
    mark_to_market(state, {"600519": 130.0})
    assert state.equity() == pytest.approx(13_000.0)
    assert state.peak_equity == pytest.approx(13_000.0)


def test_round_trip_cost_rate_matches_plan() -> None:
    rate = round_trip_cost_rate(CostModel(), 10_000.0)
    assert rate == pytest.approx(0.00152, rel=1e-3)


def test_annual_drag_matches_plan() -> None:
    drag = annual_cost_drag(CostModel(), notional_per_trade=10_000.0, round_trips_per_month=2)
    assert drag == pytest.approx(0.0365, rel=1e-2)


def test_turnover_budget_and_trip_ceiling() -> None:
    budget = TurnoverBudget(capital=100_000.0, max_monthly_turnover_pct=100.0)
    assert budget.monthly_budget == pytest.approx(100_000.0)
    assert budget.allows(90_000.0, 10_000.0)
    assert not budget.allows(95_000.0, 10_000.0)
    assert budget.remaining(95_000.0) == pytest.approx(5_000.0)

    trips = max_round_trips_within_drag(CostModel(), notional_per_trade=10_000.0, max_annual_drag=0.05)
    assert trips == 2


def test_invalid_turnover_inputs() -> None:
    with pytest.raises(ValueError):
        round_trip_cost_rate(CostModel(), 0.0)
    with pytest.raises(ValueError):
        annual_cost_drag(CostModel(), notional_per_trade=1000.0, round_trips_per_month=-1)
