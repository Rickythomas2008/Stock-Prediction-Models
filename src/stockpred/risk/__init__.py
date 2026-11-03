"""Risk gates that sit at the end of the decision pipeline.

Design rule: the gate does not trust the model, the strategy, or the caller. It
takes a proposal and either approves or refuses it with a reason, and every
refusal is part of the audit trail. "Do not trade" is a valid outcome.

Sized for a 100,000 CNY A-share account: 100-share lots and a 5 CNY minimum
commission make turnover the dominant risk, not volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from stockpred.backtest.rules import LOT_SIZE


@dataclass(frozen=True)
class RiskLimits:
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.25
    max_gross_exposure_pct: float = 0.95
    max_single_order_pct: float = 0.25
    max_turnover_per_month_pct: float = 100.0
    max_drawdown_stop_pct: float = 0.15
    max_orders_per_day: int = 5
    min_cash_buffer_pct: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "max_position_pct",
            "max_gross_exposure_pct",
            "max_single_order_pct",
            "min_cash_buffer_pct",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        if self.max_orders_per_day < 1:
            raise ValueError("max_orders_per_day must be >= 1")


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int
    price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.price


@dataclass(frozen=True)
class Proposal:
    date: date
    symbol: str
    side: str
    shares: int
    price: float

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.shares <= 0:
            raise ValueError("shares must be positive")
        if self.shares % LOT_SIZE != 0:
            raise ValueError(f"shares must be a multiple of {LOT_SIZE}")
        if self.price <= 0:
            raise ValueError("price must be positive")

    @property
    def notional(self) -> float:
        return self.shares * self.price


@dataclass(frozen=True)
class GateResult:
    approved: bool
    violations: tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        return not self.approved


@dataclass
class RiskState:
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    peak_equity: float = 0.0
    orders_today: dict[date, int] = field(default_factory=dict)
    turnover_month: dict[str, float] = field(default_factory=dict)

    def equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    def gross_exposure(self) -> float:
        return sum(position.market_value for position in self.positions.values())


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(self, proposal: Proposal, state: RiskState) -> GateResult:
        violations: list[str] = []
        limits = self.limits
        equity = max(state.equity(), 1e-9)

        if state.orders_today.get(proposal.date, 0) >= limits.max_orders_per_day:
            violations.append("max_orders_per_day")

        if proposal.side == "buy":
            violations.extend(self._buy_violations(proposal, state, equity))
        else:
            violations.extend(self._sell_violations(proposal, state))

        if equity < state.peak_equity * (1.0 - limits.max_drawdown_stop_pct):
            violations.append("drawdown_circuit_breaker")

        month_key = proposal.date.strftime("%Y-%m")
        used = state.turnover_month.get(month_key, 0.0)
        budget = limits.max_turnover_per_month_pct / 100.0 * limits.initial_capital
        if used + proposal.notional > budget:
            violations.append("turnover_budget_exceeded")

        return GateResult(approved=not violations, violations=tuple(violations))

    def _buy_violations(self, proposal: Proposal, state: RiskState, equity: float) -> list[str]:
        limits = self.limits
        out: list[str] = []
        if proposal.notional > limits.max_single_order_pct * equity:
            out.append("single_order_too_large")

        held = state.positions.get(proposal.symbol)
        held_value = held.market_value if held else 0.0
        if held_value + proposal.notional > limits.max_position_pct * equity:
            out.append("position_limit")

        if state.gross_exposure() + proposal.notional > limits.max_gross_exposure_pct * equity:
            out.append("gross_exposure_limit")

        buffer = limits.min_cash_buffer_pct * equity
        if proposal.notional > state.cash - buffer:
            out.append("insufficient_cash")
        return out

    def _sell_violations(self, proposal: Proposal, state: RiskState) -> list[str]:
        held = state.positions.get(proposal.symbol)
        if held is None or held.shares < proposal.shares:
            return ["no_position_to_sell"]
        return []

    def apply(self, proposal: Proposal, state: RiskState, result: GateResult) -> RiskState:
        """Mutate ``state`` only for approved proposals; refusals leave no trace in holdings."""
        if result.refused:
            state.orders_today[proposal.date] = state.orders_today.get(proposal.date, 0) + 1
            return state

        month_key = proposal.date.strftime("%Y-%m")
        state.turnover_month[month_key] = (
            state.turnover_month.get(month_key, 0.0) + proposal.notional
        )
        state.orders_today[proposal.date] = state.orders_today.get(proposal.date, 0) + 1

        if proposal.side == "buy":
            state.cash -= proposal.notional
            held = state.positions.get(proposal.symbol)
            shares = (held.shares if held else 0) + proposal.shares
            state.positions[proposal.symbol] = Position(
                proposal.symbol, shares, proposal.price
            )
        else:
            state.cash += proposal.notional
            held = state.positions[proposal.symbol]
            remaining = held.shares - proposal.shares
            if remaining:
                state.positions[proposal.symbol] = Position(
                    proposal.symbol, remaining, proposal.price
                )
            else:
                del state.positions[proposal.symbol]

        state.peak_equity = max(state.peak_equity, state.equity())
        return state


def mark_to_market(state: RiskState, prices: dict[str, float]) -> RiskState:
    """Re-price holdings; the drawdown breaker needs current values, not entry prices."""
    updated: dict[str, Position] = {}
    for symbol, position in state.positions.items():
        price = prices.get(symbol, position.price)
        updated[symbol] = Position(symbol, position.shares, price)
    state.positions = updated
    state.peak_equity = max(state.peak_equity, state.equity())
    return state
