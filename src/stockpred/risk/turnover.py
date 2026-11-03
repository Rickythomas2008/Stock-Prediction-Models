"""Turnover budget: the cost guard for a 100k A-share account.

With a 5 CNY commission floor, a 10,000 CNY round trip costs ~0.15%. Two round
trips a month is ~3.6% a year of drag. Turnover is therefore a *hard* budget,
not a statistic reported after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from stockpred.backtest.costs import CostModel


@dataclass(frozen=True)
class TurnoverBudget:
    capital: float = 100_000.0
    max_monthly_turnover_pct: float = 100.0

    @property
    def monthly_budget(self) -> float:
        return self.max_monthly_turnover_pct / 100.0 * self.capital

    def remaining(self, used: float) -> float:
        return max(self.monthly_budget - used, 0.0)

    def allows(self, used: float, notional: float) -> bool:
        return used + notional <= self.monthly_budget + 1e-9


def round_trip_cost_rate(cost_model: CostModel, notional: float) -> float:
    """Cost of buying and selling ``notional`` once, as a fraction of notional."""
    if notional <= 0:
        raise ValueError("notional must be positive")
    return (cost_model.buy_cost(notional) + cost_model.sell_cost(notional)) / notional


def annual_cost_drag(cost_model: CostModel, *, notional_per_trade: float, round_trips_per_month: int) -> float:
    """Yearly cost drag from a steady turnover schedule (fraction of capital per year).

    ``notional_per_trade`` is the size of one side of a round trip; 12 months of
    ``round_trips_per_month`` round trips are assumed.
    """
    if round_trips_per_month < 0:
        raise ValueError("round_trips_per_month must be non-negative")
    per_round_trip = cost_model.buy_cost(notional_per_trade) + cost_model.sell_cost(
        notional_per_trade
    )
    return per_round_trip * round_trips_per_month * 12 / notional_per_trade


def max_round_trips_within_drag(
    cost_model: CostModel,
    *,
    notional_per_trade: float,
    max_annual_drag: float,
) -> int:
    """How many round trips per month fit inside a drag ceiling."""
    per_month = annual_cost_drag(
        cost_model, notional_per_trade=notional_per_trade, round_trips_per_month=1
    )
    if per_month <= 0:
        return 0
    return int(max_annual_drag / per_month)
