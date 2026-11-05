"""The alpha -> portfolio -> execution pipeline, and the decision package it emits.

Three stages, each replaceable without touching the others:

* **alpha** turns bars into cross-sectional scores
* **constructor** turns scores into target weights (and then into whole lots)
* **planner** turns target holdings into orders, then hands each order to the
  risk gate

The output is a :class:`DecisionPackage`, which is the only thing the rest of the
system may consume. It carries five mandatory fields -- signal, target position,
risk verdict, rationale, and the data timestamp -- because a decision without any
one of them cannot be reviewed after the fact, and review is the whole point of a
live decision log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from stockpred.backtest.rules import LOT_SIZE
from stockpred.errors import StockpredError
from stockpred.portfolio import Allocation, apply_caps, to_lots
from stockpred.risk import GateResult, Proposal, RiskGate, RiskState


class PipelineError(StockpredError):
    """A stage violated its contract (missing stage, bad frame, bad output)."""


REQUIRED_DECISION_FIELDS = (
    "as_of",
    "bar_date",
    "signal",
    "target_shares",
    "risk",
    "rationale",
)


@runtime_checkable
class AlphaModel(Protocol):
    name: str

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series: ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    name: str

    def construct(
        self, scores: pd.Series, returns: pd.DataFrame, *, capital: float
    ) -> pd.Series: ...


@dataclass(frozen=True)
class DecisionPackage:
    """One reviewable decision.

    ``target_invested`` is the allocation the constructor asked for;
    ``executed_notional`` is what the risk gate actually approved. They differ
    whenever an order is refused, and conflating them would overstate the
    position taken.
    """

    as_of: date
    bar_date: date
    signal: dict[str, float]
    target_shares: dict[str, int]
    risk: dict[str, tuple[str, ...]]
    rationale: str
    cash_after: float
    target_invested: float
    executed_notional: float
    leftover_cash: float
    dropped: tuple[str, ...] = ()
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, object]:
        payload = {
            "as_of": self.as_of.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "bar_date": self.bar_date.isoformat(),
            "signal": self.signal,
            "target_shares": self.target_shares,
            "risk": {key: list(value) for key, value in self.risk.items()},
            "rationale": self.rationale,
            "cash_after": round(self.cash_after, 2),
            "target_invested": round(self.target_invested, 2),
            "executed_notional": round(self.executed_notional, 2),
            "leftover_cash": round(self.leftover_cash, 2),
            "dropped": list(self.dropped),
        }
        missing = [name for name in REQUIRED_DECISION_FIELDS if _is_blank(payload, name)]
        if missing:
            raise PipelineError(f"decision package is missing required fields: {missing}")
        return payload


@dataclass(frozen=True)
class MomentumAlpha:
    """Cross-sectional momentum: score = close / trailing mean - 1."""

    lookback: int = 20
    name: str = "momentum"

    def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol, frame in bars.items():
            history = _history_until(frame, as_of)
            if history is None or len(history) < self.lookback + 1:
                continue
            close = history["close"].astype(float).to_numpy()
            mean = close[-self.lookback :].mean()
            if mean <= 0:
                continue
            scores[symbol] = float(close[-1] / mean - 1.0)
        if not scores:
            raise PipelineError(f"no symbol has {self.lookback}+ bars up to {as_of}")
        return pd.Series(scores, dtype=float).sort_values(ascending=False)


@dataclass(frozen=True)
class EqualWeightConstructor:
    name: str = "equal_weight"

    def construct(
        self,
        scores: pd.Series,
        returns: pd.DataFrame,
        *,
        capital: float,
        held: frozenset[str] = frozenset(),
    ) -> pd.Series:
        if scores.empty:
            raise PipelineError("no scores to construct from")
        return pd.Series(1.0 / len(scores), index=scores.index, dtype=float)


@dataclass(frozen=True)
class TopNConstructor:
    """Keep the ``top_n`` best scores, equal weight them, then cap each name."""

    top_n: int = 5
    max_position: float = 0.25
    name: str = "top_n"

    def construct(
        self,
        scores: pd.Series,
        returns: pd.DataFrame,
        *,
        capital: float,
        held: frozenset[str] = frozenset(),
    ) -> pd.Series:
        if scores.empty:
            raise PipelineError("no scores to construct from")
        chosen = scores.sort_values(ascending=False).head(self.top_n)
        weights = pd.Series(1.0 / len(chosen), index=chosen.index, dtype=float)
        return apply_caps(weights, max_position=self.max_position)


@dataclass(frozen=True)
class BufferedTopNConstructor:
    """Top-N with a stay band: a holding is kept while it stays in ``top_n + buffer``.

    Re-ranking every period churns the portfolio on tiny score differences;
    every switch pays commission, stamp duty and slippage twice. Giving incumbents
    a place in the queue means a challenger must be clearly better to displace
    them, which is the cheapest turnover reduction that does not change the alpha.
    """

    top_n: int = 4
    buffer: int = 2
    max_position: float = 0.40
    name: str = "buffered_top_n"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n!r}")
        if self.buffer < 0:
            raise ValueError(f"buffer must be non-negative, got {self.buffer!r}")

    def construct(
        self,
        scores: pd.Series,
        returns: pd.DataFrame,
        *,
        capital: float,
        held: frozenset[str] = frozenset(),
    ) -> pd.Series:
        if scores.empty:
            raise PipelineError("no scores to construct from")
        if self.top_n + self.buffer >= len(scores):
            raise PipelineError(
                f"stay band top_n={self.top_n} + buffer={self.buffer} covers all "
                f"{len(scores)} scored names: holdings could never rotate, which turns the "
                "strategy into buy-and-hold. Use a smaller buffer."
            )
        ranked = scores.sort_values(ascending=False)
        candidates = list(ranked.head(self.top_n + self.buffer).index)
        keep = [symbol for symbol in candidates if symbol in held][: self.top_n]
        fill = [symbol for symbol in candidates if symbol not in keep][: self.top_n - len(keep)]
        chosen = keep + fill
        if not chosen:
            raise PipelineError("stay band produced an empty portfolio")
        weights = pd.Series(1.0 / len(chosen), index=chosen, dtype=float)
        return apply_caps(weights, max_position=self.max_position)


@dataclass(frozen=True)
class TiltedIndexConstructor:
    """Index enhancement: start from the benchmark weights, tilt by score, stay close.

    ``w_i ∝ bench_i * exp(tilt * z_i)`` where ``z`` is the cross-sectional z-score of
    the alpha and ``tilt`` is in units of "log-weight per sigma". ``tilt=0``
    reproduces the index; larger values buy more active risk. Names outside the
    benchmark are ignored — an enhanced index that leaves its own universe is just
    an active fund with a tracking-error footnote.
    """

    benchmark_weights: Mapping[str, float]
    tilt: float = 0.5
    max_position: float = 0.10
    name: str = "tilted_index"

    def construct(
        self,
        scores: pd.Series,
        returns: pd.DataFrame,
        *,
        capital: float,
        held: frozenset[str] = frozenset(),
    ) -> pd.Series:
        if scores.empty:
            raise PipelineError("no scores to construct from")
        pool = [symbol for symbol in scores.index if symbol in self.benchmark_weights]
        if not pool:
            raise PipelineError("no scored symbol is in the benchmark")
        bench = pd.Series(
            {symbol: float(self.benchmark_weights[symbol]) for symbol in pool}, dtype=float
        )
        bench = bench / bench.sum()

        values = scores.loc[pool].astype(float)
        spread = float(values.std(ddof=0))
        if spread <= 0:
            return apply_caps(bench, max_position=self.max_position)
        z = (values - values.mean()) / spread
        tilted = bench * np.exp(self.tilt * z)
        tilted = tilted / tilted.sum()
        return apply_caps(tilted, max_position=self.max_position)


@dataclass(frozen=True)
class ExecutionPlan:
    proposals: tuple[Proposal, ...]
    verdicts: dict[str, GateResult]

    @property
    def approved(self) -> tuple[Proposal, ...]:
        return tuple(p for p in self.proposals if self.verdicts[p.symbol].approved)


def plan_execution(
    target_shares: Mapping[str, int],
    state: RiskState,
    *,
    as_of: date,
    prices: Mapping[str, float],
    gate: RiskGate,
    min_trade_notional: float = 0.0,
) -> ExecutionPlan:
    """Diff target vs current holdings into sell-first, then buy orders.

    ``min_trade_notional`` is the no-trade band: a drift correction smaller than
    that is skipped, because re-buying every name back to its exact weight each
    period pays costs for nothing. Exits are never skipped — leaving a position is
    a decision, not drift.
    """
    sells: list[Proposal] = []
    buys: list[Proposal] = []

    symbols = sorted(set(target_shares) | set(state.positions))
    for symbol in symbols:
        target = int(target_shares.get(symbol, 0))
        held = state.positions.get(symbol)
        current = held.shares if held else 0
        delta = target - current
        if delta == 0:
            continue
        price = prices.get(symbol, held.price if held else None)
        if price is None or price <= 0:
            raise PipelineError(f"no usable price for {symbol}")
        if delta < 0:
            shares = min(-delta, current)
            leaving = target == 0
            proposal = Proposal(as_of, symbol, "sell", shares, float(price))
        else:
            leaving = False
            proposal = Proposal(as_of, symbol, "buy", _whole_lots(delta), float(price))
        if (
            min_trade_notional > 0
            and not leaving
            and proposal.shares > 0
            and proposal.notional < min_trade_notional
        ):
            continue
        if delta < 0:
            sells.append(proposal)
        else:
            buys.append(proposal)

    verdicts: dict[str, GateResult] = {}
    ordered: list[Proposal] = []
    for proposal in sells + buys:
        if proposal.shares <= 0:
            continue
        verdict = gate.evaluate(proposal, state)
        verdicts[proposal.symbol] = verdict
        ordered.append(proposal)
        gate.apply(proposal, state, verdict)
    return ExecutionPlan(proposals=tuple(ordered), verdicts=verdicts)


class Pipeline:
    """Compose the three stages. Any stage can be swapped for a fake in tests."""

    def __init__(
        self,
        alpha: AlphaModel,
        constructor: PortfolioConstructor,
        *,
        gate: RiskGate | None = None,
        capital: float = 100_000.0,
    ) -> None:
        if alpha is None or constructor is None:
            raise PipelineError("alpha and constructor are required")
        self.alpha = alpha
        self.constructor = constructor
        self.gate = gate or RiskGate()
        self.capital = float(capital)

        # A constructor that targets more than the gate permits would have every
        # order refused -- a silent dead pipeline. Fail at wiring time instead.
        constructor_cap = getattr(constructor, "max_position", None)
        if (
            constructor_cap is not None
            and constructor_cap > self.gate.limits.max_position_pct + 1e-9
        ):
            raise PipelineError(
                f"{constructor.name} targets up to {constructor_cap:.0%} per name but the risk "
                f"gate caps positions at {self.gate.limits.max_position_pct:.0%}; "
                "align RiskLimits.max_position_pct with the constructor"
            )

    def decide(
        self,
        bars: Mapping[str, pd.DataFrame],
        *,
        as_of: date,
        bar_date: date,
        prices: Mapping[str, float],
        state: RiskState | None = None,
        returns: pd.DataFrame | None = None,
        capital: float | None = None,
        min_trade_notional: float = 0.0,
    ) -> tuple[DecisionPackage, ExecutionPlan]:
        if bar_date > as_of:
            raise PipelineError(f"bar_date {bar_date} is in the future of as_of {as_of}")

        risk_state = state if state is not None else RiskState(cash=self.capital)
        # Size off live equity when the caller passes it: a rebalance in a running
        # account must use what the account is worth now, not the day-one capital.
        budget = float(capital) if capital is not None else self.capital

        scores = self.alpha.score(bars, as_of=as_of)
        weights = self.constructor.construct(
            scores,
            returns if returns is not None else pd.DataFrame(),
            capital=budget,
            held=frozenset(risk_state.positions),
        )
        allocation: Allocation = to_lots(weights, capital=budget, prices=dict(prices))

        plan = plan_execution(
            allocation.shares,
            risk_state,
            as_of=as_of,
            prices=prices,
            gate=self.gate,
            min_trade_notional=min_trade_notional,
        )

        refused = {
            symbol: verdict.violations
            for symbol, verdict in plan.verdicts.items()
            if verdict.refused
        }
        executed = sum(
            proposal.notional
            for proposal in plan.proposals
            if plan.verdicts[proposal.symbol].approved
        )
        rationale = _rationale(self.alpha.name, self.constructor.name, scores, allocation, refused)

        package = DecisionPackage(
            as_of=as_of,
            bar_date=bar_date,
            signal={symbol: round(float(value), 6) for symbol, value in scores.items()},
            target_shares=dict(allocation.shares),
            risk=refused,
            rationale=rationale,
            cash_after=risk_state.cash,
            target_invested=allocation.invested,
            executed_notional=executed,
            leftover_cash=allocation.leftover_cash,
            dropped=allocation.dropped,
        )
        return package, plan


def _is_blank(payload: Mapping[str, object], name: str) -> bool:
    """Missing means "cannot be reviewed", not "happens to be empty".

    ``risk`` legitimately arrives empty when the gate refused nothing; that is a
    complete answer, so only the structural fields are treated as required.
    """
    if name not in payload or payload[name] is None:
        return True
    value = payload[name]
    if isinstance(value, str):
        return not value.strip()
    if name in {"signal", "target_shares"}:
        return not value
    return False


def _rationale(
    alpha_name: str,
    constructor_name: str,
    scores: pd.Series,
    allocation: Allocation,
    refused: Mapping[str, Sequence[str]],
) -> str:
    top = ", ".join(f"{symbol}={scores[symbol]:+.4f}" for symbol in scores.index[:3])
    parts = [
        f"alpha={alpha_name}",
        f"constructor={constructor_name}",
        f"top_scores[{top}]",
        f"positions={len(allocation.shares)}",
        f"invested={allocation.invested:.0f}/{allocation.invested + allocation.leftover_cash:.0f}",
    ]
    if allocation.dropped:
        parts.append(f"dropped_below_one_lot={list(allocation.dropped)}")
    if refused:
        parts.append(f"risk_refusals={ {k: list(v) for k, v in refused.items()} }")
    return "; ".join(parts)


def _history_until(frame: pd.DataFrame, as_of: date) -> pd.DataFrame | None:
    if "date" not in frame.columns:
        raise PipelineError("bars frame must have a 'date' column")
    dates = pd.to_datetime(frame["date"])
    mask = dates <= pd.Timestamp(as_of)
    history = frame.loc[mask].sort_values("date")
    return None if history.empty else history


def _whole_lots(shares: int) -> int:
    return int(shares // LOT_SIZE) * LOT_SIZE


def append_decision_log(path, package: DecisionPackage) -> None:
    """Append one JSON line. Append-only: a log that can be rewritten is not an audit trail."""
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(package.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = [
    "REQUIRED_DECISION_FIELDS",
    "AlphaModel",
    "BufferedTopNConstructor",
    "DecisionPackage",
    "EqualWeightConstructor",
    "ExecutionPlan",
    "MomentumAlpha",
    "Pipeline",
    "PipelineError",
    "PortfolioConstructor",
    "TiltedIndexConstructor",
    "TopNConstructor",
    "append_decision_log",
    "plan_execution",
]
