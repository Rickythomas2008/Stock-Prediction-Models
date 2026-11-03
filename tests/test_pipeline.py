from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stockpred.pipeline import (
    REQUIRED_DECISION_FIELDS,
    DecisionPackage,
    EqualWeightConstructor,
    MomentumAlpha,
    Pipeline,
    PipelineError,
    TopNConstructor,
    append_decision_log,
    plan_execution,
)
from stockpred.risk import Position, RiskGate, RiskLimits, RiskState

AS_OF = date(2024, 3, 20)


@pytest.fixture
def bars() -> dict[str, pd.DataFrame]:
    days = pd.date_range("2024-01-01", periods=60, freq="B")
    out: dict[str, pd.DataFrame] = {}
    trends = {"600519": 0.004, "000001": 0.0005, "300750": -0.003, "600000": 0.001}
    for symbol, drift in trends.items():
        price = 20.0
        closes = []
        for _ in days:
            price *= 1.0 + drift
            closes.append(round(price, 2))
        out[symbol] = pd.DataFrame(
            {
                "date": days,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": 1_000_000.0,
            }
        )
    return out


@pytest.fixture
def prices(bars: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {symbol: float(frame["close"].iloc[-1]) for symbol, frame in bars.items()}


def _pipeline(**limit_overrides) -> Pipeline:
    limits = RiskLimits(max_position_pct=0.35, max_single_order_pct=0.35, **limit_overrides)
    return Pipeline(
        MomentumAlpha(lookback=20),
        TopNConstructor(top_n=2, max_position=0.35),
        gate=RiskGate(limits),
        capital=100_000.0,
    )


def test_momentum_alpha_ranks_by_strength(bars: dict[str, pd.DataFrame]) -> None:
    scores = MomentumAlpha(lookback=20).score(bars, as_of=AS_OF)
    assert scores.index[0] == "600519"
    assert scores.index[-1] == "300750"


def test_alpha_ignores_future_rows(bars: dict[str, pd.DataFrame]) -> None:
    trimmed = {k: v[v["date"] <= pd.Timestamp(AS_OF)] for k, v in bars.items()}
    full = MomentumAlpha(lookback=20).score(bars, as_of=AS_OF)
    cut = MomentumAlpha(lookback=20).score(trimmed, as_of=AS_OF)
    pd.testing.assert_series_equal(full, cut)


def test_decision_package_has_required_fields(bars, prices) -> None:
    package, _ = _pipeline().decide(
        bars, as_of=AS_OF, bar_date=AS_OF, prices=prices, state=RiskState(cash=100_000.0)
    )
    payload = package.as_dict()
    for field_name in REQUIRED_DECISION_FIELDS:
        assert field_name in payload, f"{field_name} must be present"
    # risk may legitimately be empty (nothing refused); structure is what matters
    assert payload["signal"]
    assert payload["target_shares"]
    assert isinstance(payload["risk"], dict)


def test_package_rejects_future_bar_date(bars, prices) -> None:
    package = DecisionPackage(
        as_of=AS_OF,
        bar_date=AS_OF + timedelta(days=1),
        signal={"600519": 1.0},
        target_shares={"600519": 100},
        risk={},
        rationale="x",
        cash_after=1.0,
        target_invested=1.0,
        executed_notional=0.0,
        leftover_cash=0.0,
    )
    with pytest.raises(PipelineError, match="future of as_of"):
        _pipeline().decide(
            bars,
            as_of=AS_OF,
            bar_date=package.bar_date,
            prices=prices,
            state=RiskState(cash=100_000.0),
        )


def test_targets_are_whole_lots(bars, prices) -> None:
    package, _ = _pipeline().decide(
        bars, as_of=AS_OF, bar_date=AS_OF, prices=prices, state=RiskState(cash=100_000.0)
    )
    assert all(shares % 100 == 0 for shares in package.target_shares.values())


def test_pipeline_wiring_guard_rejects_mismatched_caps() -> None:
    with pytest.raises(PipelineError, match="align RiskLimits"):
        Pipeline(
            MomentumAlpha(),
            TopNConstructor(top_n=3, max_position=0.5),
            gate=RiskGate(RiskLimits(max_position_pct=0.25)),
            capital=100_000.0,
        )


def test_fake_portfolio_stage_is_swappable(bars, prices) -> None:
    """A replacement constructor must slot in without touching alpha or execution."""

    class FixedConstructor:
        name = "fixed"
        max_position = 0.35

        def construct(self, scores, returns, *, capital):
            return pd.Series({scores.index[0]: 0.35})

    pipeline = Pipeline(
        MomentumAlpha(lookback=20),
        FixedConstructor(),
        gate=RiskGate(RiskLimits(max_position_pct=0.35, max_single_order_pct=0.35)),
        capital=100_000.0,
    )
    package, plan = pipeline.decide(
        bars, as_of=AS_OF, bar_date=AS_OF, prices=prices, state=RiskState(cash=100_000.0)
    )
    assert list(package.target_shares) == ["600519"]
    assert plan.approved


def test_risk_gate_refusals_are_reported_not_hidden(bars, prices) -> None:
    strict = Pipeline(
        MomentumAlpha(lookback=20),
        EqualWeightConstructor(),
        gate=RiskGate(RiskLimits(max_position_pct=0.01, max_single_order_pct=0.01)),
        capital=100_000.0,
    )
    package, plan = strict.decide(
        bars, as_of=AS_OF, bar_date=AS_OF, prices=prices, state=RiskState(cash=100_000.0)
    )
    assert not plan.approved
    assert package.risk
    assert "risk_refusals" in package.rationale


def test_plan_execution_sells_before_buys(bars, prices) -> None:
    state = RiskState(cash=50_000.0)
    state.positions["300750"] = Position("300750", 200, prices["300750"])
    plan = plan_execution(
        {"600519": 100},
        state,
        as_of=AS_OF,
        prices=prices,
        gate=RiskGate(RiskLimits(max_position_pct=0.9, max_single_order_pct=0.9)),
    )
    sides = [proposal.side for proposal in plan.proposals]
    assert sides[0] == "sell"


def test_plan_execution_skips_unchanged_positions(bars, prices) -> None:
    state = RiskState(cash=50_000.0)
    state.positions["600519"] = Position("600519", 100, prices["600519"])
    plan = plan_execution(
        {"600519": 100},
        state,
        as_of=AS_OF,
        prices=prices,
        gate=RiskGate(),
    )
    assert plan.proposals == ()


def test_plan_execution_requires_price(bars) -> None:
    with pytest.raises(PipelineError, match="no usable price"):
        plan_execution({"600519": 100}, RiskState(cash=10_000.0), as_of=AS_OF, prices={}, gate=RiskGate())


def test_decision_log_is_append_only(tmp_path: Path, bars, prices) -> None:
    package, _ = _pipeline().decide(
        bars, as_of=AS_OF, bar_date=AS_OF, prices=prices, state=RiskState(cash=100_000.0)
    )
    log = tmp_path / "logs" / "decisions.jsonl"
    append_decision_log(log, package)
    append_decision_log(log, package)

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["as_of"] == AS_OF.isoformat()


def test_momentum_needs_enough_history(bars) -> None:
    with pytest.raises(PipelineError, match="no symbol has"):
        MomentumAlpha(lookback=500).score(bars, as_of=AS_OF)


def test_alpha_requires_date_column(bars) -> None:
    broken = {symbol: frame.drop(columns=["date"]) for symbol, frame in bars.items()}
    with pytest.raises(PipelineError, match="date"):
        MomentumAlpha().score(broken, as_of=AS_OF)
