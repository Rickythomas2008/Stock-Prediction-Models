"""Produce one decision package end to end, on synthetic data.

Demonstrates the S3 contract: alpha -> portfolio -> execution -> risk gate ->
decision package with the five mandatory fields, written to an append-only log.

Usage:  python tools/demo_decide.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.pipeline import MomentumAlpha, Pipeline, TopNConstructor, append_decision_log
from stockpred.risk import RiskGate, RiskLimits, RiskState

CAPITAL = 100_000.0
SEED = 20240311
LOG_PATH = Path("logs/decisions.jsonl")


def synthetic_bars(*, n: int = 120, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-09-01", periods=n)
    trends = {
        "600519": 0.0010,
        "000001": 0.0006,
        "300750": -0.0004,
        "600000": 0.0003,
        "601318": 0.0008,
        "000002": -0.0007,
    }
    out: dict[str, pd.DataFrame] = {}
    for symbol, drift in trends.items():
        price = float(rng.uniform(8.0, 45.0))
        closes = []
        for _ in range(n):
            price = max(1.0, price * (1.0 + drift + float(rng.normal(0.0, 0.012))))
            closes.append(round(price, 2))
        out[symbol] = pd.DataFrame(
            {
                "date": days,
                "open": closes,
                "high": [round(p * 1.01, 2) for p in closes],
                "low": [round(p * 0.99, 2) for p in closes],
                "close": closes,
                "volume": 1_000_000.0,
            }
        )
    return out


def main() -> int:
    bars = synthetic_bars()
    as_of = pd.Timestamp(bars["600519"]["date"].iloc[-1]).date()
    prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in bars.items()}

    limits = RiskLimits(
        initial_capital=CAPITAL,
        max_position_pct=0.35,
        max_single_order_pct=0.35,
        max_turnover_per_month_pct=100.0,
    )
    pipeline = Pipeline(
        MomentumAlpha(lookback=20),
        TopNConstructor(top_n=3, max_position=0.35),
        gate=RiskGate(limits),
        capital=CAPITAL,
    )

    state = RiskState(cash=CAPITAL)
    package, plan = pipeline.decide(
        bars, as_of=as_of, bar_date=as_of, prices=prices, state=state
    )
    append_decision_log(LOG_PATH, package)

    payload = package.as_dict()
    print("=== decision package ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print("\n=== orders ===")
    for proposal in plan.proposals:
        verdict = plan.verdicts[proposal.symbol]
        status = "APPROVED" if verdict.approved else f"REFUSED {list(verdict.violations)}"
        print(
            f"{proposal.side:4} {proposal.symbol} {proposal.shares:>5} @ {proposal.price:8.2f} "
            f"= {proposal.notional:10,.2f}  {status}"
        )
    print(
        f"\ntarget invested: {package.target_invested:,.2f} | "
        f"executed: {package.executed_notional:,.2f} | cash after: {state.cash:,.2f}"
    )
    print(f"log appended: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
