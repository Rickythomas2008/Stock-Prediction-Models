"""End-to-end demonstration of the S2 evaluation gate.

Runs a momentum signal on a deterministic synthetic A-share-like series through
walk-forward folds, the rule-aware engine (with and without costs) and the
benchmark suite, then renders a report to ``docs/SAMPLE-REPORT.md``.

The point is not the strategy's return -- it is that every required number
(gross, net, three benchmarks, fold boundaries) appears in one artifact, and
that the pipeline runs headless.

Usage:  python tools/demo_backtest.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.backtest import CostModel, Order, run_backtest
from stockpred.eval import BenchmarkSuite, PerformanceMetrics, compute_metrics, walk_forward_folds
from stockpred.eval.report import (
    benchmark_warnings,
    cost_impact,
    failing_gates,
    render_report,
)
from stockpred.eval.walkforward import assert_no_overlap

TRADING_DAYS = 260
SEED = 20240102
INITIAL_CASH = 100_000.0


def synthetic_bars(n: int = TRADING_DAYS, seed: int = SEED) -> pd.DataFrame:
    """A-share-plausible daily bars: drift, vol clustering, ±10% clamp."""
    rng = np.random.default_rng(seed)
    days: list[date] = []
    cursor = date(2024, 1, 2)
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)

    returns = rng.normal(0.0006, 0.018, size=n)
    returns = np.clip(returns, -0.099, 0.099)
    close = 12.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n)))
    volume = rng.integers(500_000, 3_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(days),
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
        }
    )


def momentum_orders(bars: pd.DataFrame, *, window: int = 20, symbol: str = "600519") -> list[Order]:
    """Buy when the close is above its trailing mean, flat otherwise.

    Signals use only bars up to and including day ``i``; the order is dated
    ``i + 1``, which is the day it fills on the open.
    """
    close = bars["close"].to_numpy(dtype=float)
    dates = [pd.Timestamp(value).date() for value in bars["date"]]
    mean = pd.Series(close).rolling(window).mean().to_numpy()

    orders: list[Order] = []
    position = 0
    for i in range(window, len(close) - 1):
        want_long = close[i] > mean[i]
        if want_long and position == 0:
            orders.append(Order(dates[i + 1], symbol, "buy", 5000))
            position = 5000
        elif not want_long and position > 0:
            orders.append(Order(dates[i + 1], symbol, "sell", position))
            position = 0
    return orders


def run_scenario(cost_model: CostModel, bars: pd.DataFrame, orders: list[Order]):
    return run_backtest(
        {"600519": bars},
        orders,
        initial_cash=INITIAL_CASH,
        cost_model=cost_model,
    )


def suite_metrics(suite: BenchmarkSuite) -> dict[str, PerformanceMetrics]:
    """Metrics for each benchmark curve, for the advisory comparison."""
    return {
        label: compute_metrics(curve)
        for label, curve in suite.all().items()
    }


def main() -> int:
    bars = synthetic_bars()
    orders = momentum_orders(bars)
    zero_cost = replace(CostModel(), commission_rate=0.0, min_commission=0.0,
                        stamp_duty_rate=0.0, transfer_fee_rate=0.0, slippage_bps=0.0)

    net = run_scenario(CostModel(slippage_bps=5.0), bars, orders)
    gross = run_scenario(zero_cost, bars, orders)

    traded = sum(trade.notional for trade in net.trades)
    impact = cost_impact(gross.equity_curve["equity"], net.equity_curve["equity"],
                         traded_notional=traded)
    suite = BenchmarkSuite(bars["close"], initial_cash=INITIAL_CASH)
    warnings = benchmark_warnings(impact.net, suite_metrics(suite))

    folds = walk_forward_folds(
        bars["date"], train_size=120, test_size=20, purge=2, embargo=1
    )
    assert_no_overlap(folds)

    report = render_report(
        strategy_name="momentum(20) 合成 A 股序列",
        impact=impact,
        benchmark_table=suite.table(net.equity_curve["equity"]),
        folds=folds,
        traded_notional=traded,
        extra_notes={
            "样本": f"{len(bars)} 个交易日（合成，seed={SEED}）",
            "成交笔数": f"{len(net.trades)} 笔，拒单 {len(net.rejected)} 笔",
            "拒单原因统计": str(pd.Series([r["reason"] for r in net.rejected]).value_counts().to_dict()),
            "晋级门": "；".join(failing_gates(impact)) or "通过",
            "基准警示": "；".join(warnings) or "无",
        },
    )

    out = Path("docs/SAMPLE-REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"wrote {out}")
    print(f"gross return {impact.gross.total_return:+.2%} | net return {impact.net.total_return:+.2%}")
    print(f"cost consumed {impact.return_consumed:.2%} ({impact.consumed_share:.0%} of gross)")
    print(f"trades {len(net.trades)} | rejected {len(net.rejected)} | folds {len(folds)}")
    gates = failing_gates(impact)
    print("gates:", gates or "pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
