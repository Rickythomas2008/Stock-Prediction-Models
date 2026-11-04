"""Real-data walk-forward evaluation: real A-share bars through the decision chain.

Closes TECH-DEBT #7 ("every strategy ranking came from a synthetic random walk"):
the same decision chain the demos use — ``load_daily_bars`` -> ``Pipeline.decide``
(momentum alpha -> top-N constructor -> risk gate) -> ``run_backtest`` (A-share
rules + costs) — is driven by **real** akshare bars, once per walk-forward fold.

Protocol:
  * universe   : 20 names across SH / STAR / SZ / ChiNext
  * folds      : walk-forward, train ``--train`` / test ``--test`` sessions,
                 purge 2 + embargo 1, no fold overlapping another
  * decision   : at each fold, score on bars up to the session *before* the test
                 window, size with capital reset to ``--capital``, place the
                 approved orders on the first test session
  * hold       : to the end of the fold (no intra-fold rebalancing — a decision
                 per fold keeps the accounting exact)
  * prices     : 前复权 (qfq) for both signal and fill, so dividends do not show
                 up as fake jumps; disclosed as a limitation, not hidden

Read-only w.r.t. the repo except the report. Bars are cached under ``data/cache``.

Usage:
    python tools/real_data_eval.py                      # 3 years, writes the report
    python tools/real_data_eval.py --days 500 --train 80 --test 20

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from stockpred.backtest import CostModel, Order, run_backtest
from stockpred.config import Settings
from stockpred.data.cache import ParquetCache
from stockpred.data.loader import load_daily_bars
from stockpred.data.providers import AkshareProvider
from stockpred.eval import (
    Fold,
    benchmark_warnings,
    compute_metrics,
    cost_impact,
    failing_gates,
    metrics_to_dict,
    render_report,
    walk_forward_folds,
)
from stockpred.eval.walkforward import assert_no_overlap
from stockpred.pipeline import MomentumAlpha, Pipeline, TopNConstructor
from stockpred.risk import RiskGate, RiskLimits, RiskState

UNIVERSE: tuple[str, ...] = (
    "600519",
    "601318",
    "600036",
    "600000",
    "601899",
    "600276",
    "601012",
    "603259",
    "688981",
    "688111",
    "000001",
    "000002",
    "000651",
    "000858",
    "002415",
    "002594",
    "002230",
    "300750",
    "300059",
    "300124",
)

LOOKBACK = 20
TOP_N = 4
DEFAULT_CAPITAL = 100_000.0
ZERO_COST = replace(
    CostModel(),
    commission_rate=0.0,
    min_commission=0.0,
    stamp_duty_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_bps=0.0,
)


def build_pipeline(capital: float) -> Pipeline:
    return Pipeline(
        MomentumAlpha(lookback=LOOKBACK),
        TopNConstructor(top_n=TOP_N, max_position=0.40),
        gate=RiskGate(
            RiskLimits(
                initial_capital=capital,
                max_position_pct=0.40,
                max_single_order_pct=0.40,
                max_gross_exposure_pct=1.0,
                max_turnover_per_month_pct=200.0,
                max_orders_per_day=10,
                min_cash_buffer_pct=0.01,
            )
        ),
        capital=capital,
    )


def load_universe(days: int, adjust: str) -> dict[str, pd.DataFrame]:
    end = date.today()
    start = end - timedelta(days=days)
    cache = ParquetCache(Settings().cache_root)
    provider = AkshareProvider()
    bars: dict[str, pd.DataFrame] = {}
    for symbol in UNIVERSE:
        try:
            result = load_daily_bars(
                symbol, start, end, provider=provider, cache=cache, adjust=adjust
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the run
            print(f"  {symbol}: skipped ({type(exc).__name__}: {exc})")
            continue
        bars[symbol] = result.bars
        print(
            f"  {symbol}: {len(result.bars)} bars {result.bars['date'].iloc[0].date()}..{result.last_bar_date}"
        )
    if len(bars) < TOP_N + 1:
        raise SystemExit(f"only {len(bars)} symbols loaded; need at least {TOP_N + 1}")
    return bars


def run_fold(
    bars: dict[str, pd.DataFrame],
    fold: Fold,
    *,
    previous_session: date,
    capital: float,
) -> dict[str, object]:
    """One decision + one held position per fold; returns the fold's curves."""
    prices = {}
    for symbol, frame in bars.items():
        history = frame[frame["date"] <= pd.Timestamp(previous_session)]
        if not history.empty:
            prices[symbol] = float(history["close"].iloc[-1])

    pipeline = build_pipeline(capital)
    state = RiskState(cash=capital)
    try:
        package, plan = pipeline.decide(
            bars,
            as_of=previous_session,
            bar_date=previous_session,
            prices=prices,
            state=state,
        )
    except Exception as exc:  # noqa: BLE001 - a fold with no decision is data, not a crash
        return {"fold": fold, "orders": 0, "refused": 0, "reason": f"{type(exc).__name__}: {exc}"}

    orders = [
        Order(fold.test_start.date(), proposal.symbol, proposal.side, proposal.shares)
        for proposal in plan.approved
    ]
    refused = len(plan.proposals) - len(orders)

    test_start, test_end = fold.test_start.date(), fold.test_end.date()
    window = {
        symbol: frame[(frame["date"].dt.date >= test_start) & (frame["date"].dt.date <= test_end)]
        for symbol, frame in bars.items()
    }
    net = run_backtest(window, orders, initial_cash=capital, cost_model=CostModel(slippage_bps=5.0))
    gross = run_backtest(window, orders, initial_cash=capital, cost_model=ZERO_COST)
    return {
        "fold": fold,
        "orders": len(orders),
        "refused": refused,
        "refusals": [str(item.get("reason", "")) for item in net.rejected],
        "dropped": sorted(package.dropped) if package.dropped else [],
        "net": net,
        "gross": gross,
        "traded": sum(trade.notional for trade in net.trades),
    }


def stitch(runs: list[dict[str, object]], key: str, capital: float) -> pd.Series:
    """Chain per-fold equity curves into one continuous, date-indexed curve.

    ``BacktestResult.equity_curve`` carries a positional index with the session in
    a ``date`` column, so the index has to be rebuilt before anything can be
    aligned against bars.
    """
    pieces: list[pd.Series] = []
    running = capital
    for run in runs:
        result = run.get(key)
        if result is None:
            continue
        frame = result.equity_curve  # type: ignore[union-attr]
        curve = frame.set_index(pd.to_datetime(frame["date"]))["equity"].astype(float)
        multiplier = float(curve.iloc[-1]) / capital
        pieces.append(curve * (running / capital))
        running *= multiplier
    if not pieces:
        raise SystemExit("no fold produced an equity curve")
    return pd.concat(pieces)


def equal_weight_benchmark(
    bars: dict[str, pd.DataFrame], index: pd.Index, capital: float
) -> pd.Series:
    """Buy-and-hold the universe equal-weight on the same sessions as the strategy.

    The strategy curve is indexed by ``datetime.date`` objects, the bar frames by
    ``Timestamp``; reindexing without normalising both yields an all-NaN join and
    a benchmark that silently reads 0.00%.
    """
    closes = pd.DataFrame(
        {
            symbol: frame.set_index(pd.to_datetime(frame["date"]))["close"].astype(float)
            for symbol, frame in bars.items()
        }
    ).sort_index()
    returns = closes.pct_change().mean(axis=1).fillna(0.0)
    target = pd.DatetimeIndex(pd.to_datetime(list(index)))
    aligned = returns.reindex(target).fillna(0.0)
    curve = capital * (1.0 + aligned).cumprod()
    curve.index = index
    return curve.rename("buy_and_hold")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095, help="calendar days of history")
    parser.add_argument("--train", type=int, default=120, help="train sessions per fold")
    parser.add_argument("--test", type=int, default=20, help="test sessions per fold")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--adjust", default="qfq", help="price adjustment: qfq / hfq / ''")
    parser.add_argument("--report", default="docs/REAL-DATA-EVAL.md")
    args = parser.parse_args(argv)

    print(f"loading {len(UNIVERSE)} symbols over {args.days} days (adjust={args.adjust!r})")
    bars = load_universe(args.days, args.adjust)
    sessions = sorted({d.date() for frame in bars.values() for d in frame["date"]})
    print(f"universe {len(bars)} symbols | {len(sessions)} sessions")

    folds = walk_forward_folds(
        sessions, train_size=args.train, test_size=args.test, purge=2, embargo=1
    )
    assert_no_overlap(folds)
    print(f"folds {len(folds)}: {folds[0].describe()} ... {folds[-1].describe()}")

    runs: list[dict[str, object]] = []
    for fold in folds:
        previous = max(d for d in sessions if d < fold.test_start.date())
        run = run_fold(bars, fold, previous_session=previous, capital=args.capital)
        runs.append(run)
        curve = run.get("net")
        if curve is None:
            print(f"  {fold.describe()} -> skipped: {run['reason']}")
            continue
        end_equity = float(curve.equity_curve["equity"].iloc[-1])
        print(
            f"  {fold.describe()} orders={run['orders']} refused={run['refused']} "
            f"end={end_equity:,.0f} ({end_equity / args.capital - 1:+.2%})"
        )

    net_curve = stitch(runs, "net", args.capital)
    gross_curve = stitch(runs, "gross", args.capital)
    traded = sum(float(run.get("traded", 0.0)) for run in runs)
    impact = cost_impact(gross_curve, net_curve, traded_notional=traded)

    benchmark = equal_weight_benchmark(bars, net_curve.index, args.capital)
    if benchmark.nunique() <= 1:
        raise SystemExit(
            "benchmark curve is flat: strategy/benchmark index alignment broke — "
            "refusing to write a report that would show a 0.00% buy-and-hold"
        )
    cash_curve = pd.Series(args.capital, index=net_curve.index, dtype=float).rename("cash")
    curves = {"strategy_net": net_curve, "buy_and_hold": benchmark, "cash": cash_curve}
    table = pd.DataFrame(
        {
            label: metrics_to_dict(
                compute_metrics(curve, traded_notional=traded if label == "strategy_net" else 0.0)
            )
            for label, curve in curves.items()
        }
    ).T
    # 逐笔盈亏没有建模（引擎只记成交，不记平仓），一列恒为 0.00% 的 win_rate 只会误导。
    table = table.drop(columns=["win_rate"])

    bench_metrics = {
        "buy_and_hold": compute_metrics(benchmark),
        "cash": compute_metrics(cash_curve),
    }
    refusals = pd.Series(
        [reason for run in runs for reason in (run.get("refusals") or [])], dtype=object
    )
    dropped = sorted({symbol for run in runs for symbol in (run.get("dropped") or [])})
    scored = [run for run in runs if run.get("net") is not None]
    positive = sum(
        1
        for run in scored
        if float(run["net"].equity_curve["equity"].iloc[-1]) > args.capital  # type: ignore[union-attr]
    )

    report = render_report(
        strategy_name=f"横截面动量({LOOKBACK}) top{TOP_N}，真实 A 股日线（{args.adjust or 'raw'}）",
        impact=impact,
        benchmark_table=table,
        folds=folds,
        traded_notional=traded,
        extra_notes={
            "数据源": f"akshare（{len(bars)} 标的，{sessions[0]}..{sessions[-1]}，adjust={args.adjust!r}）",
            "折数": f"{len(folds)} 折，每折 1 次决策、持有到折末（train {args.train} / test {args.test}，purge 2 / embargo 1）",
            "成交": f"下单 {sum(int(run.get('orders', 0)) for run in runs)} 笔，"
            f"风控拒单 {sum(int(run.get('refused', 0)) for run in runs)} 笔",
            "折胜率": f"{positive}/{len(scored)} 折为正（{positive / max(len(scored), 1):.0%}）；"
            "逐笔胜率未建模（引擎不记平仓盈亏）",
            "撮合拒单原因（引擎）": str(refusals.value_counts().to_dict())
            if not refusals.empty
            else "无",
            "整手约束丢弃": "、".join(dropped) if dropped else "无",
            "晋级门": "；".join(failing_gates(impact)) or "通过",
            "基准警示": "；".join(benchmark_warnings(impact.net, bench_metrics)) or "无",
            "局限": "前复权价格用于撮合（成本按复权价计算）；每折资金重置；"
            "仅日频、仅做多、单折不调仓——结论是「这条链在真实数据上能不能跑」，不是策略定论",
        },
    )
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"wrote {out}")
    print(
        f"net {impact.net.total_return:+.2%} | gross {impact.gross.total_return:+.2%} | "
        f"net sharpe {impact.net.sharpe:.2f} | max dd {impact.net.max_drawdown:.2%}"
    )
    print(f"gates: {failing_gates(impact) or 'pass'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
