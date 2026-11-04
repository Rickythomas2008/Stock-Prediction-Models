"""Real-data walk-forward evaluation: real A-share bars through the decision chain.

Closes TECH-DEBT #7 ("every strategy ranking came from a synthetic random walk"):
the same decision chain the demos use — ``load_daily_bars`` -> ``Pipeline.decide``
(momentum alpha -> buffered top-N -> risk gate) -> ``run_backtest`` (A-share rules
+ costs) — is driven by **real** akshare bars.

Two protocols, same signals, so the effect of the accounting is measurable:

``continuous`` (default)
    one account: cash and holdings carry from rebalance to rebalance, sized off
    live equity, orders are target-vs-held diffs. This is what a real account does.

``reset``
    every rebalance period restarts from ``--capital`` with an empty book — the
    earlier protocol. It reports the cost of churn that ``continuous`` removes.

Signals: 20-session momentum, top-N with a stay band (``--buffer``) so an
incumbent is only displaced by a clearly better name. Rebalance every ``--test``
sessions; training windows exist to pin the out-of-sample start, not to fit
parameters (momentum has none).

Prices: 前复权 (qfq) for signal and fill, so dividends do not appear as fake
jumps — disclosed as a limitation, not hidden.

Read-only w.r.t. the repo except the report. Bars are cached under ``data/cache``.

Usage:
    python tools/real_data_eval.py                      # 3 years, continuous
    python tools/real_data_eval.py --protocol reset     # before/after comparison
    python tools/real_data_eval.py --buffer 2 --test 20

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

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
from stockpred.pipeline import BufferedTopNConstructor, MomentumAlpha, Pipeline
from stockpred.risk import Position, RiskGate, RiskLimits, RiskState

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
#: 12 names keeps each position near 8% of the book, so one name switch costs ~17%
#: of equity instead of the 50% a 4-name book pays. That is the turnover lever;
#: the alpha itself is unchanged.
TOP_N = 12
DEFAULT_BUFFER = 4
DEFAULT_BAND = 0.02
DEFAULT_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5.0
ZERO_COST = replace(
    CostModel(),
    commission_rate=0.0,
    min_commission=0.0,
    stamp_duty_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_bps=0.0,
)


def build_pipeline(
    capital: float, buffer: int, lookback: int = LOOKBACK, top_n: int = TOP_N
) -> Pipeline:
    return Pipeline(
        MomentumAlpha(lookback=lookback),
        BufferedTopNConstructor(top_n=top_n, buffer=buffer, max_position=0.40),
        gate=RiskGate(
            RiskLimits(
                initial_capital=capital,
                max_position_pct=0.40,
                max_single_order_pct=0.40,
                max_gross_exposure_pct=1.0,
                max_turnover_per_month_pct=200.0,
                max_orders_per_day=16,
                min_cash_buffer_pct=0.01,
            )
        ),
        capital=capital,
    )


def load_universe(days: int, adjust: str, end: date | None = None) -> dict[str, pd.DataFrame]:
    end = end or date.today()
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
    print(f"  loaded {len(bars)}/{len(UNIVERSE)} symbols")
    if len(bars) < TOP_N + 1:
        raise SystemExit(f"only {len(bars)} symbols loaded; need at least {TOP_N + 1}")
    return bars


def last_closes(bars: dict[str, pd.DataFrame], day: date) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol, frame in bars.items():
        history = frame[frame["date"] <= pd.Timestamp(day)]
        if not history.empty:
            prices[symbol] = float(history["close"].iloc[-1])
    return prices


def shift_positions(positions: dict[str, int], result: Any) -> dict[str, int]:
    """Apply the engine's fills to a holdings map (the engine keeps its own ledger)."""
    updated = dict(positions)
    for trade in result.trades:
        delta = trade.shares if trade.side == "buy" else -trade.shares
        updated[trade.symbol] = updated.get(trade.symbol, 0) + delta
    return {symbol: shares for symbol, shares in updated.items() if shares > 0}


def run_period(
    bars: dict[str, pd.DataFrame],
    fold: Fold,
    *,
    previous_session: date,
    capital: float,
    buffer: int,
    protocol: str,
    band: float,
    books: dict[str, dict[str, Any]],
    lookback: int = LOOKBACK,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """One rebalance: decide, fill through the engine, then roll both books forward.

    Two books run on the same orders: ``net`` pays costs and drives the decisions,
    ``gross`` pays nothing. Sharing one book would make the cost comparison a
    fiction the moment the two equity paths diverge.
    """
    prices = last_closes(bars, previous_session)
    net_book, gross_book = books["net"], books["gross"]
    if protocol == "continuous":
        state = RiskState(
            cash=float(net_book["cash"]),
            positions={
                symbol: Position(symbol, shares, prices.get(symbol, 0.0))
                for symbol, shares in net_book["positions"].items()
                if prices.get(symbol)
            },
            peak_equity=float(net_book["peak"]),
        )
        budget = state.equity()
        net_opening, net_cash = dict(net_book["positions"]), float(net_book["cash"])
        gross_opening, gross_cash = dict(gross_book["positions"]), float(gross_book["cash"])
    else:
        state = RiskState(cash=capital)
        budget = capital
        net_opening, net_cash = {}, capital
        gross_opening, gross_cash = {}, capital

    pipeline = build_pipeline(capital, buffer, lookback, top_n)
    try:
        package, plan = pipeline.decide(
            bars,
            as_of=previous_session,
            bar_date=previous_session,
            prices=prices,
            state=state,
            capital=budget,
            min_trade_notional=band * budget,
        )
    except Exception as exc:  # noqa: BLE001 - a period with no decision is data, not a crash
        return {"fold": fold, "orders": 0, "refused": 0, "reason": f"{type(exc).__name__}: {exc}"}

    orders = [
        Order(fold.test_start.date(), proposal.symbol, proposal.side, proposal.shares)
        for proposal in plan.approved
    ]
    start, end = fold.test_start.date(), fold.test_end.date()
    window = {
        symbol: frame[(frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)]
        for symbol, frame in bars.items()
    }
    net = run_backtest(
        window,
        orders,
        initial_cash=net_cash,
        cost_model=CostModel(slippage_bps=SLIPPAGE_BPS),
        initial_positions=net_opening or None,
    )
    gross = run_backtest(
        window,
        orders,
        initial_cash=gross_cash,
        cost_model=ZERO_COST,
        initial_positions=gross_opening or None,
    )

    if protocol == "continuous":
        net_book["cash"] = float(net.equity_curve["cash"].iloc[-1])
        net_book["positions"] = shift_positions(net_opening, net)
        net_book["peak"] = max(float(net_book["peak"]), float(net.equity_curve["equity"].max()))
        gross_book["cash"] = float(gross.equity_curve["cash"].iloc[-1])
        gross_book["positions"] = shift_positions(gross_opening, gross)

    return {
        "fold": fold,
        "orders": len(orders),
        "refused": len(plan.proposals) - len(orders),
        "refusals": [str(item.get("reason", "")) for item in net.rejected],
        "dropped": sorted(package.dropped),
        "net": net,
        "gross": gross,
        "traded": sum(trade.notional for trade in net.trades),
    }


def stitch(runs: list[dict[str, Any]], key: str, capital: float, protocol: str) -> pd.Series:
    """Build one equity curve; continuous books are already sequential."""
    pieces: list[pd.Series] = []
    running = capital
    for run in runs:
        result = run.get(key)
        if result is None:
            continue
        frame = result.equity_curve
        curve = frame.set_index(pd.to_datetime(frame["date"]))["equity"].astype(float)
        if protocol == "continuous":
            pieces.append(curve)
        else:
            pieces.append(curve * (running / capital))
            running *= float(curve.iloc[-1]) / capital
    if not pieces:
        raise SystemExit("no period produced an equity curve")
    return pd.concat(pieces)


def equal_weight_benchmark(
    bars: dict[str, pd.DataFrame], index: pd.Index, capital: float
) -> pd.Series:
    """Buy-and-hold the universe equal-weight on the same sessions as the strategy."""
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
    parser.add_argument(
        "--end", default=None, help="window end (YYYY-MM-DD); default today — use it for a "
        "second, non-overlapping validation window"
    )
    parser.add_argument("--train", type=int, default=120, help="sessions before the OOS span")
    parser.add_argument("--test", type=int, default=20, help="sessions per rebalance period")
    parser.add_argument(
        "--buffer", type=int, default=DEFAULT_BUFFER, help="stay band for incumbents"
    )
    parser.add_argument(
        "--band", type=float, default=DEFAULT_BAND, help="no-trade band as a fraction of equity"
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="momentum lookback")
    parser.add_argument("--top", type=int, default=TOP_N, help="names held per period")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--adjust", default="qfq", help="price adjustment: qfq / hfq / ''")
    parser.add_argument("--protocol", choices=["continuous", "reset"], default="continuous")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    print(f"loading {len(UNIVERSE)} symbols over {args.days} days (adjust={args.adjust!r})")
    window_end = date.fromisoformat(args.end) if args.end else None
    bars = load_universe(args.days, args.adjust, window_end)
    sessions = sorted({d.date() for frame in bars.values() for d in frame["date"]})
    print(f"universe {len(bars)} symbols | {len(sessions)} sessions")

    folds = walk_forward_folds(
        sessions, train_size=args.train, test_size=args.test, purge=2, embargo=0
    )
    assert_no_overlap(folds)
    print(
        f"protocol={args.protocol} top={args.top} buffer={args.buffer} band={args.band:.0%} "
        f"lookback={args.lookback} | {len(folds)} rebalance periods "
        f"| OOS {folds[0].test_start.date()}..{folds[-1].test_end.date()}"
    )

    books: dict[str, dict[str, Any]] = {
        "net": {"cash": args.capital, "positions": {}, "peak": args.capital},
        "gross": {"cash": args.capital, "positions": {}, "peak": args.capital},
    }
    runs: list[dict[str, Any]] = []
    for fold in folds:
        previous = max(d for d in sessions if d < fold.test_start.date())
        run = run_period(
            bars,
            fold,
            previous_session=previous,
            capital=args.capital,
            buffer=args.buffer,
            protocol=args.protocol,
            band=args.band,
            books=books,
            lookback=args.lookback,
            top_n=args.top,
        )
        runs.append(run)
        result = run.get("net")
        if result is None:
            print(f"  {fold.index}: skipped ({run['reason']})")
            continue
        end_equity = float(result.equity_curve["equity"].iloc[-1])
        start_equity = float(result.equity_curve["equity"].iloc[0])
        print(
            f"  {fold.index}: orders={run['orders']} refused={run['refused']} "
            f"period {end_equity - start_equity:+,.0f} -> {end_equity:,.0f}"
        )

    net_curve = stitch(runs, "net", args.capital, args.protocol)
    gross_curve = stitch(runs, "gross", args.capital, args.protocol)
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
    # 引擎不记平仓盈亏，一列恒为 0.00% 的 win_rate 只会误导。
    table = table.drop(columns=["win_rate"])

    scored = [run for run in runs if run.get("net") is not None]
    positive = sum(
        1
        for run in scored
        if float(run["net"].equity_curve["equity"].iloc[-1])
        > float(run["net"].equity_curve["equity"].iloc[0])
    )
    refusals = pd.Series(
        [reason for run in runs for reason in (run.get("refusals") or [])], dtype=object
    )
    dropped = sorted({symbol for run in runs for symbol in (run.get("dropped") or [])})
    bench_metrics = {
        "buy_and_hold": compute_metrics(benchmark),
        "cash": compute_metrics(cash_curve),
    }

    report = render_report(
        strategy_name=(
            f"横截面动量({args.lookback}) top{args.top} stay-band={args.buffer}，"
            f"真实 A 股日线（{args.adjust or 'raw'}，{args.protocol}）"
        ),
        impact=impact,
        benchmark_table=table,
        folds=folds,
        traded_notional=traded,
        extra_notes={
            "数据源": f"akshare（{len(bars)} 标的，{sessions[0]}..{sessions[-1]}，adjust={args.adjust!r}）",
            "协议": (
                "连续资金：持仓与现金跨期结转，按当期权益定仓，下单为目标-持有差额"
                if args.protocol == "continuous"
                else "每期资金重置：每期清空重建，换手与成本被系统性放大"
            ),
            "调仓": f"每 {args.test} 个交易日一次，共 {len(folds)} 次"
            f"（stay band {args.buffer}，免交易带 {args.band:.0%} 权益）",
            "年化换手": f"{impact.net.turnover / max(impact.net.n_days / 242.0, 1e-9):.0%}"
            f"（总换手 {impact.net.turnover:.0%}，{impact.net.n_days} 个交易日）",
            "成交": f"下单 {sum(int(run.get('orders', 0)) for run in runs)} 笔，"
            f"风控拒单 {sum(int(run.get('refused', 0)) for run in runs)} 笔",
            "期间胜率": f"{positive}/{len(scored)} 期为正（{positive / max(len(scored), 1):.0%}）",
            "撮合拒单原因（引擎）": str(refusals.value_counts().to_dict())
            if not refusals.empty
            else "无",
            "整手约束丢弃": "、".join(dropped) if dropped else "无",
            "晋级门": "；".join(failing_gates(impact)) or "通过",
            "基准警示": "；".join(benchmark_warnings(impact.net, bench_metrics)) or "无",
            "局限": "前复权价格用于撮合（成本按复权价计算）；仅日频、仅做多；"
            "训练窗只用于确定样本外起点，动量无待拟合参数——"
            "结论是「这条链在真实数据上是否成立」，不是策略定论",
        },
    )
    out = Path(
        args.report
        or (
            "docs/REAL-DATA-EVAL.md"
            if args.protocol == "continuous"
            else f"docs/REAL-DATA-EVAL-{args.protocol}.md"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"wrote {out}")
    print(
        f"net {impact.net.total_return:+.2%} | gross {impact.gross.total_return:+.2%} | "
        f"net sharpe {impact.net.sharpe:.2f} | max dd {impact.net.max_drawdown:.2%} | "
        f"turnover {impact.net.turnover:.0%}"
    )
    print(f"buy_and_hold {bench_metrics['buy_and_hold'].total_return:+.2%}")
    print(f"gates: {failing_gates(impact) or 'pass'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
