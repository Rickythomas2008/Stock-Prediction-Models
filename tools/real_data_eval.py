"""Real-data walk-forward evaluation: real A-share bars through the decision chain.

The same decision chain the demos use — ``load_daily_bars`` -> ``Pipeline.decide``
(momentum alpha -> buffered top-N -> risk gate) -> ``run_backtest`` (A-share rules
+ costs) — driven by real bars, benchmarked against a **real index** (沪深300 by
default), because the objective is beating the index, not merely having a
prettier equity curve.

Accounting protocol:

``continuous`` (default)
    one account: cash and holdings carry from rebalance to rebalance, sized off
    live equity, orders are target-vs-held diffs.

``reset``
    every period restarts from ``--capital`` with an empty book. Kept only as a
    control: it neither pays the sell side nor carries positions, so it understates
    both turnover and cost.

Turnover control (alpha untouched): a wide book (``--top``) so one switch is a
small slice of equity, a stay band (``--buffer``) so incumbents are only displaced
by clearly better names, and a no-trade band (``--band``) so weight drift is not
traded.

Read-only w.r.t. the repo except the report; bars are cached per provider.

Usage:
    python tools/real_data_eval.py                          # baostock, last 3 years
    python tools/real_data_eval.py --provider akshare
    python tools/real_data_eval.py --end 2023-10-31 --days 1095

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stockpred.backtest import CostModel, Order, run_backtest
from stockpred.config import Settings
from stockpred.data.cache import ParquetCache
from stockpred.data.loader import load_daily_bars
from stockpred.data.providers import (
    AkshareIndexProvider,
    AkshareProvider,
    BaostockProvider,
)
from stockpred.data.schema import validate_bars
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
from stockpred.pipeline import (
    BufferedTopNConstructor,
    MomentumAlpha,
    Pipeline,
    TiltedIndexConstructor,
)
from stockpred.risk import Position, RiskGate, RiskLimits, RiskState
from stockpred.signals.factors import LowVolatilityAlpha, ReversalAlpha

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

BENCHMARK_INDEX = "000300"  # 沪深300
LOOKBACK = 20
#: 12 names keeps each position near 8% of the book, so one name switch costs ~17%
#: of equity instead of the 50% a 4-name book pays. That is the turnover lever.
TOP_N = 12
DEFAULT_BUFFER = 4
DEFAULT_BAND = 0.02
DEFAULT_CAPITAL = 100_000.0
DEFAULT_TEST = 20
DEFAULT_TRAIN = 120
SLIPPAGE_BPS = 5.0
ZERO_COST = replace(
    CostModel(),
    commission_rate=0.0,
    min_commission=0.0,
    stamp_duty_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_bps=0.0,
)


@dataclass(frozen=True)
class Config:
    alpha: str = "momentum"
    top: int = TOP_N
    buffer: int = DEFAULT_BUFFER
    band: float = DEFAULT_BAND
    lookback: int = LOOKBACK
    test: int = DEFAULT_TEST
    mode: str = "topn"  # "topn" (active book) or "index" (tilt around benchmark weights)
    tilt: float = 0.0

    def describe(self) -> str:
        if self.mode == "index":
            return f"{self.alpha}({self.lookback}) index-tilt{self.tilt:g} test{self.test}"
        return (
            f"{self.alpha}({self.lookback}) top{self.top} band{self.band:.0%} "
            f"stay{self.buffer} test{self.test}"
        )


@dataclass
class WindowRun:
    folds: list[Fold]
    runs: list[dict[str, Any]]
    net: pd.Series
    gross: pd.Series
    traded: float


def stock_provider(name: str):
    if name == "baostock":
        return BaostockProvider()
    return AkshareProvider()


def index_provider(name: str):
    if name == "baostock":
        return BaostockProvider(is_index=True)
    return AkshareIndexProvider()


ALPHAS: tuple[str, ...] = ("momentum", "reversal", "low_vol")


def make_alpha(name: str, lookback: int):
    """Alpha by name; all three are price-only so comparisons share one data path."""
    if name == "reversal":
        return ReversalAlpha(lookback=lookback)
    if name == "low_vol":
        return LowVolatilityAlpha(lookback=lookback)
    return MomentumAlpha(lookback=lookback)


def build_pipeline(
    capital: float, config: Config, weights: dict[str, float] | None = None
) -> Pipeline:
    alpha = make_alpha(config.alpha, config.lookback)
    if config.mode == "index":
        if not weights:
            raise SystemExit("index mode needs benchmark weights")
        constructor = TiltedIndexConstructor(
            benchmark_weights=weights, tilt=config.tilt, max_position=0.15
        )
        limits = RiskLimits(
            initial_capital=capital,
            max_position_pct=0.15,
            max_single_order_pct=0.15,
            max_gross_exposure_pct=1.0,
            max_turnover_per_month_pct=200.0,
            max_orders_per_day=len(weights) + 5,
            min_cash_buffer_pct=0.01,
        )
        return Pipeline(alpha, constructor, gate=RiskGate(limits), capital=capital)

    return Pipeline(
        alpha,
        BufferedTopNConstructor(top_n=config.top, buffer=config.buffer, max_position=0.40),
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


def load_universe(
    days: int, adjust: str, provider_name: str, end: date | None = None
) -> dict[str, pd.DataFrame]:
    end = end or date.today()
    start = end - timedelta(days=days)
    cache = ParquetCache(Settings().cache_root)
    provider = stock_provider(provider_name)
    bars: dict[str, pd.DataFrame] = {}
    for symbol in UNIVERSE:
        for attempt in range(3):
            try:
                result = load_daily_bars(
                    symbol, start, end, provider=provider, cache=cache, adjust=adjust
                )
            except Exception as exc:  # noqa: BLE001 - vendor endpoints flake; retry then skip
                if attempt == 2:
                    print(f"  {symbol}: skipped ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            bars[symbol] = result.bars
            break
    print(f"  loaded {len(bars)}/{len(UNIVERSE)} symbols from {provider_name}")
    if len(bars) < TOP_N + 1:
        raise SystemExit(f"only {len(bars)} symbols loaded; need at least {TOP_N + 1}")
    return bars


def load_benchmark(
    code: str, bars: dict[str, pd.DataFrame], provider_name: str, adjust: str
) -> pd.Series:
    frame = pd.concat([frame[["date", "close"]] for frame in bars.values()], ignore_index=True)
    start = pd.to_datetime(frame["date"]).min().date()
    end = pd.to_datetime(frame["date"]).max().date()
    raw = index_provider(provider_name).fetch_daily(code, start, end)
    index_bars = validate_bars(raw, symbol=code)
    return index_bars.set_index(pd.to_datetime(index_bars["date"]))["close"].astype(float)


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
    config: Config,
    protocol: str,
    books: dict[str, dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One rebalance: decide, fill through the engine, roll both books forward.

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

    pipeline = build_pipeline(capital, config, weights)
    try:
        package, plan = pipeline.decide(
            bars,
            as_of=previous_session,
            bar_date=previous_session,
            prices=prices,
            state=state,
            capital=budget,
            min_trade_notional=config.band * budget,
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


def run_window(
    bars: dict[str, pd.DataFrame],
    sessions: list[date],
    config: Config,
    *,
    capital: float = DEFAULT_CAPITAL,
    protocol: str = "continuous",
    train: int = DEFAULT_TRAIN,
    window: tuple[date, date] | None = None,
    weights: dict[str, float] | None = None,
) -> WindowRun:
    """Walk forward over ``sessions``, optionally restricted to a date window."""
    folds = walk_forward_folds(
        sessions, train_size=train, test_size=config.test, purge=2, embargo=0
    )
    assert_no_overlap(folds)
    if window is not None:
        folds = [f for f in folds if window[0] <= f.test_start.date() <= window[1]]
        if not folds:
            raise SystemExit(f"no fold starts inside {window[0]}..{window[1]}")

    books: dict[str, dict[str, Any]] = {
        "net": {"cash": capital, "positions": {}, "peak": capital},
        "gross": {"cash": capital, "positions": {}, "peak": capital},
    }
    runs: list[dict[str, Any]] = []
    for fold in folds:
        previous = max(d for d in sessions if d < fold.test_start.date())
        runs.append(
            run_period(
                bars,
                fold,
                previous_session=previous,
                capital=capital,
                config=config,
                protocol=protocol,
                books=books,
                weights=weights,
            )
        )
    return WindowRun(
        folds=folds,
        runs=runs,
        net=stitch(runs, "net", capital, protocol),
        gross=stitch(runs, "gross", capital, protocol),
        traded=sum(float(run.get("traded", 0.0)) for run in runs),
    )


def _curve_on(series: pd.Series, index: pd.Index, capital: float) -> pd.Series:
    """Rebase a close series onto the strategy's session index."""
    target = pd.DatetimeIndex(pd.to_datetime(list(index)))
    aligned = series.reindex(target).ffill().bfill()
    returns = aligned.pct_change().fillna(0.0)
    curve = capital * (1.0 + returns).cumprod()
    curve.index = index
    return curve


def equal_weight_curve(bars: dict[str, pd.DataFrame], index: pd.Index, capital: float) -> pd.Series:
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
    return curve


def summarise(result: WindowRun, capital: float) -> dict[str, float]:
    impact = cost_impact(result.gross, result.net, traded_notional=result.traded)
    return {
        "net": impact.net.total_return,
        "gross": impact.gross.total_return,
        "sharpe": impact.net.sharpe,
        "max_dd": impact.net.max_drawdown,
        "turnover_total": impact.net.turnover,
        "turnover_year": impact.net.turnover / max(impact.net.n_days / 242.0, 1e-9),
        "days": float(impact.net.n_days),
        "impact_net": impact.net,  # type: ignore[dict-item]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095, help="calendar days of history")
    parser.add_argument("--end", default=None, help="window end (YYYY-MM-DD); default today")
    parser.add_argument("--provider", choices=["baostock", "akshare"], default="baostock")
    parser.add_argument("--benchmark", default=BENCHMARK_INDEX, help="index code, e.g. 000300")
    parser.add_argument("--alpha", choices=list(ALPHAS), default="momentum")
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST, help="sessions per period")
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER)
    parser.add_argument("--band", type=float, default=DEFAULT_BAND)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--adjust", default="qfq", help="price adjustment: qfq / hfq / ''")
    parser.add_argument("--protocol", choices=["continuous", "reset"], default="continuous")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    config = Config(
        alpha=args.alpha,
        top=args.top,
        buffer=args.buffer,
        band=args.band,
        lookback=args.lookback,
        test=args.test,
    )
    window_end = date.fromisoformat(args.end) if args.end else None
    print(f"loading {len(UNIVERSE)} symbols over {args.days} days from {args.provider}")
    bars = load_universe(args.days, args.adjust, args.provider, window_end)
    sessions = sorted({d.date() for frame in bars.values() for d in frame["date"]})
    index_closes = load_benchmark(args.benchmark, bars, args.provider, args.adjust)
    print(
        f"universe {len(bars)} symbols | {len(sessions)} sessions | "
        f"benchmark {args.benchmark} ({len(index_closes)} bars)"
    )

    result = run_window(
        bars,
        sessions,
        config,
        capital=args.capital,
        protocol=args.protocol,
        train=args.train,
    )
    print(
        f"protocol={args.protocol} {config.describe()} | {len(result.folds)} periods "
        f"| OOS {result.folds[0].test_start.date()}..{result.folds[-1].test_end.date()}"
    )
    impact = cost_impact(result.gross, result.net, traded_notional=result.traded)

    index_curve = _curve_on(index_closes, result.net.index, args.capital).rename("benchmark_index")
    ew_curve = equal_weight_curve(bars, result.net.index, args.capital).rename("equal_weight")
    cash_curve = pd.Series(args.capital, index=result.net.index, dtype=float).rename("cash")
    curves = {
        "strategy_net": result.net,
        "benchmark_index": index_curve,
        "equal_weight": ew_curve,
        "cash": cash_curve,
    }
    table = pd.DataFrame(
        {
            label: metrics_to_dict(
                compute_metrics(
                    curve, traded_notional=result.traded if label == "strategy_net" else 0.0
                )
            )
            for label, curve in curves.items()
        }
    ).T
    # 引擎不记平仓盈亏，一列恒为 0.00% 的 win_rate 只会误导。
    table = table.drop(columns=["win_rate"])

    index_metrics = compute_metrics(index_curve)
    excess = impact.net.total_return - index_metrics.total_return
    scored = [run for run in result.runs if run.get("net") is not None]
    positive = sum(
        1
        for run in scored
        if float(run["net"].equity_curve["equity"].iloc[-1])
        > float(run["net"].equity_curve["equity"].iloc[0])
    )
    refusals = pd.Series(
        [r for run in result.runs for r in (run.get("refusals") or [])], dtype=object
    )
    dropped = sorted({s for run in result.runs for s in (run.get("dropped") or [])})

    report = render_report(
        strategy_name=(
            f"{config.alpha} {config.describe()}，真实 A 股日线"
            f"（{args.adjust or 'raw'}，{args.protocol}，{args.provider}）"
        ),
        impact=impact,
        benchmark_table=table,
        folds=result.folds,
        traded_notional=result.traded,
        extra_notes={
            "数据源": f"{args.provider}（{len(bars)} 标的，{sessions[0]}..{sessions[-1]}，"
            f"adjust={args.adjust!r}）",
            "基准": f"{args.benchmark} 指数买入持有（目标：跑赢指数，不是跑赢波动）",
            "超额收益": f"策略净 {impact.net.total_return:+.2%} − 指数 "
            f"{index_metrics.total_return:+.2%} = **{excess:+.2%}**",
            "协议": (
                "连续资金：持仓与现金跨期结转，按当期权益定仓，下单为目标-持有差额"
                if args.protocol == "continuous"
                else "每期资金重置：每期清空重建，换手与成本被系统性低估（仅作对照）"
            ),
            "调仓": f"每 {config.test} 个交易日一次，共 {len(result.folds)} 次"
            f"（stay band {config.buffer}，免交易带 {config.band:.0%} 权益）",
            "年化换手": f"{impact.net.turnover / max(impact.net.n_days / 242.0, 1e-9):.0%}"
            f"（总换手 {impact.net.turnover:.0%}，{impact.net.n_days} 个交易日）",
            "成交": f"下单 {sum(int(run.get('orders', 0)) for run in result.runs)} 笔，"
            f"风控拒单 {sum(int(run.get('refused', 0)) for run in result.runs)} 笔",
            "期间胜率": f"{positive}/{len(scored)} 期为正（{positive / max(len(scored), 1):.0%}）",
            "撮合拒单原因（引擎）": str(refusals.value_counts().to_dict())
            if not refusals.empty
            else "无",
            "整手约束丢弃": "、".join(dropped) if dropped else "无",
            "晋级门": "；".join(failing_gates(impact)) or "通过",
            "基准警示": "；".join(
                benchmark_warnings(
                    impact.net, {"buy_and_hold": index_metrics, "cash": compute_metrics(cash_curve)}
                )
            )
            or "无",
            "局限": "前复权价格用于撮合（成本按复权价计算）；仅日频、仅做多；"
            "训练窗只用于确定样本外起点，动量无待拟合参数",
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
        f"net {impact.net.total_return:+.2%} | index {index_metrics.total_return:+.2%} | "
        f"excess {excess:+.2%} | sharpe {impact.net.sharpe:.2f} vs {index_metrics.sharpe:.2f} | "
        f"max dd {impact.net.max_drawdown:.2%} vs {index_metrics.max_drawdown:.2%} | "
        f"turnover {impact.net.turnover / max(impact.net.n_days / 242.0, 1e-9):.0%}/yr"
    )
    print(f"gates: {failing_gates(impact) or 'pass'} | beats_index: {excess > 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
