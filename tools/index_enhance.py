"""Index enhancement: tilt inside the benchmark, then measure tracking error and IR.

This replaces "pick 12 of 20 names and hope" with the product the objective actually
asks for: stay inside an index, deviate deliberately, and report how much tracking
risk that deviation cost.

Method:
  * pool    = 沪深300 constituents, ranked by index weight (``--pool`` largest)
  * weights = index weights; ``w_i ∝ bench_i · exp(tilt · z_i)`` where ``z`` is the
              cross-sectional z-score of the alpha
  * tilt = 0 reproduces the benchmark, so the same harness measures both ends
  * rebalance every ``--test`` sessions, continuous capital, same A-share frictions
    and the same no-trade band as the active book

Reported per tilt: net return, active return vs index, **tracking error**
(annualised stdev of daily active returns) and **information ratio**.

Limitations, stated up front: one weight snapshot is used for the whole window
(index rebalances are ignored), the pool is the largest N constituents rather than
all 300, and only the tilts listed are tried.

Usage:
    python tools/index_enhance.py --pool 50 --days 1095
    python tools/index_enhance.py --provider akshare --tilts 0,0.25,0.5,1.0

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_data_eval as rde  # noqa: E402

from stockpred.data.adjust import forward_adjusted  # noqa: E402
from stockpred.data.dividends import load_akshare_actions  # noqa: E402
from stockpred.data.schema import validate_bars  # noqa: E402
from stockpred.eval import compute_metrics  # noqa: E402

TRADING_DAYS = 242


def _find(columns, *needles: str, exclude: tuple[str, ...] = ()):
    """Column lookup by substring; ``exclude`` guards against 指数代码 matching 代码."""
    for column in columns:
        text = str(column)
        if all(needle in text for needle in needles) and not any(
            blocked in text for blocked in exclude
        ):
            return column
    return None


def constituents(provider: str, pool: int) -> dict[str, float]:
    """Index weights, largest first. Falls back to equal weight if weights fail."""
    if provider == "akshare":
        for function_name, has_weights in (
            ("index_stock_cons_weight_csindex", True),
            ("index_stock_cons_csindex", False),
        ):
            try:
                import akshare as ak  # noqa: PLC0415

                frame = getattr(ak, function_name)(symbol=rde.BENCHMARK_INDEX)
                code_col = _find(frame.columns, "成分券代码") or _find(
                    frame.columns, "代码", exclude=("指数",)
                )
                if code_col is None:
                    raise ValueError(f"no constituent code column in {list(frame.columns)}")
                if has_weights:
                    weight_col = _find(frame.columns, "权重")
                    if weight_col is None:
                        raise ValueError(f"no weight column in {list(frame.columns)}")
                    weights = {
                        str(row[code_col]).zfill(6): float(row[weight_col]) / 100.0
                        for _, row in frame.iterrows()
                    }
                else:
                    codes = [str(value).zfill(6) for value in frame[code_col]]
                    weights = {code: 1.0 / len(codes) for code in codes}
                ranked = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:pool])
                print(
                    f"  index weights via {function_name}: {len(weights)} constituents, "
                    f"using top {len(ranked)}"
                )
                return ranked
            except Exception as exc:  # noqa: BLE001 - fall back, do not fail the run
                print(f"  {function_name} failed ({type(exc).__name__}: {str(exc)[:60]})")

    if provider == "baostock":
        try:
            import baostock as bs  # noqa: PLC0415

            bs.login()
            try:
                result = bs.query_hs300_stocks()
                codes = []
                while result.error_code == "0" and result.next():
                    row = dict(zip(result.fields, result.get_row_data(), strict=False))
                    codes.append(row["code"].split(".")[-1])
            finally:
                bs.logout()
            if codes:
                print(f"  hs300 list: {len(codes)} names, equal weight (no weight feed)")
                chosen = codes[:pool]
                return {code: 1.0 / len(chosen) for code in chosen}
        except Exception as exc:  # noqa: BLE001
            print(f"  hs300 list unavailable: {exc}")

    raise SystemExit("cannot obtain index constituents from this provider")


def load_pool_bars(
    weights: dict[str, float],
    days: int,
    provider: str,
    end: date,
    *,
    self_adjust: bool,
    capital: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    start = end - timedelta(days=days)
    prov = rde.stock_provider(provider)
    cache_dir = Path("data/cache")
    bars: dict[str, pd.DataFrame] = {}
    for symbol in weights:
        try:
            raw = validate_bars(prov.fetch_daily(symbol, start, end, adjust=""), symbol=symbol)
            if self_adjust:
                actions = load_akshare_actions(symbol, cache_dir=cache_dir)
                raw = forward_adjusted(raw, actions)
            bars[symbol] = raw
        except Exception:  # noqa: BLE001 - one missing name must not sink the pool
            continue
    print(f"  loaded {len(bars)}/{len(weights)} constituents")
    if len(bars) < 10:
        raise SystemExit(f"only {len(bars)} constituents loaded; too few for an index book")

    # A 100-share lot of 茅台 costs more than a 5% slice of 1M: the book cannot hold
    # every constituent, so both the strategy and its reference start from the names
    # that are actually affordable — otherwise tilt=0 would show tracking error that
    # is purely a lot-size artefact.
    affordable: dict[str, float] = {}
    for symbol, weight in weights.items():
        if symbol not in bars:
            continue
        lot_cost = 100.0 * float(bars[symbol]["close"].iloc[-1])
        if lot_cost > capital * weight:
            continue
        affordable[symbol] = weight
    dropped = sorted(set(weights) - set(affordable))
    if dropped:
        print(f"  dropped {len(dropped)} name(s) whose one lot exceeds the target slice: {dropped}")
    total = sum(affordable.values())
    return bars, {symbol: weight / total for symbol, weight in affordable.items()}


def scheduled_reference(
    bars: dict[str, pd.DataFrame],
    weights: dict[str, float],
    folds,
    capital: float,
) -> pd.Series:
    """Reference portfolio rebalanced on the strategy's own schedule.

    Comparing against a daily-rebalanced basket would charge the strategy for
    rebalancing monthly, which is the design, not the deviation. Same schedule and
    buy-and-hold within a period ⇒ the remaining tracking error is what the tilt cost.
    """
    pieces: list[pd.Series] = []
    running = capital
    for fold in folds:
        start, end = fold.test_start.date(), fold.test_end.date()
        closes: dict[str, pd.Series] = {}
        for symbol, frame in bars.items():
            mask = (frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)
            window = frame.loc[mask, ["date", "close"]]
            if len(window) < 2:
                continue
            closes[symbol] = window.set_index(pd.to_datetime(window["date"]))["close"].astype(float)
        if not closes:
            continue
        frame = pd.DataFrame(closes).sort_index()
        returns = frame.pct_change().fillna(0.0)
        weight_series = pd.Series(weights, dtype=float).reindex(frame.columns).fillna(0.0)
        period_returns = (returns * weight_series).sum(axis=1)
        curve = running * (1.0 + period_returns).cumprod()
        running = float(curve.iloc[-1])
        pieces.append(curve)
    if not pieces:
        raise SystemExit("reference portfolio produced no curve")
    return pd.concat(pieces).rename("reference")


def active_stats(net: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    """(active return, annualised tracking error, information ratio)."""
    aligned = pd.concat([net.rename("net"), bench.rename("bench")], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0, 0.0, 0.0
    active_daily = aligned["net"].pct_change() - aligned["bench"].pct_change()
    active_daily = active_daily.dropna()
    tracking_error = float(active_daily.std(ddof=1) * math.sqrt(TRADING_DAYS))
    active_return = float(aligned["net"].iloc[-1] / aligned["net"].iloc[0] - 1.0) - float(
        aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1.0
    )
    years = max(len(aligned) / TRADING_DAYS, 1e-9)
    annual_active = (1.0 + active_return) ** (1.0 / years) - 1.0
    ir = annual_active / tracking_error if tracking_error > 1e-12 else 0.0
    return active_return, tracking_error, ir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, default=50, help="largest N constituents to trade")
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--end", default=None)
    parser.add_argument("--provider", choices=["baostock", "akshare"], default="baostock")
    parser.add_argument("--alpha", choices=list(rde.ALPHAS), default="low_vol")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--test", type=int, default=20, help="sessions per rebalance")
    parser.add_argument("--tilts", default="0,0.25,0.5,1.0")
    parser.add_argument(
        "--band",
        type=float,
        default=0.002,
        help="no-trade band as a fraction of equity; index books need a much tighter "
        "band than a 12-name active book or the weights simply drift",
    )
    parser.add_argument("--capital", type=float, default=rde.DEFAULT_CAPITAL)
    parser.add_argument("--no-self-adjust", action="store_true", help="use vendor qfq instead")
    parser.add_argument("--report", default="docs/INDEX-ENHANCE.md")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else date.today()
    print(f"building index book for {rde.BENCHMARK_INDEX} from {args.provider}")
    weights = constituents(args.provider, args.pool)
    bars, weights = load_pool_bars(
        weights,
        args.days,
        args.provider,
        end,
        self_adjust=not args.no_self_adjust,
        capital=args.capital,
    )
    sessions = sorted({d.date() for frame in bars.values() for d in frame["date"]})
    index_closes = rde.load_benchmark(rde.BENCHMARK_INDEX, bars, args.provider, "")
    print(
        f"pool {len(bars)} names | {len(sessions)} sessions | "
        f"adjust={'self-computed qfq' if not args.no_self_adjust else 'vendor qfq'}"
    )

    rows: list[dict[str, object]] = []
    reference_seen: pd.Series | None = None
    for tilt in [float(t) for t in args.tilts.split(",") if t.strip()]:
        config = rde.Config(
            alpha=args.alpha,
            lookback=args.lookback,
            test=args.test,
            band=args.band,
            mode="index",
            tilt=tilt,
        )
        result = rde.run_window(
            bars, sessions, config, capital=args.capital, protocol="continuous", weights=weights
        )
        metrics = compute_metrics(result.net, traded_notional=result.traded)
        reference = scheduled_reference(bars, weights, result.folds, args.capital)
        reference_seen = reference
        index_curve = rde._curve_on(index_closes, result.net.index, args.capital)
        active, te, ir = active_stats(result.net, reference)
        vs_index = float(result.net.iloc[-1] / result.net.iloc[0] - 1.0) - float(
            index_curve.iloc[-1] / index_curve.iloc[0] - 1.0
        )
        years = max(metrics.n_days / TRADING_DAYS, 1e-9)
        rows.append(
            {
                "tilt": tilt,
                "net": metrics.total_return,
                "reference": float(reference.iloc[-1] / reference.iloc[0] - 1.0),
                "index": float(index_curve.iloc[-1] / index_curve.iloc[0] - 1.0),
                "active": active,
                "te": te,
                "ir": ir,
                "vs_index": vs_index,
                "sharpe": metrics.sharpe,
                "max_dd": metrics.max_drawdown,
                "turnover": metrics.turnover / years,
            }
        )
        print(
            f"  tilt={tilt:<5g} net={metrics.total_return:+7.2%} ref={rows[-1]['reference']:+.2%} "
            f"active={active:+7.2%} TE={te:6.2%} IR={ir:5.2f} "
            f"turnover={metrics.turnover / years:5.0%}/yr"
        )

    reference_te = 0.0
    reference_gap = 0.0
    if reference_seen is not None:
        index_on_ref = rde._curve_on(index_closes, reference_seen.index, args.capital)
        _, reference_te, _ = active_stats(reference_seen, index_on_ref)
        reference_gap = float(reference_seen.iloc[-1] / reference_seen.iloc[0] - 1.0) - float(
            index_on_ref.iloc[-1] / index_on_ref.iloc[0] - 1.0
        )

    lines = [
        "# 指数增强：在基准内偏离，量跟踪误差与信息比率",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 指数：{rde.BENCHMARK_INDEX}；池：{len(bars)} 只（按指数权重取前 {args.pool}）",
        f"- 增强基准 = **池按指数权重持有**（可持有的那个组合）：它与指数的跟踪误差 "
        f"{reference_te:.2%}、累计差 {reference_gap:+.2%}",
        "- ⚠ 这个差值是「池代理误差」：只取前 N 大权重会让组合变成大市值集中下注，"
        "不是指数。若该差值很大，下面的超额只能读成「相对这个池」，不能读成「相对沪深300」",
        f"- 因子：{args.alpha}({args.lookback})；调仓：每 {args.test} 个交易日；资金：{args.capital:,.0f} 元",
        f"- 价格：{'自算前复权（不复权价 + 分红送转）' if not args.no_self_adjust else '供应商前复权'}",
        "",
        "| tilt | 策略净 | 参考组合 | 指数 | 对参考超额 | 跟踪误差 | 信息比率 | 对指数超额 | Sharpe | 最大回撤 | 年化换手 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tilt']:g} | {float(row['net']):+.2%} | {float(row['reference']):+.2%} "
            f"| {float(row['index']):+.2%} | **{float(row['active']):+.2%}** "
            f"| {float(row['te']):.2%} | {float(row['ir']):.2f} | {float(row['vs_index']):+.2%} "
            f"| {float(row['sharpe']):.2f} | {float(row['max_dd']):.2%} | {float(row['turnover']):.0%} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "- `tilt=0` 是自检行：策略净应与参考组合几乎相同（差异只剩整手取整、免交易带与成本）。"
        "这一行明显偏离，说明口径有问题，不是好消息。",
        "- 判据：**对参考组合的信息比率 > 0.5，且跟踪误差在预算内（≤5%）** 才算可用的指数增强；"
        "超额为正但 IR 很低 = 承担了主动风险却没稳定换来收益。",
        "- 局限：整段用一个权重快照（忽略指数调样）、池为前 N 大权重而非全部成分、只试了列出的 tilt，"
        "且 tilt 与因子都是在同一窗口上挑的（见技术债 #20）。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
