"""Index enhancement against a **tradeable** benchmark: core ETF + tilted sleeve.

The earlier version benchmarked against 沪深300 itself, which nobody can hold — and
against a "pool at index weights", which turned out to be a large-cap bet rather
than an index (the pool drifted +69pp from the index in three years). Both made the
self-check meaningless: with weight-theoretic references, `tilt=0` could not
reproduce its own benchmark.

So the benchmark is now **510300 (沪深300ETF)**, bought and held. It is tradeable,
its lots are ~4 CNY × 100 units (rounding is negligible), and it tracks the index
by construction. The enhanced product is a core-satellite blend:

    r = (1 - alpha) * r_etf + alpha * r_sleeve(tilt)

`alpha = 0` is a pure ETF holding, so the self-check row must show ~0 active return
and ~0 tracking error **by construction** — if it does not, the accounting is broken.

Reported per (alpha, tilt): net, active vs ETF, tracking error, information ratio,
plus the sleeve's own turnover scaled by alpha.

Usage:
    python tools/index_enhance.py --pool 30 --sleeves 0,0.2,0.4 --tilt 0.5
    python tools/index_enhance.py --provider akshare --etf 510300 --days 1095

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
from stockpred.data.providers.akshare_provider import AkshareProvider  # noqa: E402
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


def load_etf(etf: str, start: date, end: date) -> pd.Series:
    """510300 daily closes (qfq) — the benchmark we can actually buy."""
    import akshare as ak  # noqa: PLC0415 - optional dependency

    raw = ak.fund_etf_hist_em(
        symbol=etf,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    frame = AkshareProvider._normalize(raw)
    frame = validate_bars(frame, symbol=etf)
    print(
        f"  etf {etf}: {len(frame)} bars {frame['date'].iloc[0].date()}..{frame['date'].iloc[-1].date()}"
    )
    return frame.set_index(pd.to_datetime(frame["date"]))["close"].astype(float)


def etf_curve(closes: pd.Series, index: pd.Index, capital: float) -> pd.Series:
    target = pd.DatetimeIndex(pd.to_datetime(list(index)))
    aligned = closes.reindex(target).ffill().bfill()
    returns = aligned.pct_change().fillna(0.0)
    curve = capital * (1.0 + returns).cumprod()
    curve.index = index
    return curve.rename("etf")


def active_stats(net: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    """(active return, annualised tracking error, information ratio)."""
    aligned = pd.concat([net.rename("net"), bench.rename("bench")], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0, 0.0, 0.0
    active_daily = (aligned["net"].pct_change() - aligned["bench"].pct_change()).dropna()
    tracking_error = float(active_daily.std(ddof=1) * math.sqrt(TRADING_DAYS))
    active_return = float(aligned["net"].iloc[-1] / aligned["net"].iloc[0] - 1.0) - float(
        aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1.0
    )
    years = max(len(aligned) / TRADING_DAYS, 1e-9)
    annual_active = (1.0 + active_return) ** (1.0 / years) - 1.0
    ir = annual_active / tracking_error if tracking_error > 1e-12 else 0.0
    return active_return, tracking_error, ir


def blend(etf: pd.Series, sleeve: pd.Series, alpha: float, capital: float) -> pd.Series:
    """(1-alpha)·ETF + alpha·sleeve. At alpha=0 this must equal the ETF curve."""
    aligned = pd.concat([etf.rename("etf"), sleeve.rename("sleeve")], axis=1).dropna()
    etf_returns = aligned["etf"].pct_change().fillna(0.0)
    sleeve_returns = aligned["sleeve"].pct_change().fillna(0.0)
    mixed = (1.0 - alpha) * etf_returns + alpha * sleeve_returns
    curve = capital * (1.0 + mixed).cumprod()
    return curve.rename("blend")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, default=30, help="largest N constituents to trade")
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--end", default=None)
    parser.add_argument("--provider", choices=["baostock", "akshare"], default="akshare")
    parser.add_argument("--etf", default="510300", help="benchmark ETF (tradeable)")
    parser.add_argument("--alpha", choices=list(rde.ALPHAS), default="low_vol")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--test", type=int, default=20, help="sessions per rebalance")
    parser.add_argument("--sleeves", default="0,0.2,0.4", help="sleeve weights (0 = pure ETF)")
    parser.add_argument("--tilt", type=float, default=0.5)
    parser.add_argument(
        "--compare-tilt", type=float, default=0.0, help="extra tilt row at max sleeve"
    )
    parser.add_argument("--capital", type=float, default=10_000_000.0)
    parser.add_argument("--band", type=float, default=0.002, help="sleeve no-trade band")
    parser.add_argument("--no-self-adjust", action="store_true", help="use vendor qfq instead")
    parser.add_argument("--report", default="docs/INDEX-ENHANCE.md")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = end - timedelta(days=args.days)
    print(f"index enhancement: benchmark {args.etf}, pool from {rde.BENCHMARK_INDEX}")
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
    etf_closes = load_etf(args.etf, start, end)
    index_closes = rde.load_benchmark(rde.BENCHMARK_INDEX, bars, args.provider, "")
    print(f"pool {len(bars)} names | {len(sessions)} sessions | sleeve capital {args.capital:,.0f}")

    sleeves = [float(value) for value in args.sleeves.split(",") if value.strip()]
    plan = [(alpha, args.tilt) for alpha in sleeves]
    max_sleeve = max(sleeves) if sleeves else 0.0
    if max_sleeve > 0 and args.compare_tilt != args.tilt:
        plan.append((max_sleeve, args.compare_tilt))

    rows: list[dict[str, object]] = []
    cache: dict[float, pd.Series] = {}
    for alpha, tilt in plan:
        if tilt not in cache:
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
            cache[tilt] = result.net
        sleeve = cache[tilt]
        benchmark = etf_curve(etf_closes, sleeve.index, args.capital)
        mixed = blend(benchmark, sleeve, alpha, args.capital)
        metrics = compute_metrics(mixed)
        active, te, ir = active_stats(mixed, benchmark)
        index_curve = rde._curve_on(index_closes, mixed.index, args.capital)
        sleeve_metrics = compute_metrics(sleeve)
        sleeve_turnover = sleeve_metrics.turnover / max(sleeve_metrics.n_days / TRADING_DAYS, 1e-9)
        rows.append(
            {
                "alpha": alpha,
                "tilt": tilt,
                "net": metrics.total_return,
                "etf": float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0),
                "index": float(index_curve.iloc[-1] / index_curve.iloc[0] - 1.0),
                "active": active,
                "te": te,
                "ir": ir,
                "sharpe": metrics.sharpe,
                "max_dd": metrics.max_drawdown,
                "turnover": sleeve_turnover * alpha,
            }
        )
        print(
            f"  alpha={alpha:<4g} tilt={tilt:<4g} net={metrics.total_return:+7.2%} "
            f"etf={rows[-1]['etf']:+.2%} active={active:+7.2%} TE={te:6.2%} IR={ir:5.2f}"
        )

    self_check = [row for row in rows if float(row["alpha"]) == 0.0]
    check_note = (
        "自检行缺失（未含 alpha=0）"
        if not self_check
        else (
            "[自检通过] alpha=0 与 ETF 逐日一致"
            if abs(float(self_check[0]["active"])) < 1e-9 and float(self_check[0]["te"]) < 1e-9
            else f"[自检失败] alpha=0 仍有 {float(self_check[0]['active']):+.2%} 超额"
        )
    )

    lines = [
        "# 指数增强（核心 ETF + 倾斜子组合）",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- **基准 = {args.etf}（可交易、按 100 份/手，一手约 400 元，取整误差可忽略）**",
        f"- 子组合池：{len(bars)} 只（{rde.BENCHMARK_INDEX} 前 {args.pool} 大权重，已剔除一手买不起的标的）",
        "- 组合收益：`r = (1-alpha)·r_ETF + alpha·r_子组合(tilt)`",
        f"- 因子：{args.alpha}({args.lookback})；调仓：每 {args.test} 个交易日；子组合资金：{args.capital:,.0f} 元",
        f"- 价格：{'自算前复权（不复权价 + 分红送转）' if not args.no_self_adjust else '供应商前复权'}",
        f"- {check_note}",
        "",
        "| alpha | tilt | 组合净 | ETF | 沪深300 | 对 ETF 超额 | 跟踪误差 | 信息比率 | Sharpe | 最大回撤 | 换手(×alpha) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:g} | {row['tilt']:g} | {float(row['net']):+.2%} | {float(row['etf']):+.2%} "
            f"| {float(row['index']):+.2%} | **{float(row['active']):+.2%}** | {float(row['te']):.2%} "
            f"| {float(row['ir']):.2f} | {float(row['sharpe']):.2f} | {float(row['max_dd']):.2%} "
            f"| {float(row['turnover']):.0%} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "- `alpha=0` 是**恒等式自检**：与 ETF 逐日相同才算记账正确。这一行不为 0，其它行都不必看。",
        "- 判据：**对 ETF 的信息比率 > 0.5 且跟踪误差 ≤ 5%** 才算可用的指数增强；"
        "超额为正但 IR 低 = 承担了主动风险没换来稳定收益。",
        "- 局限：ETF 腿按曲线合成（未跑撮合，其交易成本 ~2bp 已在真实交易中体现，量级远小于子组合）；"
        f"池只有前 {args.pool} 大权重；tilt 与 alpha 都在同一窗口上挑（技术债 #20）。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(check_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
