"""Alpha and config walk-forward: pick on one block, measure on the later blocks.

Two questions, one harness:

1. **Is a configuration's edge real, or did I pick it on the window I reported?**
   The timeline is cut into non-overlapping blocks; the config is chosen *on* one
   block by a rule fixed in advance and then measured on the later blocks untouched.
2. **Which alpha survives that treatment?** 20-day momentum turned out to be
   downside protection rather than an edge (docs/CONFIG-WALKFORWARD.md), so the
   same protocol now runs over momentum, short-horizon reversal and low volatility
   — all price-only, so the comparison shares one data path.

Selection rule (fixed before looking at results): among configs whose annualised
turnover is at most ``--max-turnover``, take the highest Sharpe on the block.

Usage:
    python tools/config_walkforward.py                        # baostock, 3 blocks
    python tools/config_walkforward.py --provider akshare --days 1800

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_data_eval as rde  # noqa: E402

from stockpred.eval import compute_metrics  # noqa: E402


@dataclass(frozen=True)
class Score:
    config: rde.Config
    net: float
    sharpe: float
    turnover_year: float
    periods: int


def build_grid() -> list[rde.Config]:
    return [
        rde.Config(alpha=alpha, top=top, buffer=buffer, lookback=lookback)
        for alpha in rde.ALPHAS
        for top in (8, 12)
        for buffer in (2, 4)
        for lookback in (20, 60)
    ]


def score_window(
    bars, sessions, config: rde.Config, window: tuple[date, date], *, capital: float
) -> Score:
    run = rde.run_window(
        bars, sessions, config, capital=capital, protocol="continuous", window=window
    )
    metrics = compute_metrics(run.net, traded_notional=run.traded)
    years = max(metrics.n_days / 242.0, 1e-9)
    return Score(
        config=config,
        net=metrics.total_return,
        sharpe=metrics.sharpe,
        turnover_year=metrics.turnover / years,
        periods=len(run.folds),
    )


def pick(scores: list[Score], max_turnover: float) -> Score:
    eligible = [s for s in scores if s.turnover_year <= max_turnover] or scores
    return max(eligible, key=lambda s: (s.sharpe, -s.turnover_year))


def apply_block(scores: list[Score], config: rde.Config) -> Score:
    return next(s for s in scores if s.config == config)


def index_return(closes: pd.Series, window: tuple[date, date]) -> float:
    frame = closes[(closes.index.date >= window[0]) & (closes.index.date <= window[1])]
    if len(frame) < 2:
        return 0.0
    return float(frame.iloc[-1] / frame.iloc[0] - 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--end", default=None)
    parser.add_argument("--provider", choices=["baostock", "akshare"], default="baostock")
    parser.add_argument("--benchmark", default=rde.BENCHMARK_INDEX)
    parser.add_argument("--capital", type=float, default=rde.DEFAULT_CAPITAL)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--max-turnover", type=float, default=4.0, help="annualised, e.g. 4.0 = 400%%"
    )
    parser.add_argument("--report", default="docs/CONFIG-WALKFORWARD.md")
    args = parser.parse_args(argv)

    window_end = date.fromisoformat(args.end) if args.end else None
    print(f"loading {len(rde.UNIVERSE)} symbols from {args.provider}")
    bars = rde.load_universe(args.days, "qfq", args.provider, window_end)
    sessions = sorted({d.date() for frame in bars.values() for d in frame["date"]})
    index_closes = rde.load_benchmark(args.benchmark, bars, args.provider, "qfq")

    reference = rde.run_window(bars, sessions, rde.Config(), capital=args.capital)
    folds = reference.folds
    size = len(folds) // args.blocks
    if size < 2:
        raise SystemExit(f"only {len(folds)} periods; need at least {args.blocks * 2}")
    windows = [
        (folds[i * size].test_start.date(), folds[(i + 1) * size - 1].test_end.date())
        for i in range(args.blocks)
    ]
    print(
        f"{len(folds)} periods in {args.blocks} blocks: "
        + " | ".join(f"{a}..{b}" for a, b in windows)
    )

    grid = build_grid()
    scores: dict[int, list[Score]] = {}
    for i, window in enumerate(windows):
        scores[i] = []
        for config in grid:
            score = score_window(bars, sessions, config, window, capital=args.capital)
            scores[i].append(score)
        top = sorted(scores[i], key=lambda s: s.sharpe, reverse=True)[:3]
        print(
            f"  block{i} best: " + " ; ".join(f"{s.config.describe()}={s.sharpe:.2f}" for s in top)
        )

    rows: list[dict[str, object]] = []
    for i in range(args.blocks - 1):
        for alpha in rde.ALPHAS:
            chosen = pick([s for s in scores[i] if s.config.alpha == alpha], args.max_turnover)
            for j in range(i + 1, args.blocks):
                applied = apply_block(scores[j], chosen.config)
                bench = index_return(index_closes, windows[j])
                rows.append(
                    {
                        "alpha": alpha,
                        "select_block": i,
                        "apply_block": j,
                        "config": chosen.config.describe(),
                        "select_sharpe": chosen.sharpe,
                        "select_turnover": chosen.turnover_year,
                        "oos_net": applied.net,
                        "oos_sharpe": applied.sharpe,
                        "oos_turnover": applied.turnover_year,
                        "index": bench,
                        "excess": applied.net - bench,
                    }
                )

    lines = [
        "# Alpha / 配置 walk-forward：在一个窗口选，在另一个窗口量",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 数据源：{args.provider}（{len(bars)} 标的，{sessions[0]}..{sessions[-1]}）",
        f"- 基准：{args.benchmark} 指数买入持有",
        f"- 选择规则（事先固定）：年化换手 ≤ {args.max_turnover:.0%} 的配置里取 Sharpe 最高",
        f"- 网格：{len(grid)} 组（α ∈ {{momentum, reversal, low_vol}} × top 8/12 × stay 2/4 × lookback 20/60）",
        "",
        "## 分块",
        "",
        "| 块 | 区间 | 指数 |",
        "|---|---|---|",
    ]
    lines += [
        f"| block{i} | {windows[i][0]}..{windows[i][1]} | {index_return(index_closes, windows[i]):+.2%} |"
        for i in range(args.blocks)
    ]

    lines += [
        "",
        "## 各 α 的样本外表现（选择窗口 → 应用窗口）",
        "",
        "| α | 选择 | 应用 | 选中配置 | 选择期 Sharpe | 净收益 | 指数 | 超额 | 年化换手 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']} | block{row['select_block']} | block{row['apply_block']} "
            f"| {row['config']} | {float(row['select_sharpe']):.2f} "
            f"| {float(row['oos_net']):+.2%} | {float(row['index']):+.2%} "
            f"| **{float(row['excess']):+.2%}** | {float(row['oos_turnover']):.0%} |"
        )

    lines += [
        "",
        "## α 汇总（只看没参与选择的窗口）",
        "",
        "| α | 样本外组数 | 超额均值 | 超额中位 | 正超额 | 年化换手中位 |",
        "|---|---|---|---|---|---|",
    ]
    for alpha in rde.ALPHAS:
        subset = [row for row in rows if row["alpha"] == alpha]
        if not subset:
            continue
        excesses = [float(row["excess"]) for row in subset]
        turns = [float(row["oos_turnover"]) for row in subset]
        lines.append(
            f"| {alpha} | {len(subset)} | {statistics.fmean(excesses):+.2%} "
            f"| {statistics.median(excesses):+.2%} "
            f"| {sum(1 for e in excesses if e > 0)}/{len(excesses)} "
            f"| {statistics.median(turns):.0%} |"
        )

    best = max(
        rde.ALPHAS,
        key=lambda a: statistics.fmean(
            [float(row["excess"]) for row in rows if row["alpha"] == a] or [-9.0]
        ),
    )
    lines += [
        "",
        "## 结论与读法",
        "",
        f"- 样本外超额均值最高的 α：**{best}**（这只是当前窗口的排名，不是结论）。",
        "- 判据：若某 α 的正超额次数接近全胜、且超额中位显著大于 0，才谈得上「比拿指数更多」。"
        "正负参半的 α 应被视为风格暴露（涨市跑输 / 跌市少亏），不可作为超额来源。",
        "- 本表每行只用「选择窗口之前的信息 + 选择窗口内数据」定配置，应用窗口完全未参与选择。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    for alpha in rde.ALPHAS:
        subset = [float(r["excess"]) for r in rows if r["alpha"] == alpha]
        if subset:
            print(f"  {alpha}: mean excess {statistics.fmean(subset):+.2%} ({len(subset)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
