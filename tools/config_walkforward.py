"""Config walk-forward: pick the parameters on one window, measure on the next.

TECH-DEBT #20 in one script. The strategy's configuration (book width, stay band,
lookback) was previously chosen on the very window it was reported on, so any
"beat the index" claim carried selection bias. Here the timeline is cut into
three non-overlapping blocks; the config is chosen **on** one block by a stated
rule and then **measured** on the later blocks without touching it.

Selection rule (fixed before looking at results): among configs whose annualised
turnover is at most ``--max-turnover``, take the highest Sharpe. Turnover is capped
because cost is controllable and return is not.

Reports the out-of-sample excess over the index for each selection→application
pair, plus the in-hindsight best config for contrast (how much selection costs).

Usage:
    python tools/config_walkforward.py                       # baostock, 3 blocks
    python tools/config_walkforward.py --provider akshare --days 1800

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_data_eval as rde  # noqa: E402

from stockpred.eval import compute_metrics  # noqa: E402

Grid = list[rde.Config]


@dataclass(frozen=True)
class WindowScore:
    config: rde.Config
    net: float
    sharpe: float
    turnover_year: float
    periods: int


def grid() -> Grid:
    return [
        rde.Config(top=top, buffer=buffer, lookback=lookback)
        for top in (8, 12)
        for buffer in (2, 4)
        for lookback in (20, 60)
    ]


def score_window(
    bars,
    sessions,
    config: rde.Config,
    window: tuple[date, date],
    *,
    capital: float,
) -> WindowScore:
    run = rde.run_window(
        bars, sessions, config, capital=capital, protocol="continuous", window=window
    )
    metrics = compute_metrics(run.net, traded_notional=run.traded)
    years = max(metrics.n_days / 242.0, 1e-9)
    return WindowScore(
        config=config,
        net=metrics.total_return,
        sharpe=metrics.sharpe,
        turnover_year=metrics.turnover / years,
        periods=len(run.folds),
    )


def pick(scores: list[WindowScore], max_turnover: float) -> WindowScore:
    eligible = [s for s in scores if s.turnover_year <= max_turnover] or scores
    return max(eligible, key=lambda s: (s.sharpe, -s.turnover_year))


def best_hindsight(scores: list[WindowScore], max_turnover: float) -> WindowScore:
    return pick(scores, max_turnover)


def index_window(closes: pd.Series, window: tuple[date, date]) -> float:
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
    parser.add_argument(
        "--max-turnover", type=float, default=4.0, help="annualised, e.g. 4.0=400%%"
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
    blocks = 3
    size = len(folds) // blocks
    if size < 2:
        raise SystemExit(f"only {len(folds)} periods; need at least {blocks * 2}")
    windows = [
        (folds[i * size].test_start.date(), folds[(i + 1) * size - 1].test_end.date())
        for i in range(blocks)
    ]
    print(
        f"{len(folds)} periods in {blocks} blocks: " + " | ".join(f"{a}..{b}" for a, b in windows)
    )

    configs = grid()
    scores: dict[int, list[WindowScore]] = {}
    for i, window in enumerate(windows):
        scores[i] = []
        for config in configs:
            score = score_window(bars, sessions, config, window, capital=args.capital)
            scores[i].append(score)
            print(
                f"  block{i} {config.describe():<48} net={score.net:+7.2%} "
                f"sharpe={score.sharpe:5.2f} turnover={score.turnover_year:5.0%}/yr"
            )

    rows: list[dict[str, object]] = []
    for i in range(blocks - 1):
        chosen = pick(scores[i], args.max_turnover)
        hindsight = best_hindsight(scores[i], args.max_turnover)
        for j in range(i + 1, blocks):
            applied = next(s for s in scores[j] if s.config == chosen.config)
            index_ret = index_window(index_closes, windows[j])
            rows.append(
                {
                    "select_block": i,
                    "apply_block": j,
                    "config": chosen.config.describe(),
                    "select_sharpe": chosen.sharpe,
                    "select_turnover": chosen.turnover_year,
                    "oos_net": applied.net,
                    "oos_sharpe": applied.sharpe,
                    "oos_turnover": applied.turnover_year,
                    "index": index_ret,
                    "excess": applied.net - index_ret,
                    "hindsight_sharpe": hindsight.sharpe,
                }
            )

    wins = sum(1 for row in rows if float(row["excess"]) > 0)  # type: ignore[arg-type]
    lines = [
        "# 配置 walk-forward：在一个窗口选，在另一个窗口量",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 数据源：{args.provider}（{len(bars)} 标的，{sessions[0]}..{sessions[-1]}）",
        f"- 基准：{args.benchmark} 指数买入持有",
        f"- 选择规则（事先固定）：年化换手 ≤ {args.max_turnover:.0%} 的配置里取 Sharpe 最高",
        f"- 网格：{len(configs)} 组（top 8/12 × stay 2/4 × lookback 20/60，免交易带 2%，月调仓）",
        "",
        "## 各块表现（同一配置在不同块）",
        "",
        "| 块 | 区间 |",
        "|---|---|",
    ]
    lines += [f"| block{i} | {windows[i][0]}..{windows[i][1]} |" for i in range(blocks)]
    lines += [
        "",
        "## 选择 → 应用（后者为配置的样本外）",
        "",
        "| 在 block 选 | 配置 | 选择期 Sharpe | 应用到 block | 净收益 | 指数 | 超额 | 年化换手 | 同期事后最优 Sharpe |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| block{row['select_block']} | {row['config']} | {float(row['select_sharpe']):.2f} "
            f"| block{row['apply_block']} | {float(row['oos_net']):+.2%} "
            f"| {float(row['index']):+.2%} | **{float(row['excess']):+.2%}** "
            f"| {float(row['oos_turnover']):.0%} | {float(row['hindsight_sharpe']):.2f} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"- 样本外跑赢指数：**{wins}/{len(rows)}** 组",
        "- 读法：本表只回答「按事先定的规则选出来的配置，换到没参与选择的窗口还行不行」。"
        "若多数组合的超额为正，选择的偏差小于信号本身；若正负参半，则此前的「跑赢」主要来自"
        "在同一窗口上挑配置。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"OOS beats index: {wins}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
