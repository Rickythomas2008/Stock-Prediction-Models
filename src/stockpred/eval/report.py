"""Strategy report rendering with mandatory disclosures.

Two rules are enforced in code rather than by convention:

1. A result is reported **with and without costs**; a single number hides how
   much of the edge the fee model consumes.
2. A result is reported **with its benchmarks**; a bare equity curve is not a
   result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stockpred.eval.metrics import PerformanceMetrics, compute_metrics, metrics_to_dict
from stockpred.eval.walkforward import Fold

DISCLOSURES: tuple[str, ...] = (
    "回测结果不代表未来收益，不构成投资建议。",
    "本报告含交易成本（佣金/印花税/过户费/滑点）前后两套数字；两者差异即成本吞噬。",
    "结果必须与买入持有、现金、随机信号三条基准同表比较。",
    "样本外（walk-forward）评估，训练窗口不进入测试窗口；purge/embargo 已启用时需注明。",
    "A 股规则已建模：T+1、100 股整数倍、涨跌停不可成交、停牌不可交易。",
)


@dataclass(frozen=True)
class CostImpact:
    gross: PerformanceMetrics
    net: PerformanceMetrics

    @property
    def return_consumed(self) -> float:
        """Absolute return lost to costs (gross minus net, in fraction points)."""
        return self.gross.total_return - self.net.total_return

    @property
    def consumed_share(self) -> float:
        """Share of gross return eaten by costs; 0 when there is no gross return."""
        if self.gross.total_return <= 0:
            return 0.0
        return self.return_consumed / self.gross.total_return


def cost_impact(
    gross_equity: pd.Series,
    net_equity: pd.Series,
    *,
    traded_notional: float = 0.0,
) -> CostImpact:
    """Compare the same strategy with and without costs."""
    if len(gross_equity) != len(net_equity):
        raise ValueError("gross and net curves must have the same length")
    return CostImpact(
        gross=compute_metrics(gross_equity.astype(float)),
        net=compute_metrics(net_equity.astype(float), traded_notional=traded_notional),
    )


def _metric_row(name: str, metrics: PerformanceMetrics) -> str:
    data = metrics_to_dict(metrics)
    return (
        f"| {name} | {data['total_return']:.2%} | {data['cagr']:.2%} | "
        f"{data['sharpe']:.2f} | {data['max_drawdown']:.2%} | {data['n_days']} |"
    )


PERCENT_COLUMNS = frozenset(
    {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate", "turnover"}
)
INT_COLUMNS = frozenset({"n_days"})


def _format_cell(column: str, value: object) -> str:
    """Format by column semantics, not by magnitude.

    Magnitude heuristics broke here once: a Sharpe of 2.41 was rendered as
    "241.27%" because it happened to be below 10.
    """
    if column in INT_COLUMNS:
        return f"{int(value)}"
    if column in PERCENT_COLUMNS:
        return f"{float(value):.2%}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_report(
    *,
    strategy_name: str,
    impact: CostImpact,
    benchmark_table: pd.DataFrame,
    folds: Sequence[Fold] | None = None,
    traded_notional: float = 0.0,
    generated_at: datetime | None = None,
    extra_notes: Mapping[str, str] | None = None,
) -> str:
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# 策略评估报告：{strategy_name}",
        "",
        f"- 生成时间：{stamp}",
        f"- 评估协议：walk-forward（样本外），折数 {len(folds) if folds else 'n/a'}",
        f"- 累计成交金额：{traded_notional:,.2f} 元",
        "",
        "## 1. 成本影响（必看）",
        "",
        "| 版本 | 总收益 | 年化 | Sharpe | 最大回撤 | 天数 |",
        "|---|---|---|---|---|---|",
        _metric_row("含成本（净）", impact.net),
        _metric_row("不含成本（毛）", impact.gross),
        "",
        f"- 成本吞噬收益：**{impact.return_consumed:.2%}**"
        f"（占毛收益 **{impact.consumed_share:.1%}**）",
        "",
        "## 2. 与基准对比",
        "",
    ]

    header = "| 序列 | " + " | ".join(benchmark_table.columns) + " |"
    separator = "|---" * (len(benchmark_table.columns) + 1) + "|"
    lines.extend([header, separator])
    for label, row in benchmark_table.iterrows():
        cells = [_format_cell(str(column), value) for column, value in row.items()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if folds:
        lines.extend(["", "## 3. 折区间", "", "| fold | train | test |", "|---|---|---|"])
        for fold in folds:
            lines.append(
                f"| {fold.index} | {fold.train_start.date()}..{fold.train_end.date()} "
                f"| {fold.test_start.date()}..{fold.test_end.date()} |"
            )

    if extra_notes:
        lines.extend(["", "## 4. 备注", ""])
        lines.extend(f"- {key}：{value}" for key, value in extra_notes.items())

    lines.extend(["", "## 免责与披露", ""])
    lines.extend(f"- {item}" for item in DISCLOSURES)
    lines.append("")
    return "\n".join(lines)


def failing_gates(impact: CostImpact, *, min_net_sharpe: float = 0.0) -> list[str]:
    """Reasons a strategy may not be promoted to a live decision pipeline."""
    reasons: list[str] = []
    if impact.net.total_return <= 0:
        reasons.append("net total return is not positive after costs")
    if impact.net.sharpe <= min_net_sharpe:
        reasons.append(f"net Sharpe {impact.net.sharpe:.2f} <= {min_net_sharpe:.2f}")
    if impact.net.max_drawdown < -0.30:
        reasons.append(f"max drawdown {impact.net.max_drawdown:.1%} exceeds -30%")
    if impact.consumed_share > 0.5 and impact.gross.total_return > 0:
        reasons.append(
            f"costs consume {impact.consumed_share:.0%} of gross return (turnover too high)"
        )
    return reasons


def benchmark_warnings(
    strategy: PerformanceMetrics,
    benchmarks: Mapping[str, PerformanceMetrics],
) -> list[str]:
    """Advisories (not blockers) from the benchmark comparison.

    A strategy can beat the cash benchmark and still lose to buy-and-hold. That
    is a real outcome, not a bug -- but shipping it silently would misrepresent
    the result, so it must be surfaced.
    """
    warnings: list[str] = []
    hold = benchmarks.get("buy_and_hold")
    cash = benchmarks.get("cash")
    random = benchmarks.get("random_signal")

    if hold is not None and strategy.total_return < hold.total_return:
        warnings.append(
            f"跑输买入持有：策略 {strategy.total_return:.2%} vs 持有 {hold.total_return:.2%}"
            "（若风险调整后更优需显式说明）"
        )
    if cash is not None and strategy.total_return <= cash.total_return:
        warnings.append("未跑赢现金基准：交易没有创造价值")
    if random is not None and strategy.sharpe <= random.sharpe:
        warnings.append(
            f"未跑赢随机信号的风险调整收益：策略 Sharpe {strategy.sharpe:.2f} "
            f"vs 随机 {random.sharpe:.2f}"
        )
    if hold is not None and strategy.sharpe > hold.sharpe and strategy.total_return < hold.total_return:
        warnings.append("风险调整后优于持有但绝对收益更低：需判断是否符合资金目标")
    return warnings


def summarize_dates(folds: Sequence[Fold]) -> tuple[date, date]:
    return folds[0].test_start.date(), folds[-1].test_end.date()
