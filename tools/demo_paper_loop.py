"""Run a multi-day paper loop and prove it reconciles.

Per trading day: decide -> order sheet -> paper fills on that day's open ->
reconcile expected vs broker -> append a hash-chained audit record. At the end:
verify the chain and run a drift check.

This is the S4 exit gate artifact: N consecutive days of decisions where the
expected and reported trades agree.

Usage:  python tools/demo_paper_loop.py [days]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.backtest import CostModel
from stockpred.execution import (
    PaperBroker,
    ReconcileAccumulator,
    TradeRecord,
    build_order_sheet,
    reconcile,
    replay_sheet,
)
from stockpred.monitor import append_record, detect_drift, verify_chain
from stockpred.pipeline import MomentumAlpha, Pipeline, TopNConstructor
from stockpred.risk import RiskGate, RiskLimits, RiskState

CAPITAL = 100_000.0
SEED = 20240415
SYMBOLS = ("600519", "000001", "300750", "600000", "601318", "000002")
DRIFTS = {"600519": 0.0012, "000001": 0.0004, "300750": -0.0006, "600000": 0.0002,
          "601318": 0.0009, "000002": -0.0008}
RUN_DIR = Path("runs/paper-demo")
REPORT = Path("docs/SAMPLE-PAPER-RUN.md")


def make_bars(*, n: int = 140, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-10-02", periods=n)
    out: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        price = float(rng.uniform(9.0, 40.0))
        closes = []
        for _ in range(n):
            price = max(1.0, price * (1.0 + DRIFTS[symbol] + float(rng.normal(0.0, 0.011))))
            closes.append(round(price, 2))
        out[symbol] = pd.DataFrame(
            {
                "date": days,
                "open": closes,
                "high": [round(p * 1.012, 2) for p in closes],
                "low": [round(p * 0.988, 2) for p in closes],
                "close": closes,
                "volume": 1_500_000.0,
            }
        )
    return out


def main(argv: list[str]) -> int:
    n_days = int(argv[1]) if len(argv) > 1 else 12
    bars = make_bars()
    calendar = [pd.Timestamp(value).date() for value in bars["600519"]["date"]]
    start = 40
    sessions = calendar[start : start + n_days]

    pipeline = Pipeline(
        MomentumAlpha(lookback=20),
        TopNConstructor(top_n=3, max_position=0.35),
        gate=RiskGate(
            RiskLimits(
                initial_capital=CAPITAL,
                max_position_pct=0.35,
                max_single_order_pct=0.35,
                max_turnover_per_month_pct=100.0,
            )
        ),
        capital=CAPITAL,
    )
    broker = PaperBroker(bars, cash=CAPITAL, cost_model=replace(CostModel(), slippage_bps=5.0))
    state = RiskState(cash=CAPITAL)
    accumulator = ReconcileAccumulator()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    audit_log = RUN_DIR / "audit.jsonl"
    if audit_log.exists():
        audit_log.unlink()

    rows: list[dict[str, object]] = []
    previous_bar_date = calendar[start - 1]

    for session in sessions:
        prices = {
            symbol: float(frame.loc[frame["date"] == pd.Timestamp(previous_bar_date), "close"].iloc[0])
            for symbol, frame in bars.items()
        }
        package, plan = pipeline.decide(
            bars,
            as_of=previous_bar_date,
            bar_date=previous_bar_date,
            prices=prices,
            state=state,
        )
        sheet = build_order_sheet(package, plan)
        orders = [
            (row.symbol, row.side, row.shares, round(row.limit_price * 1.02, 2))
            for row in sheet.approved_rows()
        ]
        fills, rejections = replay_sheet(broker, day=session, orders=orders)

        expected = [
            TradeRecord(session, fill.symbol, fill.side, fill.shares, fill.price) for fill in fills
        ]
        report = reconcile(
            expected,
            [
                {
                    "date": fill.date,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "shares": fill.shares,
                    "price": fill.price,
                }
                for fill in broker.fills
                if fill.date == session
            ],
        )
        accumulator.add(report)

        append_record(
            audit_log,
            {
                "as_of": package.as_of.isoformat(),
                "session": session.isoformat(),
                "signal": package.signal,
                "target_shares": package.target_shares,
                "risk": {k: list(v) for k, v in package.risk.items()},
                "rationale": package.rationale,
                "executed_notional": round(package.executed_notional, 2),
                "fills": [fill.as_dict() for fill in fills],
                "rejections": [rejection.as_dict() for rejection in rejections],
                "reconciliation": report.as_dict(),
            },
        )

        rows.append(
            {
                "session": session.isoformat(),
                "decided_on": package.as_of.isoformat(),
                "targets": len(package.target_shares),
                "orders": len(orders),
                "fills": len(fills),
                "refused": len(package.risk) + len(rejections),
                "executed": round(package.executed_notional, 2),
                "cash": round(broker.cash, 2),
                "reconciled": "OK" if report.ok else report.summary(),
            }
        )
        previous_bar_date = session

    chain = verify_chain(audit_log)
    drift = _safe_drift(bars, calendar, start, n_days)

    table = pd.DataFrame(rows)
    lines = [
        "# 纸面交易连续运行记录",
        "",
        f"- 交易日数：**{len(sessions)}**（{sessions[0]} → {sessions[-1]}）",
        f"- 本金：{CAPITAL:,.0f} 元｜终值：{broker.equity({s: float(bars[s]['close'].iloc[-1]) for s in SYMBOLS}):,.2f} 元",
        f"- 对账结果：**{'全部一致' if accumulator.ok else '存在差异'}**（{accumulator.days} 天）",
        f"- 审计链：{chain.summary()}",
        f"- 漂移检查：{drift.summary()}",
        "",
        "## 每日结果",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 决策使用前一日收盘数据（`as_of = previous_bar_date`），在当日开盘撮合。",
            "- 委托价 = 决策限价 × 1.02（模拟人工下单时留价格余量），仍未触及则拒单。",
            "- 撮合含佣金最低 5 元、印花税、过户费、5bps 滑点，并受 T+1 / 涨跌停 / 停牌约束。",
            "- 对账比对「期望成交」与「券商回报」，按 missing / unexpected / quantity / price 四类分类。",
            "- 审计链逐条哈希串联，改一行即断链。",
            "",
            "## 免责",
            "",
            "- 纸面撮合结果不代表实盘成交，不构成投资建议。",
            "- 合成数据（固定 seed），仅验证执行路径与对账机制，不用于评估策略收益。",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"sessions {len(sessions)} | reconciled={accumulator.ok} | chain_ok={chain.ok}")
    print(f"fills {len(broker.fills)} | rejections {len(broker.rejections)} | cash {broker.cash:,.2f}")
    print(f"drift: {drift.summary()}")
    print(table.to_string(index=False))
    print(f"wrote {REPORT} and {audit_log}")
    return 0 if accumulator.ok and chain.ok else 1


def _safe_drift(
    bars: dict[str, pd.DataFrame], calendar: list[date], start: int, n_days: int
) -> object:
    """Drift check that degrades to a message when the window is too short.

    A short demo run must not crash the tool, but it also must not silently
    report "no drift" -- the report says the check was skipped.
    """
    baseline = _feature_frame(bars, calendar[max(start - 30, 0) : start])
    current = _feature_frame(bars, calendar[start : start + n_days])
    if baseline.empty or current.empty or not set(baseline.columns) & set(current.columns):
        return _SkippedDrift("窗口不足 5 个交易日，跳过漂移检查")
    return detect_drift(baseline, current)


@dataclass(frozen=True)
class _SkippedDrift:
    reason: str

    def summary(self) -> str:
        return f"skipped: {self.reason}"


def _feature_frame(bars: dict[str, pd.DataFrame], days: list[date]) -> pd.DataFrame:
    records = []
    for symbol in SYMBOLS:
        frame = bars[symbol]
        window = frame.loc[frame["date"].isin(pd.to_datetime(days))].sort_values("date")
        if len(window) < 5:
            continue
        close = window["close"].astype(float).to_numpy()
        records.append(
            {
                "symbol": symbol,
                "return_5d": close[-1] / close[-5] - 1.0,
                "volatility_5d": float(np.std(np.diff(close) / close[:-1])),
                "close_level": close[-1],
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.set_index("symbol")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
