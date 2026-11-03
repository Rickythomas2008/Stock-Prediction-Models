"""Live universe run: 20 symbols x N months of real daily bars, with boundary checks.

This is the "test run" the debt list asked for (docs/TECH-DEBT.md #6): it goes
past "can I fetch one symbol" and exercises the edges that break a daily pipeline
— suspended sessions, zero-volume days, adjustment semantics, and provider
disagreement.

Checks per symbol:
  1. schema        -- ``validate_bars`` passes (no silent coercion)
  2. coverage      -- calendar sessions in the window vs rows returned; the gap is
                      suspension/halt evidence, not automatically a failure
  3. zero volume   -- days that traded but reported no volume
  4. adjustment    -- 前复权 <= 不复权 (pointwise max) and 后复权 >= 不复权 (min)
  5. cross-check   -- baostock must agree with akshare on prices/amount/volume

Read-only: nothing is written to ``data/``; the only output is the report file.

Usage:
    python tools/live_universe_run.py                      # 90 days, default universe
    python tools/live_universe_run.py --days 30 --no-crosscheck
    python tools/live_universe_run.py --report docs/LIVE-RUN.md

Requires the ``data`` extra:  pip install -e ".[data]"
Exit codes: 0 = every symbol passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from stockpred.config import Settings
from stockpred.data.crosscheck import compare_bars
from stockpred.data.providers import AkshareProvider, BaostockProvider
from stockpred.data.schema import validate_bars

#: 20 names across all four boards: SH main (60), STAR (688), SZ main (00), ChiNext (30/300).
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


def board_of(symbol: str) -> str:
    if symbol.startswith("688"):
        return "STAR"
    if symbol.startswith("6"):
        return "SH"
    if symbol.startswith("300"):
        return "ChiNext"
    return "SZ"


def adjustment_invariant(
    raw: pd.DataFrame, qfq: pd.DataFrame, hfq: pd.DataFrame
) -> tuple[bool, str]:
    """前复权 must not exceed raw, 后复权 must not fall below raw, on common dates."""
    pair = raw[["date", "close"]].merge(qfq[["date", "close"]], on="date", suffixes=("", "_qfq"))
    pair = pair.merge(hfq[["date", "close"]], on="date", suffixes=("", "_hfq"))
    if pair.empty:
        return False, "no overlapping dates"
    qfq_max = float((pair["close_qfq"] / pair["close"]).max())
    hfq_min = float((pair["close_hfq"] / pair["close"]).min())
    ok = qfq_max <= 1.001 and hfq_min >= 0.999
    return ok, f"qfq/raw max={qfq_max:.4f} hfq/raw min={hfq_min:.4f}"


def run_symbol(
    symbol: str,
    start: date,
    end: date,
    calendar: set[date],
    *,
    crosscheck: bool,
) -> dict[str, object]:
    ak = AkshareProvider()
    row: dict[str, object] = {"symbol": symbol, "board": board_of(symbol), "ok": True, "notes": []}
    try:
        raw = validate_bars(ak.fetch_daily(symbol, start, end), symbol=symbol)
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the run
        row.update(ok=False, rows=0, missing="-", zero_volume="-", adjust="-", xcheck="-")
        row["notes"].append(f"fetch/schema: {type(exc).__name__}: {exc}")
        return row

    seen = {pd.Timestamp(d).date() for d in raw["date"]}
    expected = {d for d in calendar if start <= d <= end}
    missing = sorted(expected - seen)
    # A session that has not closed yet is not "missing" — the vendors only publish
    # after the close, and counting it would make every pre-close run look broken.
    now = datetime.now()
    if missing and missing[-1] == now.date() and now.time() < Settings().market_close:
        missing.pop()
        row["pre_close"] = True
    zero_volume = int((raw["volume"] <= 0).sum())
    row.update(
        rows=len(raw),
        missing=len(missing),
        zero_volume=zero_volume,
        first=str(raw["date"].iloc[0].date()),
        last=str(raw["date"].iloc[-1].date()),
    )
    if missing:
        head = ",".join(str(d) for d in missing[:5])
        row["notes"].append(f"{len(missing)} session(s) absent (halt/suspension): {head}")

    try:
        qfq = validate_bars(ak.fetch_daily(symbol, start, end, adjust="qfq"), symbol=symbol)
        hfq = validate_bars(ak.fetch_daily(symbol, start, end, adjust="hfq"), symbol=symbol)
        adjusted_ok, detail = adjustment_invariant(raw, qfq, hfq)
    except Exception as exc:  # noqa: BLE001
        adjusted_ok, detail = False, f"{type(exc).__name__}: {exc}"
    row["adjust"] = "ok" if adjusted_ok else "FAIL"
    row["ok"] = bool(row["ok"]) and adjusted_ok
    row["notes"].append(f"adjust: {detail}")

    if crosscheck:
        try:
            other = validate_bars(BaostockProvider().fetch_daily(symbol, start, end), symbol=symbol)
            result = compare_bars(raw, other, name_left="akshare", name_right="baostock")
        except Exception as exc:  # noqa: BLE001
            result_ok, detail = False, f"{type(exc).__name__}: {exc}"
        else:
            result_ok = result.ok
            volume_line = next(
                (f for f in result.findings if f.startswith("volume")), "volume: n/a"
            )
            detail = f"{result.sessions} sessions; {volume_line}"
        row["xcheck"] = "ok" if result_ok else "FAIL"
        row["ok"] = bool(row["ok"]) and result_ok
        row["notes"].append(f"cross-check: {detail}")
    else:
        row["xcheck"] = "skip"
    return row


def render_report(rows: list[dict[str, object]], *, start: date, end: date, days: int) -> str:
    passed = sum(1 for r in rows if r["ok"])
    lines = [
        "# 真实行情测试 run（live universe）",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 窗口：{start} ~ {end}（{days} 自然日）",
        f"- 标的：{len(rows)} 个（SH / STAR / SZ / ChiNext）",
        f"- 结果：**{passed}/{len(rows)} 通过**",
        "",
    ]
    if any(r.get("pre_close") for r in rows):
        lines.insert(
            6,
            f"- 说明：运行于收盘前，当日（{date.today().isoformat()}）行情未发布，"
            "已从「缺失交易日」中排除",
        )
    lines += [
        "## 逐标的",
        "",
        "| 标的 | 板块 | 行数 | 缺失交易日 | 零成交量 | 复权不变式 | 双源校验 | 结论 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {symbol} | {board} | {rows} | {missing} | {zero_volume} | {adjust} | {xcheck} | {verdict} |".format(
                verdict="PASS" if row["ok"] else "FAIL", **row
            )
        )
    notes = [
        f"- {row['symbol']}: {note}"
        for row in rows
        for note in row["notes"]  # type: ignore[union-attr]
    ]
    lines += ["", "## 明细", "", *notes, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90, help="calendar days of history")
    parser.add_argument("--no-crosscheck", action="store_true", help="skip baostock comparison")
    parser.add_argument(
        "--report",
        default=None,
        help="markdown report path (default: docs/LIVE-RUN-<date>.md)",
    )
    args = parser.parse_args(argv)

    end = date.today()
    start = end - timedelta(days=max(args.days, 1))
    calendar = set(AkshareProvider().trading_calendar())
    print(f"window {start}..{end} | {len(UNIVERSE)} symbols | calendar {len(calendar)} sessions")

    rows = [
        run_symbol(symbol, start, end, calendar, crosscheck=not args.no_crosscheck)
        for symbol in UNIVERSE
    ]
    for row in rows:
        print(
            f"  {row['symbol']} {row['board']:<7} rows={row['rows']!s:<4} "
            f"missing={row['missing']!s:<3} zerovol={row['zero_volume']!s:<3} "
            f"adj={row['adjust']!s:<5} xcheck={row['xcheck']!s:<5} "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )

    passed = sum(1 for r in rows if r["ok"])
    report = Path(args.report or f"docs/LIVE-RUN-{end.isoformat()}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(rows, start=start, end=end, days=args.days), encoding="utf-8")
    print(f"report {report}")
    print(f"RESULT: {passed}/{len(rows)} passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
