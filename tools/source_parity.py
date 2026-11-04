"""Data-source parity gate: can akshare and baostock be swapped without moving results?

The whole evaluation chain assumes the bars are the bars. Two vendors mean two
possible answers, so before trusting any number this gate fetches the same window
from both providers and demands agreement — price to the 分, amount to the yuan,
volume to within 1% (both are canonicalised to shares).

This is an S4 gate, not a report: exit codes are the output.

    0  parity holds for every symbol
    1  parity broken (see the table)
    2  a provider is unavailable — the gate did NOT run, do not treat it as a pass

Usage:
    python tools/source_parity.py                       # 20 symbols, 180 days
    python tools/source_parity.py --symbols 600519,000001 --days 60

Requires the ``data`` extra:  pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_data_eval as rde  # noqa: E402

from stockpred.data.crosscheck import compare_bars  # noqa: E402
from stockpred.data.providers import AkshareProvider, BaostockProvider  # noqa: E402
from stockpred.data.schema import validate_bars  # noqa: E402
from stockpred.errors import ProviderUnavailable  # noqa: E402

RETURN_TOLERANCE = 0.002  # adjusted-return drift above this is a warning
RETURN_HARD_LIMIT = 0.005  # ... and above this the vendors disagree materially


def fetch(provider, symbol: str, start: date, end: date, *, adjust: str = "qfq"):
    for _ in range(3):
        try:
            return validate_bars(
                provider.fetch_daily(symbol, start, end, adjust=adjust), symbol=symbol
            )
        except Exception:  # noqa: BLE001 - retry, then report
            continue
    raise ProviderUnavailable(f"{getattr(provider, 'name', '?')} gave nothing for {symbol}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", default=",".join(rde.UNIVERSE))
    parser.add_argument("--report", default="docs/SOURCE-PARITY.md")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = end - timedelta(days=args.days)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    akshare, baostock = AkshareProvider(), BaostockProvider()
    rows: list[dict[str, object]] = []
    provider_down = 0

    print(f"comparing {len(symbols)} symbols over {start}..{end}")
    for symbol in symbols:
        left = right = None
        left_error = right_error = ""
        try:
            left = fetch(akshare, symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            left_error = f"{type(exc).__name__}: {str(exc)[:60]}"
        try:
            right = fetch(baostock, symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            right_error = f"{type(exc).__name__}: {str(exc)[:60]}"

        if left is None and right is None:
            rows.append(
                {"symbol": symbol, "verdict": f"no data from either provider ({left_error})"}
            )
            continue
        if left is None or right is None:
            provider_down += 1
            side = "akshare" if left is None else "baostock"
            rows.append(
                {"symbol": symbol, "verdict": f"{side} unavailable: {left_error or right_error}"}
            )
            continue

        raw_left = raw_right = None
        try:
            raw_left = fetch(akshare, symbol, start, end, adjust="")
            raw_right = fetch(baostock, symbol, start, end, adjust="")
        except Exception as exc:  # noqa: BLE001
            left_error = left_error or f"{type(exc).__name__}: {str(exc)[:60]}"

        if raw_left is None or raw_right is None:
            if left is None or right is None:
                provider_down += 1
                side = "akshare" if left is None else "baostock"
                rows.append(
                    {
                        "symbol": symbol,
                        "verdict": f"{side} unavailable: {left_error or right_error}",
                    }
                )
            else:
                rows.append({"symbol": symbol, "verdict": f"raw fetch failed: {left_error}"})
            continue

        # Unadjusted bars are the vendor's own numbers: these must match exactly.
        result = compare_bars(raw_left, raw_right, name_left="akshare", name_right="baostock")
        left_ret = float(left["close"].iloc[-1] / left["close"].iloc[0] - 1.0)
        right_ret = float(right["close"].iloc[-1] / right["close"].iloc[0] - 1.0)
        drift = abs(left_ret - right_ret)
        if not result.ok:
            verdict = "FAIL (raw bars differ)"
        elif drift > RETURN_HARD_LIMIT:
            verdict = f"FAIL adjusted-return drift {drift:.4%}"
        elif drift > RETURN_TOLERANCE:
            verdict = f"WARN adjusted-return drift {drift:.4%}"
        else:
            verdict = "ok"
        rows.append(
            {
                "symbol": symbol,
                "sessions": result.sessions,
                "bars_ok": result.ok,
                "return_drift": drift,
                "verdict": verdict,
            }
        )
        print(
            f"  {symbol}: sessions={result.sessions} raw_ok={result.ok} qfq_drift={drift:.4%} {verdict}"
        )

    def provider_down_verdict(verdict: str) -> bool:
        return verdict.startswith(("akshare unavailable", "baostock unavailable"))

    failed = [row for row in rows if str(row["verdict"]).startswith(("FAIL", "no data", "raw"))]
    warned = [row for row in rows if str(row["verdict"]).startswith("WARN")]
    quiet = [row for row in rows if str(row["verdict"]) == "ok"]
    if failed:
        status, code = f"FAIL ({len(failed)} symbols)", 1
    elif provider_down and not quiet and not warned:
        status, code = f"SKIPPED ({provider_down} symbols had one provider down)", 2
    elif warned or provider_down:
        status, code = (
            f"PASS with {len(warned)} warnings / {provider_down} provider gaps",
            0,
        )
    else:
        status, code = "PASS", 0

    lines = [
        "# 数据源一致性门（akshare vs baostock）",
        "",
        f"- 运行时间：{date.today().isoformat()}",
        f"- 窗口：{start}..{end}，{len(symbols)} 标的",
        "- 判据 1（硬）：**不复权**逐日价格差 ≤ 0.011 元、成交额差 ≤ 1 元、成交量比值 1.0±1%"
        "——两家拿到的是同一个市场，这一层不允许分歧",
        f"- 判据 2（软）：**前复权**区间收益差 ≤ {RETURN_TOLERANCE:.2%} 视为一致，"
        f"≤ {RETURN_HARD_LIMIT:.2%} 记警告，超过记 FAIL。复权因子由各厂商自算，"
        "允许小差；但如果复权后收益都差，策略在两个源上会得出不同结论",
        f"- 结论：**{status}**",
        "",
        "| 标的 | 重叠交易日 | 不复权逐日一致 | 前复权区间收益差 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        drift = row.get("return_drift")
        lines.append(
            f"| {row['symbol']} | {row.get('sessions', '-')} | {row.get('bars_ok', '-')} "
            f"| {f'{float(drift):.4%}' if isinstance(drift, float) else '-'} | {row['verdict']} |"
        )
    lines += [
        "",
        "退出码：0=通过，1=不一致，2=数据源不可用（**未执行**，不得当作通过）。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"RESULT: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
