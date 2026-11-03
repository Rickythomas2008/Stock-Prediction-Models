"""Live smoke test against the real A-share data providers.

Closes the "adapters have never seen real data" gap (docs/TECH-DEBT.md #6):
fetch daily bars for one symbol, validate them against the canonical schema, and
— when two providers run — require them to agree on prices, amount and volume
(volume is canonicalised to shares, so the expected ratio is 1.0).

Read-only: nothing is written to ``data/`` and no cache entry is created.

Usage:
    python tools/live_smoke.py
    python tools/live_smoke.py --symbol 000001 --days 45 --provider akshare

Requires the ``data`` extra:  pip install -e ".[data]"
Exit codes: 0 = every requested provider passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from stockpred.data.crosscheck import compare_bars
from stockpred.data.providers import AkshareProvider, BaostockProvider
from stockpred.data.schema import validate_bars

PROVIDERS = {"akshare": AkshareProvider, "baostock": BaostockProvider}


def fetch(provider, symbol: str, start: date, end: date) -> pd.DataFrame:
    return validate_bars(provider.fetch_daily(symbol, start, end), symbol=symbol)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="600519", help="A-share code, e.g. 600519")
    parser.add_argument("--days", type=int, default=30, help="calendar days of history")
    parser.add_argument(
        "--provider",
        choices=[*PROVIDERS, "both"],
        default="both",
        help="which provider(s) to exercise",
    )
    args = parser.parse_args(argv)

    end = date.today()
    start = end - timedelta(days=max(args.days, 1))
    names = list(PROVIDERS) if args.provider == "both" else [args.provider]
    frames: dict[str, pd.DataFrame] = {}
    failed = 0

    print(f"symbol={args.symbol} window={start}..{end} providers={','.join(names)}")
    for name in names:
        try:
            frame = fetch(PROVIDERS[name](), args.symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - the point is to surface any failure
            print(f"  {name:<9} FAIL  {type(exc).__name__}: {exc}")
            failed += 1
            continue
        frames[name] = frame
        print(
            f"  {name:<9} OK    rows={len(frame)} "
            f"columns={list(frame.columns)} first={frame['date'].iloc[0].date()} "
            f"last={frame['date'].iloc[-1].date()} last_close={frame['close'].iloc[-1]:.2f} "
            f"last_volume={frame['volume'].iloc[-1]:,.0f} shares"
        )

    if len(frames) == 2:
        left, right = names
        result = compare_bars(frames[left], frames[right], name_left=left, name_right=right)
        print(f"cross-check ({'ok' if result.ok else 'FAIL'}):")
        for finding in result.findings:
            print(f"  {finding}")
        if not result.ok:
            failed += 1

    print("RESULT: " + ("PASS" if failed == 0 else f"FAIL ({failed})"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
