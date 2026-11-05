"""xtquant probe: is the vendor SDK installed, and can it actually reach a QMT client?

`xtquant` ships with the QMT client, not with this project. It is also Windows-only
and its licence is unverified, so it must never become a dependency here — this
probe is how we find out what it can do **on this machine, right now**, with no
trading side effects.

It answers three questions in order:

1. installed?  (and from which path — the project ``.venv`` and the user site-packages
   are different interpreters, so "installed" often means "installed elsewhere")
2. connected?  ``xtdata`` talks to a locally running QMT 投研版 / 极简版; without the
   client every call fails with 无法连接xtquant服务
3. usable?     if connected, pull a few daily bars for one stock and one index and
   report what came back

Exit codes are the answer, and they are deliberately distinct:

    0  usable   — connected and returned data
    1  present but not usable — installed, no client answering (the normal state
       on a machine that has the SDK but has never opened QMT)
    2  not installed — **the probe did not run**; never treat it as a pass

This tool never calls ``xttrader``. Order placement requires a logged-in account and
a broker-confirmed program-trading filing (CSRC 2024 No.8 art.8); nothing here is a
step towards it.

Usage:
    python tools/xtquant_probe.py
    python tools/xtquant_probe.py --path "C:/某路径/Lib/site-packages" --timeout 15
    python tools/xtquant_probe.py --stock 000001.SZ --index 000300.SH
"""

from __future__ import annotations

import argparse
import platform
import sys
import threading
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPORT_DEFAULT = "docs/XTQUANT-PROBE.md"


def candidate_parents(explicit: str | None) -> list[Path]:
    """Directories that may contain the ``xtquant`` package."""
    home = Path.home()
    patterns = [
        home / "AppData/Local/Packages",
        home / "AppData/Roaming/Python",
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.base_prefix) / "Lib" / "site-packages",
    ]
    if explicit:
        return [Path(explicit)]

    parents: list[Path] = []
    packages = home / "AppData/Local/Packages"
    if packages.exists():
        for found in packages.glob("*/LocalCache/local-packages/*/site-packages/xtquant"):
            parents.append(found.parent)
    for root in patterns[1:]:
        if root.exists():
            parents.extend(p.parent for p in root.glob("*/site-packages/xtquant"))
            if (root / "xtquant").is_dir():
                parents.append(root)
    # QMT client installs bundle their own interpreter: <client>/bin.x64/Lib/site-packages
    for drive in ("C:/", "D:/"):
        base = Path(drive)
        if base.exists():
            for pattern in ("*/bin.x64/Lib/site-packages/xtquant", "*/Lib/site-packages/xtquant"):
                parents.extend(p.parent for p in base.glob(pattern))
    seen: set[str] = set()
    unique: list[Path] = []
    for parent in parents:
        key = str(parent).lower()
        if key not in seen:
            seen.add(key)
            unique.append(parent)
    return unique


def import_xtquant(explicit: str | None) -> tuple[Any | None, str, list[str]]:
    """Import ``xtquant``, adding discovered site-packages to ``sys.path`` if needed."""
    notes: list[str] = []
    try:
        import xtquant  # noqa: PLC0415 - optional vendor SDK, probed at runtime

        return xtquant, str(Path(xtquant.__file__).parent), notes
    except Exception as exc:  # noqa: BLE001
        notes.append(f"直接 import 失败：{type(exc).__name__}: {exc}")

    for parent in candidate_parents(explicit):
        if not (parent / "xtquant").is_dir():
            continue
        sys.path.insert(0, str(parent))
        try:
            import xtquant  # noqa: PLC0415

            notes.append(f"已从 {parent} 注入路径后导入成功")
            return xtquant, str(Path(xtquant.__file__).parent), notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{parent} 下导入失败：{type(exc).__name__}: {exc}")
            sys.path.pop(0)

    notes.append("未在常见位置找到 xtquant 包目录")
    return None, "", notes


def call_with_timeout(fn: Callable[[], Any], timeout: float) -> tuple[Any, str]:
    """Vendor calls block forever when no client answers; bound them."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None, f"超时（{timeout:.0f}s 无响应）"
    return box.get("value"), str(box.get("error", ""))


def probe_data(xtdata: Any, stock: str, index: str, timeout: float) -> dict[str, Any]:
    end = date.today()
    start = end - timedelta(days=30)
    out: dict[str, Any] = {"calls": []}
    for period in ("1d", "1m"):
        data, error = call_with_timeout(
            lambda p=period: xtdata.get_market_data_ex(
                field_list=["close"],
                stock_list=[stock, index],
                period=p,
                start_time=start.strftime("%Y%m%d"),
                end_time=end.strftime("%Y%m%d"),
            ),
            timeout,
        )
        rows = {}
        if isinstance(data, dict):
            rows = {symbol: len(frame) for symbol, frame in data.items()}
        out["calls"].append({"period": period, "rows": rows, "error": error})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="site-packages directory holding xtquant")
    parser.add_argument("--stock", default="600519.SH")
    parser.add_argument("--index", default="000300.SH")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--report", default=REPORT_DEFAULT)
    args = parser.parse_args(argv)

    xtquant, location, notes = import_xtquant(args.path)
    rows: list[tuple[str, str]] = [
        ("平台", f"{platform.system()} {platform.release()}"),
        ("解释器", f"{sys.executable} ({platform.python_version()})"),
    ]
    detail: dict[str, Any] = {}

    if xtquant is None:
        rows.append(("xtquant", "未安装"))
        status, code = "NOT-INSTALLED（探针未执行）", 2
    else:
        from xtquant import xtdata  # noqa: PLC0415 - imported only when present

        version = str(getattr(xtquant, "__version__", "n/a"))
        rows.append(("xtquant", f"{version} @ {location}"))
        detail = probe_data(xtdata, args.stock, args.index, args.timeout)
        usable = any(call["rows"] for call in detail["calls"])
        errors = [call["error"] for call in detail["calls"] if call["error"]]
        if usable:
            status, code = "USABLE", 0
            rows.append(("客户端", "已连接"))
        elif errors:
            status, code = "PRESENT-BUT-NO-CLIENT", 1
            rows.append(("客户端", f"未连接：{errors[0]}"))
        else:
            status, code = "PRESENT-BUT-EMPTY", 1
            rows.append(("客户端", "已连接但返回空"))

    lines = [
        "# xtquant 探针",
        "",
        f"- 运行时间：{date.today().isoformat()}",
        f"- 结论：**{status}**",
        "",
        "| 项 | 值 |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value} |" for key, value in rows]
    if detail:
        lines += ["", "| 周期 | 标的 | 行数 | 错误 |", "|---|---|---|---|"]
        for call in detail["calls"]:
            fetched = call["rows"] or {}
            symbols = ", ".join(fetched) or "-"
            counts = ", ".join(str(v) for v in fetched.values()) or "-"
            lines.append(f"| {call['period']} | {symbols} | {counts} | {call['error'] or '-'} |")
    if notes:
        lines += ["", "## 探测过程", ""] + [f"- {note}" for note in notes]
    lines += [
        "",
        "## 退出码",
        "",
        "0 = 可用；1 = 已安装但客户端未连接/无数据；2 = 未安装（**未执行**，不得当通过）。",
        "",
        "## 边界",
        "",
        "- 本探针不调用 `xttrader`，不登录、不下单。下单需要已登录账户 + 券商书面确认 "
        "（CSRC 2024 No.8 第 8 条），本工具不是这一步的前置动作。",
        "- `xtquant` 随 QMT 客户端分发、仅 Windows、许可证未核实：**不得加入 "
        "`pyproject.toml` 或 `tools/license_allowlist.toml`**。",
        "",
    ]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    for key, value in rows:
        print(f"  {key}: {value}")
    for call in detail.get("calls", []):
        print(f"  period={call['period']} rows={call['rows']} error={call['error'] or '-'}")
    print(f"wrote {out}")
    print(f"RESULT: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
