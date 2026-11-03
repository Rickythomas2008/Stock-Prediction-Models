"""Generate the program-trading filing draft from the current runtime.

The version number and the declared rate ceilings are read from the installed
package and the runtime settings, so the draft cannot disagree with what the
software actually does.

Usage:
    python tools/filing.py --name 张三 --account A123456789 --broker 某某证券 [--out runs/filing.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stockpred.compliance import (
    AccountInfo,
    CapitalInfo,
    ContactInfo,
    StrategyInfo,
    build_filing,
    runtime_identity,
    write_filing,
)
from stockpred.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成程序化交易报告材料草稿")
    parser.add_argument("--name", required=True, help="投资者姓名")
    parser.add_argument("--account", required=True, help="证券账户号")
    parser.add_argument("--broker", required=True, help="券商名称")
    parser.add_argument("--capital", type=float, default=100_000.0, help="资金规模（元）")
    parser.add_argument("--leverage", type=float, default=0.0, help="杠杆资金（元）")
    parser.add_argument(
        "--strategy",
        default="日频动量选股（低换手，人工确认后自动下单）",
        help="策略类型",
    )
    parser.add_argument(
        "--execution",
        default="程序生成限价委托指令，经券商交易接口下达",
        help="指令执行方式",
    )
    parser.add_argument("--contact-phone", default="", help="联系电话")
    parser.add_argument("--contact-email", default="", help="联系邮箱")
    parser.add_argument("--out", default="runs/program-trading-filing.json", help="输出路径")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    draft = build_filing(
        settings=settings,
        account=AccountInfo(
            investor_name=args.name,
            securities_account=args.account,
            broker_name=args.broker,
        ),
        capital=CapitalInfo(capital_yuan=args.capital, leverage_yuan=args.leverage),
        strategy=StrategyInfo(
            strategy_type=args.strategy,
            execution_method=args.execution,
            max_order_rate_per_sec=settings.max_order_rate_per_sec,
            max_orders_per_day=settings.max_orders_per_day,
        ),
        contact=ContactInfo(
            name=args.name, phone=args.contact_phone, email=args.contact_email
        ),
    )
    path = write_filing(draft, Path(args.out))

    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print("\nruntime:", json.dumps(runtime_identity(), ensure_ascii=False))
    print(
        "\n提醒：本文件是报备草稿。取得券商书面确认（先报告后交易）之前，"
        "系统内不存在任何自动委托路径。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
