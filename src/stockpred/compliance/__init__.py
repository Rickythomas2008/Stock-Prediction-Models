"""Program-trading filing material (程序化交易报告材料).

《证券市场程序化交易管理规定（试行）》(CSRC 2024 No.8, effective 2024-10-08) art.7
requires five buckets of information before a client may trade programmatically:

1. 账户基本信息       -> account holder, securities account, broker, manager
2. 账户资金信息       -> capital size, source, leverage size and ratio
3. 交易信息           -> strategy type, order execution method, max order rate, max orders/day
4. 交易软件信息       -> software name, version, developer
5. 其他               -> contact person and details

Art.8 makes it "report first, trade afterwards": the broker confirms before the
client trades. So this module generates the filing *draft*, and the software
version is taken from the installed package -- never hand-typed, because a stale
version number in a filing is a false statement.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stockpred import DEVELOPER, SOFTWARE_NAME, __version__
from stockpred.config import Settings
from stockpred.errors import StockpredError

FILING_FILENAME = "program-trading-filing.json"


class FilingIncomplete(StockpredError):
    """A required bucket has no content; an incomplete filing cannot be submitted."""


@dataclass(frozen=True)
class AccountInfo:
    investor_name: str
    securities_account: str
    broker_name: str
    manager: str = ""


@dataclass(frozen=True)
class CapitalInfo:
    capital_yuan: float
    source: str = "自有资金"
    leverage_yuan: float = 0.0

    @property
    def leverage_ratio(self) -> float:
        if self.capital_yuan <= 0:
            return 0.0
        return self.leverage_yuan / self.capital_yuan


@dataclass(frozen=True)
class StrategyInfo:
    strategy_type: str
    execution_method: str
    max_order_rate_per_sec: int
    max_orders_per_day: int


@dataclass(frozen=True)
class SoftwareInfo:
    name: str = SOFTWARE_NAME
    version: str = __version__
    developer: str = DEVELOPER
    runtime: str = field(default_factory=lambda: f"Python {platform.python_version()}")


@dataclass(frozen=True)
class ContactInfo:
    name: str
    phone: str = ""
    email: str = ""


@dataclass(frozen=True)
class FilingDraft:
    account: AccountInfo
    capital: CapitalInfo
    strategy: StrategyInfo
    software: SoftwareInfo
    contact: ContactInfo
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def validate(self) -> None:
        missing: list[str] = []
        if not self.account.investor_name:
            missing.append("account.investor_name")
        if not self.account.securities_account:
            missing.append("account.securities_account")
        if not self.account.broker_name:
            missing.append("account.broker_name")
        if self.capital.capital_yuan <= 0:
            missing.append("capital.capital_yuan")
        if not self.strategy.strategy_type:
            missing.append("strategy.strategy_type")
        if not self.strategy.execution_method:
            missing.append("strategy.execution_method")
        if self.strategy.max_order_rate_per_sec <= 0:
            missing.append("strategy.max_order_rate_per_sec")
        if self.strategy.max_orders_per_day <= 0:
            missing.append("strategy.max_orders_per_day")
        if not self.software.name or not self.software.version or not self.software.developer:
            missing.append("software")
        if not self.contact.name:
            missing.append("contact.name")
        if missing:
            raise FilingIncomplete(f"filing is incomplete: {missing}")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "generated_at": self.generated_at.isoformat(),
            "1_account": asdict(self.account),
            "2_capital": {
                **asdict(self.capital),
                "leverage_ratio": round(self.capital.leverage_ratio, 6),
            },
            "3_trading": asdict(self.strategy),
            "4_software": asdict(self.software),
            "5_contact": asdict(self.contact),
            "reference": {
                "regulation": "《证券市场程序化交易管理规定（试行）》证监会公告〔2024〕8号",
                "effective": "2024-10-08",
                "articles": "第7条（报告内容）、第8条（先报告后交易）",
                "high_frequency_thresholds": "单账户每秒申报+撤单≥300笔，或单日≥20,000笔",
            },
        }


def build_filing(
    *,
    settings: Settings,
    account: AccountInfo,
    capital: CapitalInfo,
    strategy: StrategyInfo,
    contact: ContactInfo,
) -> FilingDraft:
    """Assemble a filing draft, taking rate limits from the runtime settings.

    The declared rate/order ceilings come from ``Settings`` on purpose: the same
    numbers the risk gate enforces must be the ones written into the filing.
    """
    draft = FilingDraft(
        account=account,
        capital=capital,
        strategy=StrategyInfo(
            strategy_type=strategy.strategy_type,
            execution_method=strategy.execution_method,
            max_order_rate_per_sec=settings.max_order_rate_per_sec,
            max_orders_per_day=settings.max_orders_per_day,
        ),
        software=SoftwareInfo(),
        contact=contact,
    )
    draft.validate()
    return draft


def write_filing(draft: FilingDraft, path: Path | str = Path("runs") / FILING_FILENAME) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(draft.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def change_report(previous: FilingDraft, current: FilingDraft) -> dict[str, Any]:
    """Art.7 requires reporting material changes; this lists what changed."""
    before = previous.as_dict()
    after = current.as_dict()
    changed = {
        key: {"from": before.get(key), "to": after.get(key)}
        for key in after
        if key not in {"generated_at", "reference"} and before.get(key) != after.get(key)
    }
    return {
        "is_material_change": bool(changed),
        "changed": changed,
        "note": "软件版本号变更、申报速率/笔数上限变更须及时变更报告",
    }


def declared_ceilings(draft: FilingDraft) -> tuple[int, int]:
    """(max_order_rate_per_sec, max_orders_per_day) as filed -- for cross-checking config."""
    return draft.strategy.max_order_rate_per_sec, draft.strategy.max_orders_per_day


def runtime_identity() -> dict[str, str]:
    return {
        "software": SOFTWARE_NAME,
        "version": __version__,
        "developer": DEVELOPER,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
