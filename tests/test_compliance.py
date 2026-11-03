from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stockpred import DEVELOPER, SOFTWARE_NAME, __version__
from stockpred.compliance import (
    AccountInfo,
    CapitalInfo,
    ContactInfo,
    FilingDraft,
    FilingIncomplete,
    SoftwareInfo,
    StrategyInfo,
    build_filing,
    change_report,
    declared_ceilings,
    runtime_identity,
    write_filing,
)
from stockpred.config import Settings


def _account() -> AccountInfo:
    return AccountInfo(
        investor_name="张三",
        securities_account="A123456789",
        broker_name="某证券公司",
    )


def _draft(**overrides) -> FilingDraft:
    payload = {
        "account": _account(),
        "capital": CapitalInfo(capital_yuan=100_000.0),
        "strategy": StrategyInfo(
            strategy_type="日频动量选股（低换手）",
            execution_method="人工确认后由程序经券商交易接口下达限价委托",
            max_order_rate_per_sec=5,
            max_orders_per_day=50,
        ),
        "software": SoftwareInfo(),
        "contact": ContactInfo(name="张三", phone="13800000000"),
    }
    payload.update(overrides)
    return FilingDraft(**payload)


def test_filing_has_all_five_buckets() -> None:
    payload = _draft().as_dict()
    for key in ("1_account", "2_capital", "3_trading", "4_software", "5_contact"):
        assert payload[key], f"{key} must be populated"


def test_software_version_comes_from_the_package() -> None:
    payload = _draft().as_dict()
    assert payload["4_software"]["name"] == SOFTWARE_NAME
    assert payload["4_software"]["version"] == __version__
    assert payload["4_software"]["developer"] == DEVELOPER


def test_rates_come_from_runtime_settings() -> None:
    settings = Settings(max_order_rate_per_sec=3, max_orders_per_day=12)
    draft = build_filing(
        settings=settings,
        account=_account(),
        capital=CapitalInfo(capital_yuan=100_000.0),
        strategy=StrategyInfo("动量", "限价委托", 999, 999),
        contact=ContactInfo(name="张三"),
    )
    assert declared_ceilings(draft) == (3, 12)


def test_leverage_ratio_is_computed() -> None:
    payload = _draft(capital=CapitalInfo(capital_yuan=100_000.0, leverage_yuan=20_000.0)).as_dict()
    assert payload["2_capital"]["leverage_ratio"] == pytest.approx(0.2)


def test_incomplete_filing_is_rejected() -> None:
    draft = _draft(account=AccountInfo(investor_name="", securities_account="", broker_name=""))
    with pytest.raises(FilingIncomplete, match="incomplete"):
        draft.as_dict()


def test_missing_contact_is_rejected() -> None:
    with pytest.raises(FilingIncomplete):
        _draft(contact=ContactInfo(name="")).as_dict()


def test_zero_rate_ceiling_is_rejected() -> None:
    with pytest.raises(FilingIncomplete):
        _draft(
            strategy=StrategyInfo(
                "动量", "限价", max_order_rate_per_sec=0, max_orders_per_day=10
            )
        ).as_dict()


def test_write_filing_roundtrip(tmp_path: Path) -> None:
    draft = _draft()
    path = write_filing(draft, tmp_path / "filing.json")

    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["1_account"]["securities_account"] == "A123456789"
    assert "第7条" in payload["reference"]["articles"]


def test_change_report_flags_version_bump() -> None:
    previous = _draft(software=SoftwareInfo(version="0.0.9"))
    current = _draft(software=SoftwareInfo(version="0.1.0"))
    diff = change_report(previous, current)

    assert diff["is_material_change"] is True
    assert "4_software" in diff["changed"]


def test_change_report_flags_rate_change() -> None:
    previous = _draft()
    current = _draft(
        strategy=StrategyInfo("动量", "限价", max_order_rate_per_sec=5, max_orders_per_day=80)
    )
    diff = change_report(previous, current)
    assert "3_trading" in diff["changed"]


def test_change_report_is_quiet_when_nothing_changed() -> None:
    draft = _draft()
    same = FilingDraft(
        account=draft.account,
        capital=draft.capital,
        strategy=draft.strategy,
        software=draft.software,
        contact=draft.contact,
        generated_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    diff = change_report(draft, same)
    assert diff["is_material_change"] is False


def test_runtime_identity_reports_the_package() -> None:
    identity = runtime_identity()
    assert identity["software"] == SOFTWARE_NAME
    assert identity["version"] == __version__
