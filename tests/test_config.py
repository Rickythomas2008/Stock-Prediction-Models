from __future__ import annotations

from pathlib import Path

import pytest

from stockpred.config import Settings
from stockpred.errors import BrokerNotConfirmed


def test_auto_order_blocked_by_default() -> None:
    with pytest.raises(BrokerNotConfirmed, match="no broker confirmation"):
        Settings().assert_auto_order_allowed()


def test_auto_order_allowed_after_broker_confirmation() -> None:
    settings = Settings(broker_confirmed_at="2026-11-10T09:00:00+08:00")
    settings.assert_auto_order_allowed()


def test_env_overrides_are_parsed() -> None:
    settings = Settings.from_env(
        {
            "STOCKPRED_DATA_ROOT": "D:/market-data",
            "STOCKPRED_MAX_STALE_TRADING_DAYS": "2",
            "STOCKPRED_MAX_ORDER_RATE_PER_SEC": "3",
            "STOCKPRED_MAX_ORDERS_PER_DAY": "20",
            "STOCKPRED_BROKER_CONFIRMED_AT": "2026-11-10T09:00:00+08:00",
        }
    )

    assert settings.data_root == Path("D:/market-data")
    assert settings.cache_root == Path("D:/market-data/cache")
    assert settings.max_stale_trading_days == 2
    assert settings.max_order_rate_per_sec == 3
    assert settings.max_orders_per_day == 20
    settings.assert_auto_order_allowed()


def test_defaults_are_conservative() -> None:
    settings = Settings.from_env({})
    assert settings.max_stale_trading_days == 0
    assert settings.broker_confirmed_at is None
    assert settings.max_orders_per_day <= 50
