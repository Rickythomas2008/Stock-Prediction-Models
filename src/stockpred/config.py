"""Runtime settings with hard safety defaults.

Two of these fields are not tunables but guards:
``max_order_rate_per_sec`` / ``max_orders_per_day`` are the values declared in the
program-trading filing (CSRC 2024 No.8, art.7) and are enforced downstream;
``broker_confirmed_at`` must stay empty until the broker confirms the filing
(art.8), which makes ``assert_auto_order_allowed`` the single gate before any
automatic order.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import time
from pathlib import Path

from stockpred.errors import BrokerNotConfirmed

ENV_PREFIX = "STOCKPRED_"


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("data")
    max_stale_trading_days: int = 0
    market_close: time = time(15, 0)
    max_order_rate_per_sec: int = 5
    max_orders_per_day: int = 50
    broker_confirmed_at: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        load = cls()
        data_root = source.get(f"{ENV_PREFIX}DATA_ROOT")
        stale = source.get(f"{ENV_PREFIX}MAX_STALE_TRADING_DAYS")
        rate = source.get(f"{ENV_PREFIX}MAX_ORDER_RATE_PER_SEC")
        per_day = source.get(f"{ENV_PREFIX}MAX_ORDERS_PER_DAY")
        confirmed = source.get(f"{ENV_PREFIX}BROKER_CONFIRMED_AT")
        return replace(
            load,
            data_root=Path(data_root) if data_root else load.data_root,
            max_stale_trading_days=int(stale) if stale else load.max_stale_trading_days,
            max_order_rate_per_sec=int(rate) if rate else load.max_order_rate_per_sec,
            max_orders_per_day=int(per_day) if per_day else load.max_orders_per_day,
            broker_confirmed_at=confirmed or None,
        )

    def assert_auto_order_allowed(self) -> None:
        """Raise unless the broker has confirmed the program-trading filing."""
        if not self.broker_confirmed_at:
            raise BrokerNotConfirmed(
                "auto-order blocked: no broker confirmation on record "
                "(report first, trade afterwards - CSRC 2024 No.8 art.8)"
            )

    @property
    def cache_root(self) -> Path:
        return self.data_root / "cache"
