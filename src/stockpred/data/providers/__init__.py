"""Provider implementations.

Both real providers import their vendor client lazily, so ``import stockpred``
and the whole test suite work without the optional ``data`` extra installed.
"""

from __future__ import annotations

from stockpred.data.providers.akshare_provider import AkshareIndexProvider, AkshareProvider
from stockpred.data.providers.baostock_provider import BaostockProvider, to_baostock_code
from stockpred.data.providers.base import MarketDataProvider

__all__ = [
    "AkshareIndexProvider",
    "AkshareProvider",
    "BaostockProvider",
    "MarketDataProvider",
    "to_baostock_code",
]
