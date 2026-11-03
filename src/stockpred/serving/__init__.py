"""Decision serving: a pure contract plus a thin HTTP binding.

``contract`` is the single source of truth for request validation, status codes
and response bodies; ``app`` only routes HTTP onto it. Nothing is imported from
``app`` here, so importing this package never binds a socket.
"""

from __future__ import annotations

from stockpred.serving.contract import (
    API_VERSION,
    LEGACY_TRADE_FIELDS,
    ApiResponse,
    decide,
    error_response,
    health,
    legacy_trade,
    openapi_schema,
)

__all__ = [
    "API_VERSION",
    "LEGACY_TRADE_FIELDS",
    "ApiResponse",
    "decide",
    "error_response",
    "health",
    "legacy_trade",
    "openapi_schema",
]
