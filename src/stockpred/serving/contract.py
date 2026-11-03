"""Transport-free contract for the decision-serving API.

Every rule of the API lives here as a pure function over plain dictionaries: no
socket, no file, no global state. ``stockpred.serving.app`` is the only module
that knows HTTP exists, so a transport bug cannot change an answer and the whole
contract is testable without starting a server.

Failure codes are a three-way split, not the usual "worked / errored" pair:

* ``400`` -- the request itself is wrong (missing field, unparseable date, no
  bar history). The caller can fix the request; retrying it unchanged is
  pointless.
* ``422`` -- the request is well formed but a decision cannot be produced
  (not enough history for the lookback, no usable price). Same payload, same
  answer.
* ``500`` -- reserved for genuine server faults. A business failure reported as
  500 hides the real cause from the caller, so it is never used for one.

Nothing here places an order: ``decide`` returns a reviewable decision package
plus a human-executable order sheet. Any future automatic submission must first
pass ``Settings.assert_auto_order_allowed`` (broker confirmation of the
program-trading filing, CSRC 2024 No.8 art.8).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

import stockpred
from stockpred.errors import StockpredError
from stockpred.execution import SHEET_COLUMNS, build_order_sheet
from stockpred.pipeline import MomentumAlpha, Pipeline, TopNConstructor
from stockpred.risk import RiskGate, RiskLimits, RiskState

#: Shape of the legacy ``/trade`` answer. Frozen because old clients read exactly
#: these four keys; adding to it is safe, removing or retyping is a break.
LEGACY_TRADE_FIELDS = ("action", "balance", "status", "timestamp")

#: Version of this response contract, independent of the package version.
#: ``/health`` reports the package version; this reports the wire shape.
API_VERSION = "v1"

#: Columns every bar row must carry. Validated up front so a malformed row is a
#: 400 (the caller's fault) instead of a KeyError deep inside the alpha stage.
BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume")

#: Fields the ``/decide`` body must contain. All of them are required rather
#: than defaulted: a decision sized by a server-side default capital is not
#: reviewable after the fact, which defeats the point of the decision log.
DECIDE_REQUIRED_FIELDS = (
    "as_of",
    "bar_date",
    "capital",
    "bars",
    "prices",
    "lookback",
    "top_n",
    "max_position_pct",
    "cash",
)

LEGACY_WINDOW = 20
LEGACY_LOT = 100
LEGACY_DEFAULT_CAPITAL = 100_000.0

_INT_SHEET_COLUMNS = frozenset({"shares"})
_FLOAT_SHEET_COLUMNS = frozenset({"limit_price", "notional"})


@dataclass(frozen=True)
class ApiResponse:
    """A status code plus a JSON-ready body. Deliberately not an HTTP object."""

    status: int
    body: dict[str, Any]


def error_response(
    status: int,
    error: str,
    *,
    field: str | None = None,
    detail: str | None = None,
) -> ApiResponse:
    """Build the one error envelope: ``{"error": ..., "field": ..., "detail": ...}``.

    Public because the transport layer raises 404/405/413 on its own and a client
    must not have to parse two different error shapes.
    """
    body: dict[str, Any] = {"error": error}
    if field is not None:
        body["field"] = field
    if detail is not None:
        body["detail"] = detail
    return ApiResponse(status=status, body=body)


class _BadRequest(Exception):
    """Carries the 400 reply a validator wants back, unwinding to the entry point.

    Validation is spread over per-field helpers; threading a return value
    through each of them would bury the actual checks in plumbing.
    """

    def __init__(self, response: ApiResponse) -> None:
        super().__init__(response.body.get("error", "invalid_request"))
        self.response = response


def openapi_schema() -> dict[str, Any]:
    """A hand-written OpenAPI 3.1 description.

    Hand-written on purpose: the package has no web framework, and adding one to
    generate this would cost more than the three endpoints are worth. It is a
    static literal, so it is stable enough to snapshot in tests and can be
    returned verbatim to any client that asks for it.
    """
    version = stockpred.__version__
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "stockpred decision API",
            "version": version,
            "summary": "Decision packages and human-executable order sheets for A-share trading.",
            "description": (
                "Read-only decision support. The API never submits an order; the "
                "order sheet it returns is executed by a person at a broker app."
            ),
        },
        "paths": {
            "/health": {
                "get": {
                    "operationId": "health",
                    "summary": "Liveness probe with the package version.",
                    "responses": {
                        "200": {
                            "description": "Service is up.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["status", "version"],
                                        "properties": {
                                            "status": {"type": "string", "const": "ok"},
                                            "version": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/decide": {
                "post": {
                    "operationId": "decide",
                    "summary": "Run alpha -> constructor -> risk gate and return the decision.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DecisionRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "A decision package plus its order sheet rows.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DecisionResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "422": {"$ref": "#/components/responses/DecisionFailed"},
                    },
                }
            },
            "/trade": {
                "get": {
                    "operationId": "legacyTrade",
                    "summary": "Compatibility layer for the legacy Flask /trade endpoint.",
                    "parameters": [
                        {
                            "name": "data",
                            "in": "query",
                            "required": True,
                            "description": "JSON array [close, volume] of the latest bar.",
                            "schema": {"type": "string", "example": "[13.1, 13407500]"},
                        },
                        {
                            "name": "history",
                            "in": "query",
                            "required": False,
                            "description": "JSON array of previous closes; 20+ produce a signal.",
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "capital",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "number", "default": LEGACY_DEFAULT_CAPITAL},
                        },
                        {
                            "name": "position",
                            "in": "query",
                            "required": False,
                            "description": "Shares currently held.",
                            "schema": {"type": "integer", "default": 0},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Legacy trade signal.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LegacyTradeResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "DecisionRequest": {
                    "type": "object",
                    "required": list(DECIDE_REQUIRED_FIELDS),
                    "properties": {
                        "as_of": {"type": "string", "format": "date"},
                        "bar_date": {"type": "string", "format": "date"},
                        "capital": {"type": "number", "exclusiveMinimum": 0},
                        "cash": {"type": "number", "minimum": 0},
                        "lookback": {"type": "integer", "minimum": 1},
                        "top_n": {"type": "integer", "minimum": 1},
                        "max_position_pct": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 1,
                        },
                        "prices": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "bars": {
                            "type": "object",
                            "description": "Per-symbol daily bars, oldest to newest.",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": list(BAR_COLUMNS),
                                    "properties": {
                                        "date": {"type": "string", "format": "date"},
                                        "open": {"type": "number"},
                                        "high": {"type": "number"},
                                        "low": {"type": "number"},
                                        "close": {"type": "number"},
                                        "volume": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
                "DecisionResponse": {
                    "type": "object",
                    "required": [
                        "as_of",
                        "bar_date",
                        "signal",
                        "target_shares",
                        "risk",
                        "rationale",
                        "orders",
                        "api_version",
                    ],
                    "properties": {
                        "as_of": {"type": "string", "format": "date"},
                        "generated_at": {"type": "string", "format": "date-time"},
                        "bar_date": {"type": "string", "format": "date"},
                        "signal": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "target_shares": {
                            "type": "object",
                            "additionalProperties": {"type": "integer"},
                        },
                        "risk": {
                            "type": "object",
                            "description": "Refused symbols and the limits they violated.",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "rationale": {"type": "string"},
                        "cash_after": {"type": "number"},
                        "target_invested": {"type": "number"},
                        "executed_notional": {"type": "number"},
                        "leftover_cash": {"type": "number"},
                        "dropped": {"type": "array", "items": {"type": "string"}},
                        "orders": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/OrderRow"},
                        },
                        "api_version": {"type": "string", "const": API_VERSION},
                    },
                },
                "OrderRow": {
                    "type": "object",
                    "required": list(SHEET_COLUMNS),
                    "properties": {
                        "as_of": {"type": "string", "format": "date"},
                        "symbol": {"type": "string"},
                        "side": {"type": "string", "enum": ["buy", "sell"]},
                        "shares": {"type": "integer"},
                        "limit_price": {"type": "number"},
                        "notional": {"type": "number"},
                        "valid_until": {"type": "string", "format": "date"},
                        "risk_status": {"type": "string", "enum": ["approved", "refused"]},
                        "note": {"type": "string"},
                    },
                },
                "LegacyTradeResponse": {
                    "type": "object",
                    "required": list(LEGACY_TRADE_FIELDS),
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell", "nothing"]},
                        "balance": {"type": "number"},
                        "status": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "signal": {"type": "string"},
                    },
                },
                "Error": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {"type": "string"},
                        "field": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                },
            },
            "responses": {
                "BadRequest": {
                    "description": "The request is malformed and can be fixed by the caller.",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}
                    },
                },
                "DecisionFailed": {
                    "description": "The request is valid but no decision can be produced from it.",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}
                    },
                },
            },
        },
    }


def health() -> ApiResponse:
    """Liveness probe. Reports the package version so a client can prove what it reached."""
    return ApiResponse(status=200, body={"status": "ok", "version": stockpred.__version__})


def decide(payload: dict[str, Any]) -> ApiResponse:
    """Run the decision pipeline over caller-supplied bars and prices.

    Bars and prices arrive in the body because the serving layer must not own
    data acquisition: fetching, freshness checks and caching stay in
    ``stockpred.data``, so a stale-bar bug cannot be mistaken for an API bug.
    """
    try:
        return _decide(payload)
    except _BadRequest as exc:
        return exc.response


def _decide(payload: Any) -> ApiResponse:
    if not isinstance(payload, Mapping):
        raise _BadRequest(
            error_response(400, "invalid_request", detail="body must be a JSON object")
        )

    as_of = _date_field(payload, "as_of")
    bar_date = _date_field(payload, "bar_date")
    capital = _number_field(payload, "capital", minimum=0.0, exclusive=True)
    cash = _number_field(payload, "cash", minimum=0.0)
    lookback = _integer_field(payload, "lookback")
    top_n = _integer_field(payload, "top_n")
    max_position_pct = _number_field(payload, "max_position_pct", minimum=0.0, exclusive=True)
    if max_position_pct > 1.0:
        raise _BadRequest(error_response(400, "invalid_parameter", field="max_position_pct"))

    prices = _prices(_require(payload, "prices"))
    bars = _bars(_require(payload, "bars"))

    pipeline = Pipeline(
        MomentumAlpha(lookback=lookback),
        TopNConstructor(top_n=top_n, max_position=max_position_pct),
        gate=RiskGate(
            RiskLimits(
                initial_capital=capital,
                max_position_pct=max_position_pct,
                max_single_order_pct=max_position_pct,
            )
        ),
        capital=capital,
    )

    try:
        package, plan = pipeline.decide(
            bars,
            as_of=as_of,
            bar_date=bar_date,
            prices=prices,
            state=RiskState(cash=cash),
        )
        sheet = build_order_sheet(package, plan)
        body = package.as_dict()
    except StockpredError as exc:
        # Stages refuse with typed errors (short history, no price, unusable
        # output). The request itself was readable, so this is 422, not 400.
        return error_response(422, "decision_failed", detail=str(exc))
    except ValueError as exc:
        # RiskLimits/Proposal reject unusable sizing this way. Limits were
        # validated above, so reaching here means the payload implied an
        # impossible position -- still the caller's payload, still not a 500.
        return error_response(422, "decision_failed", detail=str(exc))

    # Orders are parsed from the sheet's CSV form because that string is what a
    # person reads at the broker app; parsing it guarantees the API returns
    # exactly the rows the human executes.
    body["orders"] = _sheet_rows(sheet.to_csv())
    body["api_version"] = API_VERSION
    return ApiResponse(status=200, body=body)


def legacy_trade(payload: dict[str, Any]) -> ApiResponse:
    """Compatibility layer for the legacy Flask ``GET /trade`` endpoint.

    What is kept from the old service (``realtime-agent/app.py``): the four-key
    wire shape, the ``bool``-free string ``action``, and the exact status string
    ``"data not enough to trade"`` that old clients keyed on. What is explicitly
    *not* kept: the old handler was stateful (module-level ``_queue``,
    ``_inventory``, ``_capital``) and its signal came from a softmax over a
    deep-evolution policy, so its answer depended on every earlier request. This
    shim is a pure function of the payload and computes the documented 20-day
    mean rule instead. Old and new answers therefore are **not** interchangeable
    in value -- only in shape -- which is exactly why this layer must never be
    described as a reviewed decision (兼容层，语义与旧版仅保证报文形状一致，不参与新管线).

    Input: ``{"data": [close, volume], "history": [close...], "capital": float,
    "position": int}``. ``history`` is optional; ``capital`` defaults to
    100,000 and ``position`` to 0.

    Rules, evaluated in this order:

    1. fewer than 20 closes in ``history`` -> ``nothing`` with status
       ``"data not enough to trade"``; "no signal" must never look like a hold.
    2. ``close > mean(history[-20:])`` and ``position == 0`` -> ``buy``, unless
       one lot costs more than ``capital``, in which case ``nothing`` with the
       reason in ``status``.
    3. ``close < mean(history[-20:])`` and ``position > 0`` -> ``sell``.
    4. otherwise -> ``nothing``.

    ``balance`` is cash after the trade, assuming exactly one 100-share lot
    moves: buy ``capital - close*100``, sell ``capital + close*100``, nothing
    ``capital``. ``volume`` is accepted for wire compatibility and never used --
    the legacy rule had no volume term either.
    """
    try:
        return _legacy_trade(payload)
    except _BadRequest as exc:
        return exc.response


def _legacy_trade(payload: Any) -> ApiResponse:
    if not isinstance(payload, Mapping):
        raise _BadRequest(
            error_response(400, "invalid_request", detail="body must be a JSON object")
        )

    close, history, capital, position = _legacy_inputs(payload)

    if len(history) < LEGACY_WINDOW:
        action = "nothing"
        balance = capital
        status = "data not enough to trade"
        signal = f"history={len(history)} < {LEGACY_WINDOW}, no signal"
    else:
        mean = sum(history[-LEGACY_WINDOW:]) / LEGACY_WINDOW
        signal = f"close={close:.4f} ma{LEGACY_WINDOW}={mean:.4f} position={position}"
        cost = close * LEGACY_LOT
        if close > mean and position == 0:
            if cost > capital:
                action = "nothing"
                balance = capital
                status = (
                    f"capital {capital:.2f} not enough for 1 lot "
                    f"({LEGACY_LOT} shares at {close:.2f} = {cost:.2f})"
                )
            else:
                action = "buy"
                balance = capital - cost
                status = f"buy 1 lot ({LEGACY_LOT} shares) at latest close"
        elif close < mean and position > 0:
            action = "sell"
            balance = capital + cost
            status = f"sell 1 lot ({LEGACY_LOT} shares) at latest close"
        else:
            action = "nothing"
            balance = capital
            status = "no signal: close and moving average agree with the current position"

    body: dict[str, Any] = {
        "action": action,
        "balance": float(round(balance, 2)),
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "signal": signal,
    }
    return ApiResponse(status=200, body=body)


def _legacy_inputs(payload: Mapping[str, Any]) -> tuple[float, list[float], float, int]:
    """Validate the legacy body, returning already-coerced values."""
    raw_data = _require(payload, "data")
    if not isinstance(raw_data, (list, tuple)) or len(raw_data) != 2:
        raise _BadRequest(
            error_response(400, "invalid_data", field="data", detail="expected [close, volume]")
        )
    close = _number(raw_data[0])
    volume = _number(raw_data[1])
    if close is None or not math.isfinite(close) or close <= 0:
        raise _BadRequest(
            error_response(
                400, "invalid_data", field="data", detail="close must be a positive number"
            )
        )
    if volume is None or not math.isfinite(volume) or volume < 0:
        raise _BadRequest(
            error_response(
                400, "invalid_data", field="data", detail="volume must be a non-negative number"
            )
        )

    history: list[float] = []
    raw_history = payload.get("history")
    if raw_history is not None:
        if not isinstance(raw_history, (list, tuple)):
            raise _BadRequest(error_response(400, "invalid_parameter", field="history"))
        for value in raw_history:
            number = _number(value)
            if number is None or not math.isfinite(number):
                raise _BadRequest(error_response(400, "invalid_parameter", field="history"))
            history.append(number)

    raw_capital = payload.get("capital")
    capital = LEGACY_DEFAULT_CAPITAL if raw_capital is None else _number(raw_capital)
    if capital is None or not math.isfinite(capital) or capital <= 0:
        raise _BadRequest(error_response(400, "invalid_parameter", field="capital"))

    raw_position = payload.get("position")
    position = 0.0 if raw_position is None else _number(raw_position)
    if position is None or position < 0 or not position.is_integer():
        raise _BadRequest(error_response(400, "invalid_parameter", field="position"))

    return close, history, capital, int(position)


def _require(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload or payload[name] is None:
        raise _BadRequest(error_response(400, "missing_field", field=name))
    return payload[name]


def _number(raw: Any) -> float | None:
    """Numeric input only. Strings are rejected: a JSON number is a number.

    Accepting ``"20"`` would paper over a client that serialises everything to
    strings, and the sizing computed from it would look correct.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _date_field(payload: Mapping[str, Any], name: str) -> date:
    raw = _require(payload, name)
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str):
        raise _BadRequest(error_response(400, "invalid_date", field=name))
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise _BadRequest(error_response(400, "invalid_date", field=name)) from None


def _number_field(
    payload: Mapping[str, Any],
    name: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float:
    value = _number(_require(payload, name))
    if value is None or not math.isfinite(value):
        raise _BadRequest(error_response(400, "invalid_parameter", field=name))
    if value < minimum or (exclusive and value == minimum):
        raise _BadRequest(error_response(400, "invalid_parameter", field=name))
    return value


def _integer_field(payload: Mapping[str, Any], name: str) -> int:
    raw = _require(payload, name)
    if not isinstance(raw, bool):
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, float) and raw.is_integer() and raw > 0:
            return int(raw)
    raise _BadRequest(error_response(400, "invalid_parameter", field=name))


def _prices(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping) or not raw:
        raise _BadRequest(
            error_response(
                400, "invalid_parameter", field="prices", detail="need at least one price"
            )
        )
    prices: dict[str, float] = {}
    for symbol, value in raw.items():
        number = _number(value)
        # A zero or negative price sizes a position to zero or to infinity;
        # refuse the request instead of guessing which was meant.
        if not isinstance(symbol, str) or number is None or not math.isfinite(number) or number <= 0:
            raise _BadRequest(
                error_response(
                    400, "invalid_parameter", field="prices", detail=str(symbol)
                )
            )
        prices[symbol] = number
    return prices


def _bars(raw: Any) -> dict[str, pd.DataFrame]:
    if not isinstance(raw, Mapping) or not raw:
        raise _BadRequest(error_response(400, "invalid_bars"))
    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in raw.items():
        if not isinstance(symbol, str) or not isinstance(rows, (list, tuple)) or not rows:
            raise _BadRequest(error_response(400, "invalid_bars", detail=str(symbol)))
        try:
            frame = pd.DataFrame(list(rows))
        except (TypeError, ValueError) as exc:
            raise _BadRequest(
                error_response(400, "invalid_bars", detail=f"{symbol}: {exc}")
            ) from None
        missing = [name for name in BAR_COLUMNS if name not in frame.columns]
        if missing:
            raise _BadRequest(
                error_response(400, "invalid_bars", detail=f"{symbol}: missing {missing}")
            )
        try:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            for column in BAR_COLUMNS[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise _BadRequest(
                error_response(400, "invalid_bars", detail=f"{symbol}: {exc}")
            ) from None
        # Sorted once here so the alpha stage never depends on wire order.
        frames[symbol] = frame.sort_values("date").reset_index(drop=True)
    return frames


def _sheet_rows(text: str) -> list[dict[str, Any]]:
    return [
        {name: _sheet_value(name, raw[name]) for name in SHEET_COLUMNS}
        for raw in csv.DictReader(io.StringIO(text))
    ]


def _sheet_value(name: str, value: str) -> Any:
    """CSV carries everything as text; hand numbers back as numbers."""
    if name in _INT_SHEET_COLUMNS:
        return int(value)
    if name in _FLOAT_SHEET_COLUMNS:
        return float(value)
    return value


__all__ = [
    "API_VERSION",
    "BAR_COLUMNS",
    "DECIDE_REQUIRED_FIELDS",
    "LEGACY_TRADE_FIELDS",
    "ApiResponse",
    "decide",
    "error_response",
    "health",
    "legacy_trade",
    "openapi_schema",
]
