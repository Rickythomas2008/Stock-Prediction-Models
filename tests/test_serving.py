"""Contract and transport tests for ``stockpred.serving``.

The contract tests call the pure functions directly -- they are the fast,
network-free loop. The two roundtrip tests bind a real ``ThreadingHTTPServer`` on
an ephemeral port to prove the routing layer speaks the same contract, because a
handler typo cannot fail a direct call.
"""

from __future__ import annotations

import contextlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import date
from typing import Any

import pandas as pd
import pytest

import stockpred
from stockpred.execution import SHEET_COLUMNS
from stockpred.pipeline import REQUIRED_DECISION_FIELDS
from stockpred.serving.app import create_server
from stockpred.serving.contract import (
    API_VERSION,
    LEGACY_TRADE_FIELDS,
    ApiResponse,
    decide,
    health,
    legacy_trade,
    openapi_schema,
)

AS_OF = "2024-03-20"
SYMBOLS = ("600519", "000001", "300750", "600000", "601318", "000858")

_MISSING = object()


def _bars(
    count: int = 40, *, drift: float = 0.005, start: float = 20.0
) -> dict[str, list[dict[str, Any]]]:
    """Six symbols x ``count`` business days, gently rising and slightly staggered."""
    days = pd.date_range(end=AS_OF, periods=count, freq="B")
    out: dict[str, list[dict[str, Any]]] = {}
    for index, symbol in enumerate(SYMBOLS):
        price = start + index
        rows: list[dict[str, Any]] = []
        for day in days:
            price *= 1.0 + drift + 0.0002 * index
            close = round(price, 2)
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "open": close,
                    "high": round(close * 1.01, 2),
                    "low": round(close * 0.99, 2),
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )
        out[symbol] = rows
    return out


def _payload(**overrides: Any) -> dict[str, Any]:
    bars = overrides.pop("bars", _MISSING)
    if bars is _MISSING:
        bars = _bars()
    payload: dict[str, Any] = {
        "as_of": AS_OF,
        "bar_date": AS_OF,
        "capital": 100_000.0,
        "bars": bars,
        "prices": {symbol: rows[-1]["close"] for symbol, rows in bars.items()},
        "lookback": 20,
        "top_n": 3,
        "max_position_pct": 0.35,
        "cash": 100_000.0,
    }
    payload.update(overrides)
    return payload


def _history(count: int, *, start: float, step: float) -> list[float]:
    return [round(start + step * i, 4) for i in range(count)]


# --- contract: health -------------------------------------------------------


def test_health_ok() -> None:
    response = health()
    assert isinstance(response, ApiResponse)
    assert response.status == 200
    assert response.body["status"] == "ok"
    assert response.body["version"] == stockpred.__version__


# --- contract: decide -------------------------------------------------------


def test_decide_success() -> None:
    response = decide(_payload())
    assert response.status == 200, response.body
    body = response.body
    for name in REQUIRED_DECISION_FIELDS:
        assert name in body, f"{name} must be present"
    assert body["as_of"] == AS_OF
    assert body["api_version"] == API_VERSION
    assert isinstance(body["orders"], list) and body["orders"], "expected an order sheet"
    for row in body["orders"]:
        assert set(row) == set(SHEET_COLUMNS)
        assert isinstance(row["shares"], int)
        assert isinstance(row["limit_price"], float)
        assert row["risk_status"] in {"approved", "refused"}
    assert all(shares % 100 == 0 for shares in body["target_shares"].values())


def test_decide_missing_field() -> None:
    payload = _payload()
    del payload["bars"]
    response = decide(payload)
    assert response.status == 400
    assert response.body == {"error": "missing_field", "field": "bars"}


def test_decide_invalid_date() -> None:
    response = decide(_payload(as_of="2024-13-45"))
    assert response.status == 400
    assert response.body == {"error": "invalid_date", "field": "as_of"}

    bar_date = decide(_payload(bar_date="not-a-date"))
    assert bar_date.status == 400
    assert bar_date.body["field"] == "bar_date"


def test_decide_invalid_parameter() -> None:
    for overrides in (
        {"lookback": 0},
        {"top_n": -1},
        {"max_position_pct": 0.0},
        {"max_position_pct": 1.5},
        {"capital": 0},
        {"prices": {"600519": "20.1"}},
    ):
        response = decide(_payload(**overrides))
        assert response.status == 400, overrides
        assert response.body["error"] == "invalid_parameter", overrides
        assert response.body["field"] in overrides, overrides


def test_decide_empty_bars() -> None:
    response = decide(_payload(bars={}, prices={"600519": 20.1}))
    assert response.status == 400
    assert response.body["error"] == "invalid_bars"


def test_decide_business_failure_is_422() -> None:
    # Well-formed request, impossible answer: 500 bars of history was never sent.
    response = decide(_payload(lookback=500))
    assert response.status == 422
    assert response.body["error"] == "decision_failed"
    assert "no symbol has" in response.body["detail"]


# --- contract: legacy /trade ------------------------------------------------


def test_legacy_trade_contract_shape() -> None:
    response = legacy_trade({"data": [13.1, 13_407_500], "history": _history(20, start=12.0, step=0.05)})
    assert response.status == 200
    body = response.body
    assert set(body) >= set(LEGACY_TRADE_FIELDS)
    assert isinstance(body["action"], str)
    assert isinstance(body["balance"], float)
    assert isinstance(body["status"], str)
    assert isinstance(body["timestamp"], str)
    assert body["action"] in {"buy", "sell", "nothing"}


def test_legacy_trade_insufficient_history() -> None:
    response = legacy_trade({"data": [13.1, 13_407_500]})
    assert response.status == 200
    assert response.body["action"] == "nothing"
    # Verbatim: old clients keyed on this exact string.
    assert response.body["status"] == "data not enough to trade"
    assert response.body["balance"] == 100_000.0

    short = legacy_trade({"data": [13.1, 13_407_500], "history": _history(19, start=10.0, step=0.1)})
    assert short.body["action"] == "nothing"
    assert short.body["status"] == "data not enough to trade"


def test_legacy_trade_buy_and_sell() -> None:
    rising = _history(20, start=10.0, step=0.1)  # mean 10.95, close 13.1 above it
    bought = legacy_trade({"data": [13.1, 1_000_000], "history": rising})
    assert bought.body["action"] == "buy"
    assert bought.body["balance"] == pytest.approx(100_000.0 - 13.1 * 100)

    falling = _history(20, start=20.0, step=-0.1)  # mean 19.05, close 13.1 below it
    sold = legacy_trade({"data": [13.1, 1_000_000], "history": falling, "position": 100})
    assert sold.body["action"] == "sell"
    assert sold.body["balance"] == pytest.approx(100_000.0 + 13.1 * 100)

    # Same falling series but flat: no reason to sell what is not held.
    held_flat = legacy_trade({"data": [13.1, 1_000_000], "history": falling})
    assert held_flat.body["action"] == "nothing"

    # Rising series while already positioned: no reason to add a second lot.
    add_on = legacy_trade({"data": [13.1, 1_000_000], "history": rising, "position": 100})
    assert add_on.body["action"] == "nothing"

    # One lot costs more than the account: refuse, and say why.
    broke = legacy_trade({"data": [500.0, 1_000_000], "history": rising, "capital": 40_000.0})
    assert broke.body["action"] == "nothing"
    assert "not enough" in broke.body["status"]


def test_legacy_trade_rejects_malformed_data() -> None:
    assert legacy_trade({}).body["error"] == "missing_field"
    assert legacy_trade({"data": [13.1]}).body["error"] == "invalid_data"
    assert legacy_trade({"data": ["x", 1]}).body["error"] == "invalid_data"


# --- contract: openapi ------------------------------------------------------


def test_openapi_schema_snapshot() -> None:
    schema = openapi_schema()
    assert schema["openapi"] == "3.1.0"
    assert set(schema["paths"]) == {"/health", "/decide", "/trade"}
    assert "get" in schema["paths"]["/health"]
    assert "post" in schema["paths"]["/decide"]
    assert "get" in schema["paths"]["/trade"]
    assert schema["info"]["version"] == stockpred.__version__

    request = schema["components"]["schemas"]["DecisionRequest"]
    assert set(request["required"]) == set(_payload())
    assert {"as_of", "bar_date", "signal", "target_shares", "risk", "rationale"} <= set(
        schema["components"]["schemas"]["DecisionResponse"]["properties"]
    )
    assert set(schema["components"]["schemas"]["LegacyTradeResponse"]["required"]) == set(
        LEGACY_TRADE_FIELDS
    )


# --- transport --------------------------------------------------------------


@contextlib.contextmanager
def _running_server() -> Iterator[str]:
    """Bind on an ephemeral port so parallel runs cannot collide."""
    server = create_server(port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def base_url() -> Iterator[str]:
    with _running_server() as url:
        yield url


def _request(
    url: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        finally:
            exc.close()


def test_http_roundtrip_health(base_url: str) -> None:
    status, body = _request(f"{base_url}/health")
    assert status == 200
    assert body == {"status": "ok", "version": stockpred.__version__}


def test_http_roundtrip_decide(base_url: str) -> None:
    status, body = _request(f"{base_url}/decide", method="POST", payload=_payload())
    assert status == 200, body
    assert body["api_version"] == API_VERSION
    assert body["orders"]

    # Same payload through the contract directly: transport must not add or drop fields.
    assert body["target_shares"] == decide(_payload()).body["target_shares"]


def test_http_roundtrip_trade(base_url: str) -> None:
    query = urllib.parse.urlencode(
        {"data": "[13.1, 13407500]", "history": "[12.0, 12.1, 12.2]"}
    )
    status, body = _request(f"{base_url}/trade?{query}")
    assert status == 200
    assert set(body) >= set(LEGACY_TRADE_FIELDS)
    assert body["action"] == "nothing"

    broken, broken_body = _request(f"{base_url}/trade?data=not-json")
    assert broken == 400
    assert broken_body["error"] == "invalid_data"


def test_unknown_path_404(base_url: str) -> None:
    status, body = _request(f"{base_url}/nope")
    assert status == 404
    assert body["error"] == "not_found"

    post_status, _ = _request(f"{base_url}/nope", method="POST", payload={})
    assert post_status == 404


def test_wrong_method_405(base_url: str) -> None:
    status, body = _request(f"{base_url}/decide")
    assert status == 405
    assert body["error"] == "method_not_allowed"

    post_status, post_body = _request(f"{base_url}/health", method="POST", payload={})
    assert post_status == 405
    assert post_body["error"] == "method_not_allowed"

    trade_status, _ = _request(f"{base_url}/trade", method="POST", payload={})
    assert trade_status == 405


def test_http_rejects_malformed_json_body(base_url: str) -> None:
    request = urllib.request.Request(
        f"{base_url}/decide", data=b"{not json", method="POST"
    )
    request.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    try:
        assert excinfo.value.code == 400
        assert json.loads(excinfo.value.read())["error"] == "invalid_json"
    finally:
        excinfo.value.close()


def test_contract_date_type_is_accepted() -> None:
    # Callers inside the process should not have to serialise dates first.
    payload = _payload()
    payload["as_of"] = date(2024, 3, 20)
    assert decide(payload).status == 200
