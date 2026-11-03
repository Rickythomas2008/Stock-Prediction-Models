"""Transport layer for the decision contract: ``http.server`` only, no rules.

Everything here is mechanical -- parse the request, call one pure function from
``stockpred.serving.contract``, serialise the ``ApiResponse``. Business rules and
status codes live in the contract, so this module cannot disagree with what the
tests exercise directly.

No order-placing code exists in this module, by design. If automatic submission
is ever added it must first pass ``Settings.assert_auto_order_allowed`` (broker
confirmation of the program-trading filing, CSRC 2024 No.8 art.8) and be gated
behind an explicit human confirmation; until then the only output is a
human-executable order sheet.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from stockpred.serving.contract import (
    ApiResponse,
    decide,
    error_response,
    health,
    legacy_trade,
)

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"

#: Paths that exist, mapped to the methods they accept. One table so a 404 and a
#: 405 can never disagree about whether a path exists.
_ROUTES: dict[str, tuple[str, ...]] = {
    "/health": ("GET",),
    "/decide": ("POST",),
    "/trade": ("GET",),
}

#: Refuse oversized bodies before reading them: ``Content-Length`` is attacker
#: controlled, and neither endpoint needs more than a few MB of bars.
_MAX_BODY_BYTES = 8 * 1024 * 1024


def _normalise_path(raw_path: str) -> str:
    path = urlsplit(raw_path).path
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


class DecisionRequestHandler(BaseHTTPRequestHandler):
    """Routes three endpoints. Holds no state between requests."""

    server_version = "stockpred-serving"
    protocol_version = "HTTP/1.1"
    # A client that opens a connection and stalls must not hold a thread forever.
    timeout = 15

    def do_GET(self) -> None:
        path = _normalise_path(self.path)
        allowed = _ROUTES.get(path)
        if allowed is None:
            self._respond(error_response(404, "not_found", detail=path))
        elif "GET" not in allowed:
            self._respond(
                error_response(405, "method_not_allowed", detail=f"{path} accepts {allowed[0]}"),
                allow=", ".join(allowed),
            )
        elif path == "/trade":
            self._respond(_trade_from_query(urlsplit(self.path).query))
        else:
            self._respond(health())

    def do_POST(self) -> None:
        path = _normalise_path(self.path)
        allowed = _ROUTES.get(path)
        if allowed is None:
            self._respond(error_response(404, "not_found", detail=path))
            return
        if "POST" not in allowed:
            self._respond(
                error_response(405, "method_not_allowed", detail=f"{path} accepts {allowed[0]}"),
                allow=", ".join(allowed),
            )
            return
        parsed, failure = self._read_json_body()
        if failure is not None:
            self._respond(failure)
            return
        self._respond(decide(parsed))

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request stderr logging; a server must not spam its caller's output."""

    def _read_json_body(self) -> tuple[Any, ApiResponse | None]:
        """Read the body, returning either the parsed payload or the reply to send."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, error_response(400, "invalid_request", detail="bad Content-Length")
        if length > _MAX_BODY_BYTES:
            return None, error_response(
                413, "payload_too_large", detail=f"limit is {_MAX_BODY_BYTES} bytes"
            )
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            # An empty body is a well-formed request for nothing: let the
            # contract report the first missing field.
            return {}, None
        try:
            return json.loads(raw), None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, error_response(400, "invalid_json", detail="body must be JSON")

    def _respond(self, response: ApiResponse, *, allow: str | None = None) -> None:
        payload = json.dumps(response.body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", _JSON_CONTENT_TYPE)
        self.send_header("Content-Length", str(len(payload)))
        if allow is not None:
            self.send_header("Allow", allow)
        self.end_headers()
        self.wfile.write(payload)


def _trade_from_query(query: str) -> ApiResponse:
    """Translate ``GET /trade?...`` into the legacy contract's payload.

    The old client sent ``data=[13.1, 13407500]`` as a JSON string in the query
    string, so the same parsing is used here rather than inventing new parameter
    names that no deployed caller knows.
    """
    params = parse_qs(query, keep_blank_values=True)

    raw_data = params.get("data")
    if not raw_data:
        return error_response(400, "missing_field", field="data")
    try:
        data = json.loads(raw_data[-1])
    except json.JSONDecodeError:
        return error_response(
            400, "invalid_data", field="data", detail="data must be JSON, e.g. [13.1, 13407500]"
        )

    payload: dict[str, Any] = {"data": data}
    for name in ("history", "capital", "position"):
        values = params.get(name)
        if not values:
            continue
        try:
            payload[name] = json.loads(values[-1])
        except json.JSONDecodeError:
            return error_response(400, "invalid_data", field=name, detail="must be JSON")
    return legacy_trade(payload)


def create_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Bind a server but do not start it.

    ``port=0`` lets the OS pick a free port, which is what keeps parallel test
    runs from colliding; read the real port from ``server.server_address[1]``.
    The caller owns ``serve_forever`` / ``shutdown`` / ``server_close``.
    """
    return ThreadingHTTPServer((host, port), DecisionRequestHandler)


def serve(host: str = "127.0.0.1", port: int = 8005) -> None:
    """Run the server in the foreground until interrupted (manual use only).

    Blocking on purpose: a decision service started implicitly by an import is a
    service nobody remembers is running.
    """
    server = create_server(host, port)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    print(f"stockpred serving on http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["DecisionRequestHandler", "create_server", "serve"]
