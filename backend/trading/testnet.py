"""Authenticated client for Deribit's TEST exchange — and only the test one.

test.deribit.com is a full exchange running the production matching engine
against paper money: free API keys, real order books, real fills, zero
financial risk. It is the free path from "pricing lab" to "trading system".

Hard safety property: this module CANNOT reach production. The base URL is
hardcoded to the testnet host and verified at construction; there is no
parameter that points it anywhere else. The read-only production client lives
in backend/quant/deribit.py and never sends an Authorization header; this one
authenticates and mutates exchange state, so the blast radius is bounded by
being structurally unable to touch real money.

Credentials come from the environment (a .env at the repo root is loaded if
python-dotenv is installed):

    DERIBIT_TESTNET_KEY=...      # from test.deribit.com -> settings -> API
    DERIBIT_TESTNET_SECRET=...

Create them yourself on test.deribit.com (an account there is free); this
codebase never stores them, and MissingCredentials says exactly what to do
when they are absent.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

TESTNET_BASE = "https://test.deribit.com/api/v2/"
_TIMEOUT = 10.0


class TradingError(RuntimeError):
    pass


class MissingCredentials(TradingError):
    def __init__(self) -> None:
        super().__init__(
            "DERIBIT_TESTNET_KEY / DERIBIT_TESTNET_SECRET are not set. "
            "Create a free account at https://test.deribit.com, generate an "
            "API key under Settings -> API, and put both values in the "
            "environment or a .env file at the repo root. Paper money only — "
            "this client is structurally unable to reach the real exchange.")


class TestnetAPIError(TradingError):
    def __init__(self, method: str, code: Any, message: str) -> None:
        super().__init__(f"testnet {method}: [{code}] {message}")
        self.code = code


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:                                   # pragma: no cover
        pass


class TestnetClient:
    """Minimal authenticated REST client. Synchronous, stdlib-only transport.

    REST polling at a 1-2 s cadence is deliberately the v1 transport: the
    quoting loop this serves re-quotes on the order of seconds, testnet does
    not reward latency, and a websocket session adds failure modes before it
    adds information. The `transport` hook exists so tests can run the whole
    order lifecycle offline against a fake exchange.
    """

    def __init__(self, key: str | None = None, secret: str | None = None,
                 transport=None) -> None:
        _load_dotenv()
        self.key = key or os.environ.get("DERIBIT_TESTNET_KEY", "")
        self.secret = secret or os.environ.get("DERIBIT_TESTNET_SECRET", "")
        self.base = TESTNET_BASE
        assert "test.deribit.com" in self.base, (
            "refusing to construct: base URL is not the test exchange")
        self._token = ""
        self._token_expiry = 0.0
        self._transport = transport or self._http

    # -- transport --------------------------------------------------------- #

    def _http(self, method: str, params: dict, token: str) -> Any:
        query = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})
        url = self.base + method + (f"?{query}" if query else "")
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                raise TradingError(f"testnet {method}: HTTP {exc.code}") from exc
        if "error" in body:
            err = body["error"] or {}
            raise TestnetAPIError(method, err.get("code"),
                                  err.get("message", "unknown"))
        return body["result"]

    def _public(self, method: str, **params: Any) -> Any:
        return self._transport(f"public/{method}", params, "")

    def _private(self, method: str, **params: Any) -> Any:
        return self._transport(f"private/{method}", params, self._bearer())

    def _bearer(self) -> str:
        if not self.key or not self.secret:
            raise MissingCredentials()
        if time.monotonic() < self._token_expiry - 30.0:
            return self._token
        result = self._transport("public/auth", {
            "grant_type": "client_credentials",
            "client_id": self.key, "client_secret": self.secret}, "")
        self._token = result["access_token"]
        self._token_expiry = time.monotonic() + float(result["expires_in"])
        return self._token

    # -- market data (public, no auth) ------------------------------------- #

    def ticker(self, instrument: str) -> dict:
        return self._public("ticker", instrument_name=instrument)

    def instrument(self, instrument: str) -> dict:
        return self._public("get_instrument", instrument_name=instrument)

    # -- orders (private) --------------------------------------------------- #

    def limit_order(self, instrument: str, side: str, amount: float,
                    price: float, label: str = "") -> dict:
        """Post-only limit order. Post-only because a MAKER strategy that
        crosses the spread by accident is a different strategy."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy|sell, got {side!r}")
        return self._private(
            side, instrument_name=instrument, amount=amount, price=price,
            type="limit", post_only="true", label=label or None)

    def cancel(self, order_id: str) -> dict:
        return self._private("cancel", order_id=order_id)

    def cancel_all(self, instrument: str | None = None) -> Any:
        if instrument:
            return self._private("cancel_all_by_instrument",
                                 instrument_name=instrument)
        return self._private("cancel_all")

    def open_orders(self, instrument: str) -> list[dict]:
        return self._private("get_open_orders_by_instrument",
                             instrument_name=instrument)

    def position(self, instrument: str) -> dict:
        return self._private("get_position", instrument_name=instrument)

    def account_summary(self, currency: str = "BTC") -> dict:
        return self._private("get_account_summary", currency=currency)
