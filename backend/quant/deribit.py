"""Deribit public v2 REST client, with an on-disk snapshot cache.

WHY THIS EXISTS
---------------
Every other number in this repository is synthetic: GBM paths, Monte Carlo
labels, a WGAN path generator, and (in `market_data.py`) retail equity quotes
from Yahoo, which give one last price and no book at all. None of that exercises
market *microstructure* - a two-sided book, a tick size, a fee schedule, a
forward curve, or the no-arbitrage violations that real quotes actually contain.

Deribit's public v2 REST API serves live institutional crypto-options data with
no API key, no auth header, and no licensing wall. This module is the read path.
It NEVER sends credentials and only ever touches `/api/v2/public/*`; there is no
code here that could place an order.

THE SNAPSHOT CACHE - AND WHY IT IS THE POINT
--------------------------------------------
This project's ethos is reproducible offline execution. A live REST call is the
opposite: the answer changes every second, so a test written against it either
asserts nothing or fails at random. So the client's real product is a
`Snapshot` - a single JSON document holding the exact payloads the exchange
returned at one instant, plus the wall-clock at which they were captured.

Everything downstream (`surface.py`, `tests/test_market.py`) consumes a
`Snapshot`, never the network. `scripts/fetch_chain.py --offline` and the entire
test suite run with the machine unplugged. Refreshing the fixture is an explicit,
deliberate act.

SNAPSHOT SCHEMA (version 1)
---------------------------
    {
      "schema_version": 1,
      "captured_at":    <float, unix seconds UTC - see below>,
      "captured_at_iso":<str, human-readable>,
      "currency":       "BTC",
      "index_price":    <float, USD per coin, from /public/get_index_price>,
      "instruments":    [ <raw /public/get_instruments rows, kind=option> ],
      "book_summary":   [ <raw /public/get_book_summary_by_currency rows> ],
      "futures":        [ <raw /public/get_instruments rows, kind=future> ],
      "futures_book":   [ <raw /public/get_book_summary_by_currency, kind=future> ],
      "order_books":    { <instrument_name>: <raw /public/get_order_book row> },
      "meta":           { "endpoint_latency_ms": {...}, "base_url": ..., ... }
    }

`captured_at` is taken from the exchange's own `creation_timestamp` on the book
summary rows, NOT from the local clock. Measured on the committed fixture, those
row timestamps span 9 ms across all 836 instruments, so the median is an
unambiguous "as of" for the whole chain; a local clock could be off by minutes
and would silently corrupt every time-to-expiry.

MEASURED ENDPOINT BEHAVIOUR (2026-08-05, BTC)
---------------------------------------------
All four bulk endpoints answered HTTP 200 in 119-258 ms from this machine.

  /public/get_instruments?currency=BTC&kind=option&expired=false
      836 rows, 12 distinct expiries, 418 strikes, every strike carrying BOTH a
      call and a put. Every row: contract_size 1.0, tick_size 0.0001,
      maker_commission == taker_commission == 0.0003, quote_currency "BTC",
      settlement_currency "BTC", counter_currency "USD",
      instrument_type "reversed" (i.e. inverse - see surface.py for what that
      does to the premium). settlement_period splits 130 day / 120 week /
      586 month. Expiries are all at 08:00:00 UTC.

  /public/get_book_summary_by_currency?currency=BTC&kind=option
      836 rows, one per instrument, in ONE request. Carries bid_price,
      ask_price, mark_price, mark_iv, underlying_price, underlying_index,
      open_interest, volume, interest_rate. Does NOT carry bid_iv/ask_iv (those
      exist only on /public/ticker) - which is fine, because inverting the book
      ourselves is the entire exercise. `interest_rate` was 0.0 on every row.

  /public/get_instruments?currency=BTC&kind=future    -> 13 rows (12 dated + PERPETUAL)
  /public/get_book_summary_by_currency?...kind=future -> the futures mid, i.e. the
      market forward per option expiry, independent of the options book.

  /public/ticker?instrument_name=...      full quote + Deribit's own greeks and
      bid_iv/ask_iv/mark_iv. One instrument per call, so it is used only for
      spot-checks, not for the 836-instrument chain.

  /public/get_order_book?instrument_name=...&depth=N   L2 ladder, plus min_price
      and max_price (the exchange's own quote band).

ERROR SHAPE
-----------
Bad parameters return HTTP 400 with a JSON-RPC error body, verified:
    {"jsonrpc":"2.0","error":{"code":-32602,
      "data":{"reason":"invalid currency","param":"currency"},
      "message":"Invalid params"}}
`DeribitAPIError` surfaces `reason`/`param` rather than a bare "HTTP 400".

References
----------
Deribit API v2 documentation, https://docs.deribit.com/ (public methods).
Deribit, "Options" contract specification (European, cash-settled, inverse).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "API_BASE",
    "SECONDS_PER_YEAR",
    "SNAPSHOT_SCHEMA_VERSION",
    "DeribitError",
    "DeribitAPIError",
    "SnapshotError",
    "Snapshot",
    "DeribitClient",
    "save_snapshot",
    "load_snapshot",
    "find_snapshots",
    "latest_snapshot",
]

API_BASE = "https://www.deribit.com/api/v2/public/"

#: ACT/365 fixed. Not a guess: reproducing Deribit's published ``mark_iv`` from
#: ``mark_price`` across all 836 instruments gives a median error of +0.0001 vol
#: points on a 365-day year, +0.0144 on 365.25, and -0.2727 on 360.
SECONDS_PER_YEAR = 365.0 * 86_400.0

SNAPSHOT_SCHEMA_VERSION = 1

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_RETRIES = 3
_USER_AGENT = "neural-options-lab/1.0 (+https://github.com/; research; public endpoints only)"

#: Committed fixtures live here and are named ``deribit_snapshot_<UTC>.json``.
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
SNAPSHOT_GLOB = "deribit_snapshot_*.json"


class DeribitError(RuntimeError):
    """Base class for every failure raised by this module."""


class DeribitAPIError(DeribitError):
    """The exchange answered, but with a JSON-RPC error."""

    def __init__(self, method: str, code: int | None, message: str,
                 reason: str | None = None, param: str | None = None) -> None:
        detail = f"{message}"
        if reason:
            detail += f" ({reason}"
            detail += f" on {param!r})" if param else ")"
        super().__init__(f"deribit {method}: [{code}] {detail}")
        self.method = method
        self.code = code
        self.message = message
        self.reason = reason
        self.param = param


class SnapshotError(DeribitError):
    """A snapshot file is missing, unreadable, or of an unknown schema."""


# --------------------------------------------------------------------------- #
#  Snapshot
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Snapshot:
    """One instant of the Deribit option chain, frozen to disk.

    Attributes mirror the schema documented in the module docstring. The raw
    exchange payloads are kept verbatim: a snapshot should be auditable against
    what the venue actually said, not against what we chose to keep.
    """

    captured_at: float
    currency: str
    index_price: float
    instruments: list[dict[str, Any]]
    book_summary: list[dict[str, Any]]
    futures: list[dict[str, Any]] = field(default_factory=list)
    futures_book: list[dict[str, Any]] = field(default_factory=list)
    order_books: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience indexes ------------------------------------------------ #

    def instrument_by_name(self) -> dict[str, dict[str, Any]]:
        return {row["instrument_name"]: row for row in self.instruments}

    def book_by_name(self) -> dict[str, dict[str, Any]]:
        return {row["instrument_name"]: row for row in self.book_summary}

    def futures_book_by_name(self) -> dict[str, dict[str, Any]]:
        return {row["instrument_name"]: row for row in self.futures_book}

    def expiries(self) -> list[int]:
        """Sorted distinct expiration timestamps (milliseconds since epoch)."""
        return sorted({int(row["expiration_timestamp"]) for row in self.instruments})

    @property
    def captured_at_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.captured_at))

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "captured_at": self.captured_at,
            "captured_at_iso": self.captured_at_iso,
            "currency": self.currency,
            "index_price": self.index_price,
            "instruments": self.instruments,
            "book_summary": self.book_summary,
            "futures": self.futures,
            "futures_book": self.futures_book,
            "order_books": self.order_books,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Snapshot":
        version = payload.get("schema_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError(
                f"snapshot schema_version {version!r}, expected "
                f"{SNAPSHOT_SCHEMA_VERSION}")
        for key in ("captured_at", "currency", "index_price", "instruments",
                    "book_summary"):
            if key not in payload:
                raise SnapshotError(f"snapshot is missing required key {key!r}")
        return cls(
            captured_at=float(payload["captured_at"]),
            currency=str(payload["currency"]),
            index_price=float(payload["index_price"]),
            instruments=list(payload["instruments"]),
            book_summary=list(payload["book_summary"]),
            futures=list(payload.get("futures", [])),
            futures_book=list(payload.get("futures_book", [])),
            order_books=dict(payload.get("order_books", {})),
            meta=dict(payload.get("meta", {})),
        )

    def summary(self) -> dict[str, Any]:
        """Counts a human (or a docstring) actually wants to see."""
        two_sided = sum(1 for r in self.book_summary
                        if r.get("bid_price") and r.get("ask_price"))
        return {
            "captured_at_iso": self.captured_at_iso,
            "currency": self.currency,
            "index_price": self.index_price,
            "n_instruments": len(self.instruments),
            "n_expiries": len(self.expiries()),
            "n_strikes": len({row["strike"] for row in self.instruments}),
            "n_quoted": len(self.book_summary),
            "n_two_sided": two_sided,
            "n_futures": len(self.futures),
            "n_order_books": len(self.order_books),
        }


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #

class DeribitClient:
    """Read-only client for Deribit's public v2 REST endpoints.

    Only ``/api/v2/public/*`` is reachable from here. No API key is accepted,
    no ``Authorization`` header is ever built, and no method mutates exchange
    state. `timeout` bounds every socket operation; `retries` applies only to
    transport failures and HTTP 5xx, never to a 4xx (a bad parameter will not
    become good on a second attempt).
    """

    def __init__(self, base_url: str = API_BASE, *,
                 timeout: float = _DEFAULT_TIMEOUT,
                 retries: int = _DEFAULT_RETRIES,
                 backoff: float = 0.5,
                 min_interval: float = 0.05,
                 user_agent: str = _USER_AGENT) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.min_interval = float(min_interval)
        self.user_agent = user_agent
        self.latency_ms: dict[str, float] = {}
        self._last_call = 0.0

    # -- transport ---------------------------------------------------------- #

    def _rpc(self, method: str, **params: Any) -> Any:
        query = "&".join(f"{k}={v}" for k, v in params.items()
                         if v is not None)
        url = self.base_url + method + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent,
                          "Accept": "application/json"})

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            # Be polite: the public tier is generous but not infinite, and
            # snapshot() issues a burst of per-instrument order-book calls.
            gap = time.monotonic() - self._last_call
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)

            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.latency_ms[method] = (time.perf_counter() - start) * 1e3
                self._last_call = time.monotonic()
                if "error" in body:
                    err = body["error"] or {}
                    data = err.get("data") or {}
                    raise DeribitAPIError(method, err.get("code"),
                                          err.get("message", "unknown error"),
                                          data.get("reason"), data.get("param"))
                if "result" not in body:
                    raise DeribitError(
                        f"deribit {method}: response has neither 'result' nor "
                        f"'error' (keys: {sorted(body)})")
                return body["result"]
            except urllib.error.HTTPError as exc:
                self._last_call = time.monotonic()
                raw = exc.read()
                try:
                    err = (json.loads(raw.decode("utf-8")).get("error") or {})
                    data = err.get("data") or {}
                    api_exc = DeribitAPIError(method, err.get("code"),
                                              err.get("message", exc.reason),
                                              data.get("reason"),
                                              data.get("param"))
                except Exception:
                    api_exc = DeribitAPIError(method, exc.code, str(exc.reason))
                if 400 <= exc.code < 500:
                    raise api_exc from exc  # a bad request stays bad
                last_exc = api_exc
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as exc:
                self._last_call = time.monotonic()
                last_exc = exc

            if attempt < self.retries - 1:
                time.sleep(self.backoff * (2 ** attempt))

        raise DeribitError(
            f"deribit {method}: failed after {self.retries} attempts "
            f"({type(last_exc).__name__}: {last_exc})") from last_exc

    # -- public endpoints --------------------------------------------------- #

    def get_instruments(self, currency: str = "BTC", kind: str = "option",
                        expired: bool = False) -> list[dict[str, Any]]:
        """Every listed instrument of `kind` for `currency`."""
        return list(self._rpc("get_instruments", currency=currency, kind=kind,
                              expired="true" if expired else "false"))

    def get_book_summary(self, currency: str = "BTC",
                         kind: str = "option") -> list[dict[str, Any]]:
        """Top of book + mark + IV for every instrument, in one request."""
        return list(self._rpc("get_book_summary_by_currency",
                              currency=currency, kind=kind))

    def get_index_price(self, index_name: str = "btc_usd") -> float:
        return float(self._rpc("get_index_price",
                               index_name=index_name)["index_price"])

    def get_ticker(self, instrument_name: str) -> dict[str, Any]:
        """Full quote for one instrument, including Deribit's own greeks."""
        return dict(self._rpc("ticker", instrument_name=instrument_name))

    def get_order_book(self, instrument_name: str,
                       depth: int = 10) -> dict[str, Any]:
        """L2 ladder for one instrument, plus the exchange's min/max price band."""
        return dict(self._rpc("get_order_book",
                              instrument_name=instrument_name, depth=depth))

    # -- the thing this class is for ---------------------------------------- #

    def snapshot(self, currency: str = "BTC", *,
                 order_book_depth: int = 10,
                 order_book_names: Sequence[str] | None = None,
                 n_order_books: int = 12) -> Snapshot:
        """Capture the whole chain in four bulk calls (+ L2 for a few names).

        `order_book_names` defaults to the `n_order_books` most-traded
        instruments by 24h volume, so the L2 sample is where the liquidity is
        rather than an arbitrary alphabetical slice. Pass an explicit sequence
        (or `n_order_books=0`) to control the request count.
        """
        index_name = f"{currency.lower()}_usd"
        index_price = self.get_index_price(index_name)
        instruments = self.get_instruments(currency, "option", expired=False)
        book_summary = self.get_book_summary(currency, "option")
        futures = self.get_instruments(currency, "future", expired=False)
        futures_book = self.get_book_summary(currency, "future")

        captured_at = _captured_at(book_summary)

        if order_book_names is None:
            ranked = sorted(book_summary,
                            key=lambda r: -float(r.get("volume") or 0.0))
            order_book_names = [r["instrument_name"]
                                for r in ranked[:max(0, n_order_books)]]
        order_books: dict[str, dict[str, Any]] = {}
        for name in order_book_names:
            try:
                order_books[name] = self.get_order_book(name, order_book_depth)
            except DeribitError:
                continue  # an instrument can be pulled between the two calls

        return Snapshot(
            captured_at=captured_at,
            currency=currency.upper(),
            index_price=index_price,
            instruments=instruments,
            book_summary=book_summary,
            futures=futures,
            futures_book=futures_book,
            order_books=order_books,
            meta={
                "base_url": self.base_url,
                "endpoint_latency_ms": {k: round(v, 1)
                                        for k, v in self.latency_ms.items()},
                "order_book_depth": order_book_depth,
                "fetched_by": "backend.quant.deribit.DeribitClient.snapshot",
                "captured_at_source": "median book_summary.creation_timestamp",
            },
        )


def _captured_at(book_summary: Iterable[Mapping[str, Any]]) -> float:
    """Exchange-side capture time, in unix seconds.

    Uses the median `creation_timestamp` of the book summary rows rather than
    the local clock: a wrong local clock would silently bias every time to
    expiry, and the row timestamps span only milliseconds.
    """
    stamps = sorted(float(r["creation_timestamp"]) / 1e3
                    for r in book_summary if r.get("creation_timestamp"))
    if not stamps:
        return time.time()
    return stamps[len(stamps) // 2]


# --------------------------------------------------------------------------- #
#  On-disk snapshot cache
# --------------------------------------------------------------------------- #

def save_snapshot(snapshot: Snapshot, path: str | Path | None = None, *,
                  directory: str | Path | None = None) -> Path:
    """Write `snapshot` as compact JSON. Returns the path written.

    With no explicit `path`, the file is named
    ``deribit_snapshot_<currency>_<YYYYmmddTHHMMSSZ>.json`` in `directory`
    (default ``artifacts/``), so snapshots sort chronologically and never
    collide.
    """
    if path is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(snapshot.captured_at))
        base = Path(directory) if directory is not None else ARTIFACTS_DIR
        path = base / f"deribit_snapshot_{snapshot.currency.lower()}_{stamp}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators: the raw payloads are ~1.1 MB pretty-printed and this
    # file is committed. No indent, no sorting - byte-for-byte stability across
    # writes matters more than readability for a fixture.
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot.to_dict(), handle, separators=(",", ":"))
    return path


def load_snapshot(path: str | Path) -> Snapshot:
    """Read a snapshot written by `save_snapshot`. No network."""
    path = Path(path)
    if not path.exists():
        raise SnapshotError(f"no snapshot at {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{path} is not valid JSON: {exc}") from exc
    return Snapshot.from_dict(payload)


def find_snapshots(directory: str | Path | None = None,
                   currency: str | None = None) -> list[Path]:
    """All committed snapshot fixtures, oldest first (names sort by capture time)."""
    base = Path(directory) if directory is not None else ARTIFACTS_DIR
    if not base.exists():
        return []
    pattern = (SNAPSHOT_GLOB if currency is None
               else f"deribit_snapshot_{currency.lower()}_*.json")
    return sorted(base.glob(pattern))


def latest_snapshot(directory: str | Path | None = None,
                    currency: str | None = None) -> Snapshot:
    """Load the most recent committed snapshot. This is the offline entry point."""
    paths = find_snapshots(directory, currency)
    if not paths:
        base = Path(directory) if directory is not None else ARTIFACTS_DIR
        raise SnapshotError(
            f"no snapshot matching {SNAPSHOT_GLOB!r} in {base}; run "
            f"`python scripts/fetch_chain.py --refresh` to capture one")
    return load_snapshot(paths[-1])
