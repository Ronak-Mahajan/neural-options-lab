"""FastAPI layer serving the neural pricer, the Monte Carlo engine, and the
static dashboard.

Run from the repo root:
    python -m uvicorn backend.api.main:app --port 8000
"""

from __future__ import annotations

import anyio
import asyncio
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import numpy as np

from fastapi import (FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.routing import Match, Mount
from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      field_validator)

from ..quant.engine import ZERO_DTE_CUTOFF, PricingEngine, time_call
from ..quant.hedging import HedgingEngine
from ..quant.iv_surface import IVSurface
from ..quant.monte_carlo import MCResult

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
ARTIFACTS = ROOT / "artifacts"
EVAL_FILE = ARTIFACTS / "eval.json"

# ZERO_DTE_CUTOFF (12 trading days, from the engine) separates two contracts:
# an arithmetic-average Asian under GBM above it, a European under rough
# Bergomi at or below it. At S=K=100, sigma=0.25, r=0.04 the price jumps +45.6%
# (1.35400 -> 1.97183) across 8.1e-5 years, about 42 minutes, so every response
# carries a `regime` field and the Monte Carlo benchmark switches measure too.
ZERO_DTE_MIN_MATURITY = 1.0 / 252.0      # 0DTE surrogate's trained floor
ZERO_DTE_MONEYNESS = (0.85, 1.15)
# Rough-Bergomi parameters for a 0DTE checkpoint that carries none of its own.
# mc_reference() reads eta, rho and H from the served checkpoint's metadata.
ROUGH_ETA, ROUGH_RHO, ROUGH_H = 1.5, -0.7, 0.1

# One price-plus-Greeks frame costs 13.7 ms median and 74.7 ms at p95. A 60 Hz
# request priced on the event loop delivers 28.8 Hz and starves every other
# coroutine. At 15 Hz in a worker thread one stream uses about a fifth of one
# core (15 x 13.7 ms per second), and MAX_STREAM_CLIENTS bounds how many run
# at once.
MAX_STREAM_HZ = 15
MAX_STREAM_CLIENTS = 3
_stream_clients = 0

# Handlers are sync `def`s on Starlette's 40-thread pool, and nothing else
# serialises them inside the 512 MB container.

# Simulation gate: one Monte Carlo or batch-inference job at a time.
# /api/price, /api/convergence and /api/benchmark in flight together have
# exhausted the container's memory (exit 137).
_HEAVY_JOB_GATE = threading.Semaphore(1)
HEAVY_JOB_TIMEOUT_S = 30.0
HEAVY_QUEUE_MAX = 8
# Inference gate: four attribution or implied-vol calls at a time. In a local
# probe forty concurrent attributions (the width of the pool) raised resident
# memory by 193 MB over an idle process near 290 MB; four raised it by 23 MB.
_LIGHT_JOB_GATE = threading.BoundedSemaphore(4)
LIGHT_JOB_TIMEOUT_S = 10.0
LIGHT_QUEUE_MAX = 8


class _WaitQueue:
    """Number of requests parked on one gate."""

    def __init__(self) -> None:
        self.waiting = 0
        self.lock = threading.Lock()


_HEAVY_QUEUE = _WaitQueue()
_LIGHT_QUEUE = _WaitQueue()


def _acquire(gate: threading.Semaphore, queue: _WaitQueue, max_waiters: int,
             timeout_s: float, what: str) -> None:
    """Take a slot, or raise 503 with Retry-After.

    Every waiter holds a pool thread, so the queue is capped: a full queue
    refuses at once, and a wait past the timeout refuses then. One job plus
    eight waiters on the simulation gate and four plus eight on the inference
    gate leave 19 of the 40 threads for stream frames and ungated routes.
    /api/health is a coroutine and needs none.
    """
    with queue.lock:
        if queue.waiting >= max_waiters:
            raise HTTPException(
                503, f"{what} queue is full; retry shortly",
                headers={"Retry-After": "5"})
        queue.waiting += 1
    try:
        acquired = gate.acquire(timeout=timeout_s)
    finally:
        with queue.lock:
            queue.waiting -= 1
    if not acquired:
        raise HTTPException(
            503, f"{what} queue is saturated; retry shortly",
            headers={"Retry-After": "10"})


@contextmanager
def heavy_job() -> Iterator[None]:
    _acquire(_HEAVY_JOB_GATE, _HEAVY_QUEUE, HEAVY_QUEUE_MAX,
             HEAVY_JOB_TIMEOUT_S, "Simulation")
    try:
        yield
    finally:
        _HEAVY_JOB_GATE.release()


@contextmanager
def light_job() -> Iterator[None]:
    _acquire(_LIGHT_JOB_GATE, _LIGHT_QUEUE, LIGHT_QUEUE_MAX,
             LIGHT_JOB_TIMEOUT_S, "Inference")
    try:
        yield
    finally:
        _LIGHT_JOB_GATE.release()


class _ResponseCache:
    """Bounded LRU of serialised JSON bodies for the seeded endpoints, whose
    response is a function of the request alone."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._items: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            body = self._items.get(key)
            if body is not None:
                self._items.move_to_end(key)
            return body

    def put(self, key: tuple, body: bytes) -> None:
        with self._lock:
            self._items[key] = body
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)


app = FastAPI(title="Neural Options Lab", version="1.0.0",
              description="Neural surrogate vs Monte Carlo for arithmetic "
                          "Asian options (maturity > 12/252) and European "
                          "options under rough Bergomi (maturity <= 12/252)")

# The desk-note body is the largest request the dashboard sends, and the
# field caps in RiskReportRequest keep it under 2 KB.
MAX_BODY_BYTES = 64 * 1024


class BodySizeLimit:
    """Refuse request bodies over `max_bytes` with 413.

    A declared Content-Length over the cap is refused before the body is
    read. A body with no declared length (chunked transfer) is counted as it
    arrives; the 413 is raised from `receive`, inside the route's body read,
    and Starlette's exception middleware renders it.
    """

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        detail = f"request body exceeds {self.max_bytes} bytes"
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": detail}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def counted_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, detail)
            return message

        await self.app(scope, counted_receive, send)


#: Every source the dashboard and the methodology page load, measured with a
#: headless Chromium pass over all four tabs, the hedging run, the live stream
#: and both WebGL surfaces (zero violations):
#:   * scripts: this origin and the pinned Plotly build on cdn.plot.ly. Plotly
#:     2.35.2, the gl3d surfaces included, runs without 'unsafe-eval'.
#:   * styles: 'unsafe-inline' is required. The page carries style attributes,
#:     the methodology page an inline <style> block, and Plotly writes its own
#:     <style> element and inline styles at runtime.
#:   * fonts: the Google Fonts stylesheet and its font files.
#:   * images: this origin and the data: URI favicon.
#:   * connect: 'self', which covers fetch and the same-origin ws:// or wss://
#:     stream socket.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://cdn.plot.ly",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": ("camera=(), microphone=(), geolocation=(), "
                           "payment=(), usb=()"),
    # Browsers honour this only over HTTPS, so local http runs are unaffected.
    "Strict-Transport-Security": "max-age=31536000",
}


# FastAPI's interactive API docs load Swagger UI and ReDoc from
# cdn.jsdelivr.net and start them with an inline script, which the page policy
# above blocks. Those pages get every other header and no CSP.
API_DOC_PATHS = frozenset(p for p in (app.docs_url, app.redoc_url,
                                      app.swagger_ui_oauth2_redirect_url) if p)


class SecurityHeaders:
    """Add SECURITY_HEADERS to every HTTP response that does not set them.

    The Content-Security-Policy is left off the API doc pages (API_DOC_PATHS).
    Pure ASGI, like BodySizeLimit, so streamed bodies (the desk note) pass
    through unbuffered."""

    def __init__(self, app, headers: dict[str, str] = SECURITY_HEADERS,
                 no_csp_paths: frozenset[str] = API_DOC_PATHS) -> None:
        self.app = app
        self.raw = [(k.lower().encode("latin-1"), v.encode("latin-1"))
                    for k, v in headers.items()]
        self.raw_no_csp = [h for h in self.raw if h[0] != b"content-security-policy"]
        self.no_csp_paths = no_csp_paths

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = self.raw_no_csp if scope.get("path") in self.no_csp_paths else self.raw

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                headers.extend(h for h in extra if h[0] not in present)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


# The dashboard is served by this app, so browsers need no cross-origin access.
# The allow-list is the deployed origin plus localhost, for a frontend dev
# server on another port. A wildcard would let any third-party page drive its
# visitors' browsers against the Monte Carlo endpoints. The stream socket
# applies the same list (CORS does not cover WebSocket handshakes).
ALLOWED_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000",
                   "https://neural-options-lab.onrender.com")

app.add_middleware(BodySizeLimit)
# CORS is added after BodySizeLimit, so it wraps it and a 413 carries the CORS
# headers too. SecurityHeaders is added last and wraps both.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.add_middleware(SecurityHeaders)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request,
                           exc: RequestValidationError) -> JSONResponse:
    """422 with the location, message and type of each error. The rejected
    input is left out: a NaN in it cannot be serialised as JSON, and echoing
    it would reflect the whole request body back to the sender."""
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


try:
    ENGINE: PricingEngine | None = PricingEngine()
except FileNotFoundError:
    ENGINE = None


def engine() -> PricingEngine:
    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m backend.quant.train")
    return ENGINE


# The main surrogate's trained box, read from the checkpoint's metadata (the
# block /api/model-info publishes). The literals apply only when no checkpoint
# is loaded, and every pricing route answers 503 in that case.
_RANGES = ENGINE.meta["param_ranges"] if ENGINE is not None else {}
ASIAN_MONEYNESS = tuple(_RANGES.get("moneyness", (0.5, 2.0)))
ASIAN_MIN_MATURITY = float(_RANGES.get("maturity", (0.05, 2.0))[0])
SIGMA_RANGE = tuple(_RANGES.get("sigma", (0.05, 0.80)))
RATE_RANGE = tuple(_RANGES.get("rate", (0.0, 0.10)))

# One policy per market dynamics the dashboard can simulate, each trained
# under the measure it serves. A policy evaluated under another measure is a
# regime test, so a missing checkpoint has no substitute: /api/hedge answers
# 503 for that dynamics.
HEDGE_DYNAMICS = {
    "rough": {"checkpoint": "hedger_rbergomi_jumps.pt",
              "measure": "rbergomi_jumps",
              "label": "rough Bergomi + jumps (SPY-calibrated)"},
    "gbm": {"checkpoint": "hedger_gbm.pt", "measure": "gbm",
            "label": "Black-Scholes (GBM)"},
}


def load_hedgers(artifacts: Path) -> dict[str, HedgingEngine]:
    """The served policies found under `artifacts`, keyed by dynamics."""
    hedgers: dict[str, HedgingEngine] = {}
    for key, spec in HEDGE_DYNAMICS.items():
        try:
            hedgers[key] = HedgingEngine(artifacts / spec["checkpoint"])
        except FileNotFoundError as exc:
            logger.warning("no hedging policy for %r dynamics: %s", key, exc)
    return hedgers


HEDGERS = load_hedgers(ARTIFACTS)

# Arbitrage-free implied-volatility surface for the 0DTE regime (a small
# network trained against the served ensemble with autograd butterfly and
# calendar penalties). Optional: the dashboard hides the panel if absent.
try:
    IV_SURFACE: IVSurface | None = IVSurface.load()
except FileNotFoundError:
    IV_SURFACE = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def regime_for(maturity: float) -> str:
    """The surrogate, and with it the contract, that serves this maturity."""
    if maturity <= ZERO_DTE_CUTOFF and ENGINE is not None and ENGINE.has_0dte:
        return "rough_bergomi_european"
    return "asian_gbm"


def validate_domain(req: "OptionParams | ImpliedVolRequest") -> str:
    """Raise 422 for a request outside the trained box of the model that
    serves it; return the serving regime so the handler can label its reply.

    Outside the box the surrogates extrapolate and still answer:

      * maturity below the 0DTE floor. At S=K=100, sigma=0.25, r=0.04 the
        surrogate plateaus near 0.399 as T -> 0 while the option decays to
        intrinsic value: T=1e-9 prices at 0.39898 against a Monte Carlo
        reference of 0.00031, about 1,300 times too high.
      * the band (12/252, 0.05): above the 0DTE cutoff, so it routes to the
        Asian net, and below that net's 0.05 training floor. The error there
        is 2.95-3.00 bps of strike against 0.83 bps at T=1.0.
      * sigma below the trained floor of 0.05, where prices violate
        no-arbitrage bounds.
    """
    m = req.spot / req.strike
    regime = regime_for(req.maturity)

    # Inputs are printed with repr and the floors as fractions, so a value a
    # hair outside a bound never prints equal to it.
    if regime == "rough_bergomi_european":
        lo, hi = ZERO_DTE_MONEYNESS
        if not (lo <= m <= hi):
            raise HTTPException(
                422, f"0DTE engine covers moneyness S/K in [{lo}, {hi}]; "
                     f"got {m!r}")
        if req.maturity < ZERO_DTE_MIN_MATURITY:
            raise HTTPException(
                422, f"maturity {req.maturity!r} years is below the 0DTE "
                     f"surrogate's floor of 1/252 years (one trading day); "
                     f"the model extrapolates to a price far above intrinsic "
                     f"value there")
    else:
        lo, hi = ASIAN_MONEYNESS
        if not (lo <= m <= hi):
            raise HTTPException(
                422, f"moneyness S/K outside trained domain [{lo}, {hi}]; "
                     f"got {m!r}")
        if ZERO_DTE_CUTOFF < req.maturity < ASIAN_MIN_MATURITY:
            raise HTTPException(
                422, f"maturity {req.maturity!r} years lies between the 0DTE "
                     f"cutoff (12/252 years) and the Asian surrogate's floor "
                     f"({ASIAN_MIN_MATURITY} years); no surrogate is valid "
                     f"there")
        if req.maturity < ASIAN_MIN_MATURITY:
            raise HTTPException(
                422, f"maturity {req.maturity!r} years is below the Asian "
                     f"surrogate's floor ({ASIAN_MIN_MATURITY} years) and no "
                     f"0DTE model is loaded")

    # An implied-vol request has no sigma: volatility is the unknown, and the
    # solver searches the whole trained range.
    sigma = getattr(req, "sigma", None)
    if sigma is not None and not (SIGMA_RANGE[0] <= sigma <= SIGMA_RANGE[1]):
        raise HTTPException(
            422, f"sigma {sigma!r} outside trained domain "
                 f"[{SIGMA_RANGE[0]}, {SIGMA_RANGE[1]}]")
    if not (RATE_RANGE[0] <= req.rate <= RATE_RANGE[1]):
        raise HTTPException(
            422, f"rate {req.rate!r} outside trained domain "
                 f"[{RATE_RANGE[0]}, {RATE_RANGE[1]}]")
    return regime


def mc_reference(req: "OptionParams", n_paths: int,
                 seed: int | None = None) -> tuple[MCResult, str]:
    """MC benchmark routed to match the surrogate pricing this request:
    Asian GBM control-variate MC above the 0DTE cutoff; rough Bergomi (the
    0DTE surrogate's teacher) at or below it, with puts via exact European
    put-call parity."""
    eng = engine()
    if req.maturity <= ZERO_DTE_CUTOFF and eng.has_0dte:
        import torch

        from ..quant.rough_vol import rough_bergomi_mc

        # Benchmark under the dynamics the served 0DTE surrogate was trained
        # on: eta, rho and H come from the checkpoint's metadata, and the
        # module constants apply to a checkpoint that carries none.
        meta0 = getattr(eng, "meta_0dte", {}) or {}
        eta = float(meta0.get("eta", ROUGH_ETA))
        rho = float(meta0.get("rho", ROUGH_RHO))
        hurst = float(meta0.get("H", ROUGH_H))
        prices, ses = rough_bergomi_mc(
            torch.tensor([float(req.spot)]),
            torch.tensor([float(req.strike)]),
            torch.tensor([float(req.maturity)]),
            torch.tensor([float(req.sigma) ** 2]),
            torch.tensor([eta]), torch.tensor([rho]),
            torch.tensor([float(req.rate)]),
            n_paths=n_paths, n_steps=50, H=hurst,
            seed=seed, return_std_error=True)
        price, se = float(prices[0]), float(ses[0])
        if req.option_type == "put":
            price += -req.spot + req.strike * float(
                np.exp(-req.rate * req.maturity))
        return MCResult(price=price, std_error=se,
                        ci_low=price - 1.96 * se, ci_high=price + 1.96 * se,
                        n_paths=n_paths, n_steps=50), "rough_bergomi"
    mc = eng.mc_price(req.spot, req.strike, req.maturity, req.sigma,
                      req.rate, n_paths=n_paths,
                      option_type=req.option_type, seed=seed)
    return mc, "asian_gbm_cv"


class ApiRequest(BaseModel):
    """Base of every request body. JSON as Python parses it admits NaN and
    Infinity; every float field, including dict values, rejects them with a
    422 before a handler or the domain gate sees one."""
    model_config = ConfigDict(allow_inf_nan=False)


# Cap on spot and strike. The price is the network output times the strike,
# and at 1e308 it overflows to inf, which JSON cannot carry.
MAX_PRICE_INPUT = 1e6


class OptionParams(ApiRequest):
    spot: float = Field(100.0, gt=0, le=MAX_PRICE_INPUT)
    strike: float = Field(100.0, gt=0, le=MAX_PRICE_INPUT)
    maturity: float = Field(1.0, gt=0, le=2.0)
    sigma: float = Field(0.25, gt=0, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    option_type: str = Field("call", pattern="^(call|put)$")


class ImpliedVolRequest(ApiRequest):
    spot: float = Field(100.0, gt=0, le=MAX_PRICE_INPUT)
    strike: float = Field(100.0, gt=0, le=MAX_PRICE_INPUT)
    maturity: float = Field(1.0, gt=0, le=2.0)
    rate: float = Field(0.04, ge=0, le=0.1)
    option_type: str = Field("call", pattern="^(call|put)$")
    price: float = Field(..., ge=0)


class PriceRequest(OptionParams):
    # The Monte Carlo engines run in fixed-size blocks, so memory is flat in
    # the path count and this cap bounds wall-clock: on the free plan's 0.1
    # CPU a 500,000-path run would hold the simulation gate, and its pool
    # thread, for tens of seconds.
    mc_paths: int = Field(50_000, ge=1_000, le=100_000)


class ConvergenceRequest(OptionParams):
    # Each entry costs a full Monte Carlo run on the request thread, so the
    # list is bounded in length, in every entry and in total.
    path_counts: list[int] = Field(
        default=[500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        min_length=1, max_length=12)

    @field_validator("path_counts")
    @classmethod
    def _bounded_paths(cls, v: list[int]) -> list[int]:
        for n in v:
            if not (100 <= n <= 100_000):
                raise ValueError(
                    f"path_counts entries must be in [100, 100000]; got {n}")
        if sum(v) > 600_000:
            raise ValueError(
                f"total simulated paths {sum(v):,} exceeds the 600,000 "
                f"budget for a single convergence request")
        return v


class SurfaceRequest(ApiRequest):
    # The grid's moneyness and maturity axes sit inside the Asian box, so the
    # schema is this route's domain gate: sigma and rate take the trained
    # ranges.
    sigma: float = Field(0.25, ge=0.05, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    strike: float = Field(100.0, gt=0, le=MAX_PRICE_INPUT)
    option_type: str = Field("call", pattern="^(call|put)$")
    resolution: int = Field(45, ge=10, le=80)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health() -> JSONResponse:
    """Render's health check. A coroutine with no I/O, so it answers from the
    event loop while every pool thread is busy. 503 without the pricing
    checkpoint, so a container that cannot price fails the check. HEAD is
    answered too, for uptime monitors that probe with it."""
    ready = ENGINE is not None
    body = {"status": "ok" if ready else "unavailable",
            "model_loaded": ready,
            "hedgers_loaded": sorted(HEDGERS),
            "iv_surface_loaded": IV_SURFACE is not None}
    return JSONResponse(body, status_code=200 if ready else 503)


_eval_report: dict | None = None
_eval_report_mtime: float | None = None
_eval_report_lock = threading.Lock()


def eval_report() -> dict | None:
    """artifacts/eval.json, parsed once per version of the file. evaluate.py
    rewrites the file in place; a read that lands mid-write keeps the last
    complete report."""
    global _eval_report, _eval_report_mtime
    try:
        mtime = EVAL_FILE.stat().st_mtime
    except FileNotFoundError:
        return None
    with _eval_report_lock:
        if mtime != _eval_report_mtime:
            try:
                _eval_report = json.loads(EVAL_FILE.read_text())
                _eval_report_mtime = mtime
            except (OSError, json.JSONDecodeError):
                logger.warning("eval.json is unreadable; keeping the last "
                               "complete report")
        return _eval_report


@app.get("/api/model-info")
def model_info() -> dict:
    from ..quant.llm import llm_available
    meta = dict(engine().meta)
    # The Report tab's lede names the writer: a language model when a key is
    # configured, the rule-based narrator otherwise.
    meta["report_writer"] = "model" if llm_available() else "rules"
    report = eval_report()
    if report is not None:
        # `checkpoint` is the sha256 of the model.pt the figures were measured
        # against, so the site can name the checkpoint its error numbers
        # describe. A key the report does not carry is skipped.
        meta["eval"] = {k: report[k] for k in
                        ("n_points", "ref_paths", "single", "ensemble",
                         "checkpoint", "gamma_reference_se_rms_bps")
                        if k in report}
    meta["zero_dte"] = zero_dte_info()
    return meta


def _checkpoint_git_stamp(path: Path) -> dict:
    """Commit and date that last touched a checkpoint, if this is a git
    checkout; the container that serves the site may not carry .git, so
    both fields are None when the lookup fails."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H %cI", "--", path.name],
            cwd=str(path.parent), capture_output=True, text=True, timeout=5)
        sha, _, date = out.stdout.strip().partition(" ")
        if out.returncode == 0 and sha:
            return {"commit": sha, "commit_date": date or None}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "commit_date": None}


def _compute_zero_dte_info() -> dict:
    """Provenance of the served 0DTE checkpoint, read from its own metadata.

    train_0dte.py writes the rough-Bergomi parameters (eta, rho, H), the
    Volterra kernel stamp, the calibrated flag and the note naming the
    calibration into artifacts/model_0dte.pt. Forward variance xi = sigma^2
    is a per-request input to the surrogate and has no entry here.

    Evaluated once, at import, by the assignment below: the metadata is fixed
    for the life of the process and the git stamp shells out, so neither runs
    on a request path.
    """
    if ENGINE is None or not getattr(ENGINE, "has_0dte", False):
        return {"available": False}
    m0 = dict(getattr(ENGINE, "meta_0dte", {}) or {})
    ckpt = ARTIFACTS / "model_0dte.pt"
    info = {
        "available": True,
        "checkpoint": ckpt.name,
        "model": m0.get("model", "rough_bergomi"),
        "n_members": m0.get("n_members"),
        "calibrated": bool(m0.get("calibrated", False)),
        "calibration_note": m0.get("calibration_note"),
        "eta": m0.get("eta"), "rho": m0.get("rho"), "H": m0.get("H"),
        "kernel": m0.get("kernel"),
        "val_rmse_bps_of_strike": m0.get("val_rmse_bps"),
        "maturity_cutoff_years": ZERO_DTE_CUTOFF,
        "maturity_floor_years": ZERO_DTE_MIN_MATURITY,
        "moneyness": list(ZERO_DTE_MONEYNESS),
        "contract": "european",
        "training": {k: m0.get(k) for k in ("epochs", "lr", "batch", "seed")},
    }
    info.update(_checkpoint_git_stamp(ckpt))
    return info


#: Immutable provenance block, resolved at import and served as-is.
_ZERO_DTE_INFO: dict = _compute_zero_dte_info()


def zero_dte_info() -> dict:
    """The provenance block built at import: the same object on every call."""
    return _ZERO_DTE_INFO


def discounted_intrinsic(spot: float, strike: float, maturity: float,
                         rate: float, option_type: str) -> float:
    """No-arbitrage floor of a European option: max(S - K e^{-rT}, 0) for a
    call, max(K e^{-rT} - S, 0) for a put. The 0DTE regime's contract must
    respect it, and the arbitrage audit in docs/no_arbitrage_surface.md
    measures the raw ensemble below it on 6.8% of its trained box, so every
    served 0DTE price is reported with this floor and a below-floor flag.
    The Asian regime's lower bound is a different quantity and is not
    reported."""
    df = math.exp(-rate * maturity)
    if option_type == "put":
        return max(strike * df - spot, 0.0)
    return max(spot - strike * df, 0.0)


def intrinsic_fields(regime: str, price: float, spot: float, strike: float,
                     maturity: float, rate: float, option_type: str) -> dict:
    """`intrinsic`, `below_intrinsic` and the shortfall in bps of strike for
    the 0DTE (European) regime; None-valued for the Asian regime, whose
    floor is not the European one. `price` is the engine's raw_price, the
    value before its floor at zero, so a put that is negative before that
    floor is reported below intrinsic."""
    if regime != "rough_bergomi_european":
        return {"intrinsic": None, "below_intrinsic": None,
                "below_intrinsic_bps_of_strike": None}
    floor = discounted_intrinsic(spot, strike, maturity, rate, option_type)
    shortfall = max(floor - price, 0.0)
    # Float noise on an exactly-at-the-floor price must not raise the flag.
    below = shortfall > 1e-9 * strike
    return {"intrinsic": floor, "below_intrinsic": below,
            "below_intrinsic_bps_of_strike": shortfall / strike * 1e4}


@app.get("/api/error-distribution")
def error_distribution() -> dict:
    """Signed pricing errors (bps of strike) of the single model vs the
    ensemble, measured against high-precision MC references by
    backend.quant.evaluate."""
    report = eval_report()
    if report is None:
        raise HTTPException(
            status_code=503,
            detail="No evaluation report. Run: "
                   "python -m backend.quant.evaluate"
        )
    return report


@app.post("/api/price")
def price(req: PriceRequest) -> dict:
    eng = engine()
    regime = validate_domain(req)

    nn_ms, nn_out = time_call(
        eng.price_with_greeks, req.spot, req.strike, req.maturity,
        req.sigma, req.rate, req.option_type)

    with heavy_job():
        t0 = time.perf_counter()
        mc, mc_engine = mc_reference(req, req.mc_paths)
        mc_ms = (time.perf_counter() - t0) * 1000.0

    diff = nn_out["price"] - mc.price
    nn = {"price": nn_out["price"], "clamped": nn_out["clamped"],
          "greeks": nn_out["greeks"], "latency_ms": nn_ms}
    # The served price is floored at zero, and `clamped` says when the floor
    # acted. The 0DTE response also carries the no-arbitrage floor and a flag
    # for a price under it, measured on the value before the zero floor.
    nn.update(intrinsic_fields(regime, nn_out["raw_price"], req.spot, req.strike,
                               req.maturity, req.rate, req.option_type))
    return {
        "regime": regime,
        "nn": nn,
        "mc": {"price": mc.price, "std_error": mc.std_error,
               "ci_low": mc.ci_low, "ci_high": mc.ci_high,
               "n_paths": mc.n_paths, "n_steps": mc.n_steps,
               "engine": mc_engine, "latency_ms": mc_ms},
        "comparison": {
            "abs_diff": abs(diff),
            # bps of strike, the unit of every figure in the README and docs.
            "diff_bps_of_strike": abs(diff) / req.strike * 1e4,
            "within_mc_ci": mc.ci_low <= nn_out["price"] <= mc.ci_high,
            "speedup": mc_ms / max(nn_ms, 1e-6),
        },
    }


@app.post("/api/implied-vol")
def implied_vol(req: ImpliedVolRequest) -> dict:
    """The volatility that reproduces a quoted price for this contract.

    The inverse of pricing: a bisection on the served model, bracketed by one
    batched sweep across its trained volatility range. It costs a handful of
    forward passes and needs no simulation.
    """
    from ..quant.solve_vol import solve_implied_vol

    eng = engine()
    validate_domain(req)

    with light_job():
        t0 = time.perf_counter()
        try:
            sol = solve_implied_vol(
                eng, req.spot, req.strike, req.maturity, req.rate, req.price,
                option_type=req.option_type)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ms = (time.perf_counter() - t0) * 1000.0

    return {
        "sigma": sol.sigma,
        "price_at_sigma": sol.price_at_sigma,
        "target_price": sol.target_price,
        "bracketed": sol.bracketed,
        "iterations": sol.iterations,
        "search_range": [sol.sigma_low, sol.sigma_high],
        "price_range": [sol.low_price, sol.high_price],
        "latency_ms": ms,
    }


def _json_body(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=False).encode()


# Every run in the convergence reply is seeded, so the reply is a function of
# the request and one computation serves each repeat of it (the default
# contract on every page load). A body is about 1.3 KB.
_CONVERGENCE_CACHE = _ResponseCache(max_entries=64)


@app.post("/api/convergence")
def convergence(req: ConvergenceRequest) -> Response:
    """MC estimate vs path count (with 95% CI) against the instant NN price."""
    eng = engine()
    validate_domain(req)
    counts = tuple(sorted(set(req.path_counts)))
    key = (req.spot, req.strike, req.maturity, req.sigma, req.rate,
           req.option_type, counts)
    body = _CONVERGENCE_CACHE.get(key)
    if body is None:
        with heavy_job():
            # A request ahead in the queue may have computed this key.
            body = _CONVERGENCE_CACHE.get(key)
            if body is None:
                body = _json_body(_convergence(eng, req, counts))
                _CONVERGENCE_CACHE.put(key, body)
    return Response(body, media_type="application/json")


def _convergence(eng: PricingEngine, req: ConvergenceRequest,
                 counts: tuple[int, ...]) -> dict:
    points = []
    mc_engine = "asian_gbm_cv"
    for n in counts:
        t0 = time.perf_counter()
        mc, mc_engine = mc_reference(req, n, seed=42)
        points.append({"n_paths": n, "price": mc.price,
                       "ci_low": mc.ci_low, "ci_high": mc.ci_high,
                       "latency_ms": (time.perf_counter() - t0) * 1000.0})

    nn_ms, nn_out = time_call(
        eng.price_with_greeks, req.spot, req.strike, req.maturity,
        req.sigma, req.rate, req.option_type)

    # A 100,000-path reference. Four times the paths would halve the interval,
    # a change too small to see on the chart, and hold the simulation gate
    # four times as long.
    ref, _ = mc_reference(req, 100_000, seed=7)
    return {"mc_points": points, "engine": mc_engine,
            "nn": {"price": nn_out["price"], "latency_ms": nn_ms},
            "reference": {"price": ref.price, "std_error": ref.std_error,
                          "n_paths": ref.n_paths}}


@app.post("/api/surface")
def surface(req: SurfaceRequest) -> dict:
    """NN price surface over (moneyness x maturity): thousands of prices in one batched forward pass to show surrogate throughput."""
    eng = engine()
    n = req.resolution
    m_axis = np.linspace(0.55, 1.95, n)
    t_axis = np.linspace(0.06, 2.0, n)
    mm, tt = np.meshgrid(m_axis, t_axis)

    spots = (mm * req.strike).ravel()
    strikes = np.full(spots.shape, req.strike)
    mats = tt.ravel()
    sigs = np.full(spots.shape, req.sigma)
    rates = np.full(spots.shape, req.rate)

    # Up to 80x80 = 6,400 rows through the 5-member ensemble is batched
    # inference, so it takes the simulation gate. The stream prices through
    # anyio.to_thread and never takes the gate, so the two cannot deadlock
    # (tests/test_api.py::test_stream_prices_while_the_heavy_gate_is_held).
    with heavy_job():
        t0 = time.perf_counter()
        prices = eng.price_batch(spots, strikes, mats, sigs, rates,
                                 option_type=req.option_type)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {"moneyness": m_axis.tolist(), "maturity": t_axis.tolist(),
            "prices": prices.reshape(n, n).tolist(),
            "n_prices": int(prices.size), "latency_ms": elapsed_ms,
            "prices_per_second": prices.size / max(elapsed_ms / 1000, 1e-9)}


@app.post("/api/benchmark")
def benchmark(req: OptionParams) -> dict:
    """Latency shoot-out: MC at increasing path budgets vs NN single-shot
    and batched inference."""
    eng = engine()
    validate_domain(req)
    with heavy_job():
        mc_rows: list[dict[str, Any]] = []
        for n in (1_000, 10_000, 100_000):
            ms, out = time_call(mc_reference, req, n, repeats=2)
            res, mc_eng = out
            label = "rBergomi MC" if mc_eng == "rough_bergomi" else "MC"
            mc_rows.append({"label": f"{label} {n:,} paths", "latency_ms": ms,
                            "std_error": res.std_error})

        nn_rows: list[dict[str, Any]] = []
        single_ms, _ = time_call(eng.price_with_greeks, req.spot, req.strike,
                                 req.maturity, req.sigma, req.rate,
                                 req.option_type, repeats=5)
        nn_rows.append({"label": "NN 1 price + Greeks",
                        "latency_ms": single_ms})

        # The rows report throughput. At one thread on a full core the per-row
        # rate is flat in batch size (one-off measurement: 59,800 prices/s at
        # 1,000 rows, 62,960 at 10,000, 62,660 at 50,000), so 10,000 rows
        # shows the batch regime and holds the simulation gate a fifth as long
        # as 50,000.
        for b in (1_000, 10_000):
            rng = np.random.default_rng(0)
            spots = rng.uniform(60, 180, b)
            strikes = np.full(b, req.strike)
            mats = rng.uniform(0.1, 2.0, b)
            sigs = rng.uniform(0.1, 0.6, b)
            rates = np.full(b, req.rate)
            ms, _ = time_call(eng.price_batch, spots, strikes, mats, sigs,
                              rates, option_type=req.option_type, repeats=2)
            nn_rows.append({"label": f"NN batch {b:,} prices",
                            "latency_ms": ms,
                            "throughput_per_s": b / max(ms / 1000, 1e-9)})

    mc_100k = float(mc_rows[-1]["latency_ms"])
    return {"mc": mc_rows, "nn": nn_rows,
            "headline_speedup": mc_100k / max(single_ms, 1e-6)}


@app.get("/api/market/{ticker}")
def market(ticker: str) -> dict:
    from ..quant.market_data import fetch_market_params, MarketDataError
    try:
        return fetch_market_params(ticker)
    except MarketDataError as exc:
        raise HTTPException(400, str(exc))


class HedgeRequest(ApiRequest):
    sigma: float = Field(0.25, gt=0, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    cost: float = Field(0.01, ge=0, le=0.05)
    # Which simulated market the hedgers are run on. "rough" is the
    # SPY-calibrated rough Bergomi model with compensated jumps (stochastic
    # rough volatility, spot-vol correlation, an incomplete market); "gbm" is
    # Black-Scholes, where a delta hedge is the benchmark to beat.
    dynamics: str = Field("rough", pattern="^(rough|gbm)$")


# compare() fixes its evaluation, tuning and bootstrap seeds, so the reply is
# a function of (dynamics, sigma, rate, cost), and under the rough dynamics of
# cost alone. One run of a few seconds serves every repeat of the question.
# A body is about 170 KB, so 32 entries hold under 6 MB.
_HEDGE_CACHE = _ResponseCache(max_entries=32)


@app.post("/api/hedge")
def hedge(req: HedgeRequest) -> Response:
    spec = HEDGE_DYNAMICS[req.dynamics]
    hedger = HEDGERS.get(req.dynamics)
    if hedger is None:
        raise HTTPException(
            503, f"No hedging policy is loaded for the {spec['label']} "
                 f"dynamics ({spec['checkpoint']}).")
    # The rough dynamics are the SPY calibration: forward vol sqrt(xi) and the
    # rate belong to the model, and the served policy was trained at those
    # values. The sliders drive Black-Scholes only.
    params = hedger.meta.get("measure_params") if req.dynamics == "rough" else None
    sigma = math.sqrt(float(params["xi"])) if params else req.sigma
    rate = float(params["rate"]) if params else req.rate

    key = (req.dynamics, sigma, rate, req.cost)
    body = _HEDGE_CACHE.get(key)
    if body is None:
        with heavy_job():
            # A request ahead in the queue may have computed this key.
            body = _HEDGE_CACHE.get(key)
            if body is None:
                out = hedger.compare(sigma, rate, req.cost,
                                     primary=spec["measure"],
                                     measures=(spec["measure"],))
                body = _json_body(_hedge_reply(out, req.dynamics, spec,
                                               hedger, bool(params)))
                _HEDGE_CACHE.put(key, body)
    return Response(body, media_type="application/json")


def _hedge_reply(out: dict, dynamics: str, spec: dict, hedger: HedgingEngine,
                 calibrated: bool) -> dict:
    # One measure is simulated, so by_measure repeats the top-level blocks.
    # It keeps every statistic and drops its copy of the per-path P&L arrays,
    # about 160 KB, as much as the rest of the body.
    out["by_measure"] = {
        m: {k: ({kk: vv for kk, vv in v.items() if kk != "pnl"}
                if isinstance(v, dict) else v)
            for k, v in block.items()}
        for m, block in out["by_measure"].items()}
    out["dynamics"] = dynamics
    out["dynamics_label"] = spec["label"]
    out["sigma_source"] = "SPY calibration" if calibrated else "slider"
    out["policy_trained_on"] = hedger.meta.get("train_measure", "unknown")
    out["measure_note"] = (
        f"Paths simulated under {spec['label']}; the deep policy was trained "
        f"under the same dynamics. Baselines are vol-matched to the realized "
        f"volatility of the simulated paths.")
    return out


class IVSurfaceRequest(ApiRequest):
    sigma: float = Field(0.25, ge=0.05, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    resolution: int = Field(41, ge=11, le=81)


def _iv_fit_summary(fit: dict) -> dict:
    """The two validation figures the panel shows, under the keys
    iv_surface.py writes: IV RMSE in vol points on the vega-resolved region
    and price RMSE in bps of strike."""
    val = fit.get("validation", {})
    return {
        "iv_rmse_volpts_resolved": val.get("iv_rmse_volpts_resolved"),
        "price_rmse_bps": val.get("price_rmse_bps"),
        "raw": val,
    }


@app.post("/api/iv-surface")
def iv_surface(req: IVSurfaceRequest) -> dict:
    """Arbitrage-free implied-vol surface over (log-moneyness, days) for the
    0DTE regime, with the Durrleman butterfly function and the calendar
    slope evaluated by autograd on the same grid."""
    if IV_SURFACE is None:
        raise HTTPException(503, "IV surface checkpoint not available.")
    rng = IV_SURFACE.meta.get("ranges", {})
    k_lo, k_hi = rng.get("k", (-0.139, 0.157))
    T_lo, T_hi = rng.get("T", (1 / 252.0, 12 / 252.0))
    n = req.resolution
    k_axis = np.linspace(k_lo, k_hi, n)
    T_axis = np.linspace(T_lo, T_hi, max(11, n // 2))
    with heavy_job():
        # Timed inside the gate, so the figure excludes the queue.
        t0 = time.perf_counter()
        g = IV_SURFACE.grid(req.sigma, req.rate, k_axis, T_axis)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    audit = (IV_SURFACE.meta.get("audit") or {})
    return {
        "k": g["k"].tolist(), "days": (g["T"] * 252.0).tolist(),
        "iv": g["iv"].tolist(), "g": g["g"].tolist(),
        "calendar": g["calendar"].tolist(),
        "g_min": float(g["g_min"]), "g_min_at": g["g_min_at"].tolist(),
        "calendar_min": float(g["calendar_min"]),
        "calendar_min_at": g["calendar_min_at"].tolist(),
        "n_points": int(g["iv"].size), "latency_ms": elapsed_ms,
        "fit": _iv_fit_summary(IV_SURFACE.meta.get("fit_metrics", {})),
        "audit": audit,
    }


@app.post("/api/explain")
def explain(req: OptionParams) -> dict:
    from ..quant.explain import integrated_gradients
    eng = engine()
    validate_domain(req)
    with light_job():
        return integrated_gradients(
            eng, req.spot, req.strike, req.maturity, req.sigma, req.rate,
            req.option_type
        )


class RiskReportRequest(ApiRequest):
    """Inputs of the desk note. The text fields reach the note and, with a
    provider key configured, the model prompt, so they are bounded to what
    the dashboard sends; every number is finite (ApiRequest)."""
    ticker: str = Field("", max_length=10, pattern=r"^[A-Za-z0-9.^-]*$")
    nn_price: float
    bs_cvar: float
    deep_cvar: float
    # Driver name -> dollar attribution, as /api/explain returns them.
    attributions: dict[str, float]
    # The hedging run's own description (contract, dynamics, cost level), so
    # the note describes the comparison that was simulated: every hedger pays
    # the same proportional cost.
    contract: str = Field("", max_length=120)
    ww_cvar: float | None = None
    deep_cost: float | None = None
    delta_cost: float | None = None
    dynamics_label: str = Field("", max_length=80)
    cost_bps: int | None = Field(None, ge=0, le=500)
    # Bootstrap standard errors from the Hedging tab. With them the note names
    # a winner only when the gap between the top two exceeds two combined
    # standard errors; without them it applies a relative-gap test.
    bs_cvar_se: float | None = None
    deep_cvar_se: float | None = None
    ww_cvar_se: float | None = None
    # The Integrated Gradients baseline's price, so the note quotes the figure
    # the attribution panel shows.
    baseline_price: float | None = None
    # Hedger pairs that the paired bootstrap on the shared paths separates,
    # keyed "a|b" with the short names ("deep", "delta", "band") in
    # alphabetical order. The hedgers run on the same paths, so hypot of two
    # individual errors overstates the error of their difference; when the
    # map is present the note uses it and both tabs apply one test to one run.
    paired_separated: dict[str, bool] | None = None

    @field_validator("attributions", "paired_separated")
    @classmethod
    def _few_short_keys(cls, v: dict | None) -> dict | None:
        if v is not None and (len(v) > 8 or any(len(k) > 32 for k in v)):
            raise ValueError("at most 8 entries with keys of at most 32 "
                             "characters")
        return v


@app.post("/api/risk-report")
def risk_report(req: RiskReportRequest):
    from ..quant.llm import get_risk_report_stream
    return get_risk_report_stream(
        req.ticker, req.nn_price, req.bs_cvar, req.deep_cvar, req.attributions,
        contract=req.contract, ww_cvar=req.ww_cvar, deep_cost=req.deep_cost,
        delta_cost=req.delta_cost, dynamics_label=req.dynamics_label,
        cost_bps=req.cost_bps, bs_cvar_se=req.bs_cvar_se,
        deep_cvar_se=req.deep_cvar_se, ww_cvar_se=req.ww_cvar_se,
        baseline_price=req.baseline_price, paired=req.paired_separated,
    )


# ---------------------------------------------------------------------------
# WebSocket: real-time tick stream with neural pricing
# ---------------------------------------------------------------------------

class StreamConfig(OptionParams):
    # Requested frame rate; the handler clamps it to [1, MAX_STREAM_HZ].
    hz: float = 10.0


STREAM_CONFIG_HELP = (
    "send one JSON object with spot, strike, maturity, sigma and rate as "
    "finite numbers in the ranges /api/price accepts, option_type 'call' or "
    "'put', and an optional hz")


# The dashboard's config message is under 200 bytes. A larger message is
# refused before it is parsed. This guard runs in the app; the frame size
# uvicorn buffers before the app sees it is set by its --ws-max-size flag.
MAX_STREAM_MESSAGE_BYTES = 1024

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def ws_origin_allowed(ws: WebSocket) -> bool:
    """Whether a browser on this Origin may open the stream.

    A handshake with no Origin header is not from a browser page and is
    allowed. Otherwise the origin must be in ALLOWED_ORIGINS, on the host the
    request was sent to (the dashboard's own page, on any port or scheme the
    server is reached by), or on a loopback host (a local frontend dev
    server)."""
    origin = ws.headers.get("origin")
    if origin is None:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    parsed = urlsplit(origin)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    host = ws.headers.get("host", "")
    if host and parsed.netloc.lower() == host.lower():
        return True
    return (parsed.hostname or "") in _LOOPBACK_HOSTS


async def _refuse(ws: WebSocket, message: str, code: int) -> None:
    await ws.send_json({"error": message})
    await ws.close(code=code)


#: Returned by _receive_config once the socket is refused or gone. A config
#: of JSON null parses to None, so None cannot mark this.
_NO_CONFIG = object()


async def _receive_config(ws: WebSocket) -> Any:
    """The client's one config message, parsed. Returns _NO_CONFIG after a
    refusal or a disconnect, with the socket already closed."""
    try:
        message = await asyncio.wait_for(ws.receive(), timeout=5.0)
    except asyncio.TimeoutError:
        await ws.close(code=1008, reason="no configuration received")
        return _NO_CONFIG
    if message["type"] == "websocket.disconnect":
        return _NO_CONFIG
    text = message.get("text")
    if text is None:
        # A binary frame where text is expected.
        await _refuse(ws, "config must be a JSON object", code=1008)
        return _NO_CONFIG
    if (len(text) > MAX_STREAM_MESSAGE_BYTES
            or len(text.encode("utf-8")) > MAX_STREAM_MESSAGE_BYTES):
        await _refuse(ws, f"config exceeds {MAX_STREAM_MESSAGE_BYTES} bytes",
                      code=1009)
        return _NO_CONFIG
    try:
        return json.loads(text)
    except ValueError:
        await _refuse(ws, "config must be a JSON object", code=1008)
        return _NO_CONFIG


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Stream simulated ticks with the surrogate's price and Greeks.

    The client sends one JSON config on connect:
        {"spot": 100, "strike": 100, "sigma": 0.25, "rate": 0.04,
         "maturity": 1.0, "option_type": "call", "hz": 20}
    and receives a JSON frame every 1/hz seconds: the GBM-simulated spot, the
    price and the five Greeks.

    The config passes the same schema and trained-domain gate as the REST
    endpoints, and a refusal carries a fixed message. Each frame is priced in
    a worker thread. The call is a 5-member ensemble forward plus a double
    backward for gamma (13.73 ms median, 74.71 ms p95); run on the event loop
    at 60 Hz it keeps the loop busy 82% of the time, and one client moves
    GET /api/health from 1.79 ms to 152.92 ms median (2,050 ms p95) on a
    uvicorn server. hz is capped at MAX_STREAM_HZ and concurrent streams at
    MAX_STREAM_CLIENTS.

    A handshake from a page on another origin is refused before it is
    accepted (CORS does not cover WebSockets), the config message is capped
    at MAX_STREAM_MESSAGE_BYTES, and the stream takes that one message: any
    later message closes it.
    """
    global _stream_clients
    if not ws_origin_allowed(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    eng = ENGINE
    if eng is None:
        await _refuse(ws, "model not loaded", code=1011)
        return

    raw = await _receive_config(ws)
    if raw is _NO_CONFIG:
        return
    if not isinstance(raw, dict):
        await _refuse(ws, "config must be a JSON object", code=1008)
        return

    try:
        config = StreamConfig.model_validate(raw)
        regime = validate_domain(config)
    except ValidationError as exc:
        # Field names only. The validator's own text names the library, its
        # version and a documentation URL.
        fields = sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
        await _refuse(ws, f"invalid config ({', '.join(fields)}): "
                          f"{STREAM_CONFIG_HELP}", code=1008)
        return
    except HTTPException as exc:
        await _refuse(ws, exc.detail, code=1008)
        return

    # Checked and taken with no await in between, so the count is exact on
    # the event loop.
    if _stream_clients >= MAX_STREAM_CLIENTS:
        await _refuse(ws, "stream capacity reached; try again shortly",
                      code=1013)
        return
    _stream_clients += 1
    # The client sends nothing after its config. A reader runs beside the
    # frame loop so a disconnect ends the stream at once, and a further
    # message, which would otherwise sit in the server's receive buffer,
    # closes it.
    frames = asyncio.ensure_future(_stream_frames(ws, eng, config, regime))
    reader = asyncio.ensure_future(ws.receive())
    try:
        done, _ = await asyncio.wait({frames, reader},
                                     return_when=asyncio.FIRST_COMPLETED)
        if reader in done and not frames.done():
            frames.cancel()
            try:
                await frames
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            message = reader.result()
            if message["type"] != "websocket.disconnect":
                await _refuse(ws, "the stream takes one config message",
                              code=1008)
        else:
            reader.cancel()
            frames.result()
    except WebSocketDisconnect:
        pass
    finally:
        for task in (frames, reader):
            if not task.done():
                task.cancel()
        _stream_clients -= 1


async def _stream_frames(ws: WebSocket, eng: PricingEngine,
                         config: StreamConfig, regime: str) -> None:
    strike, sigma, rate = config.strike, config.sigma, config.rate
    maturity, option_type = config.maturity, config.option_type
    hz = int(max(1, min(config.hz, MAX_STREAM_HZ)))
    dt = 1.0 / hz
    await ws.send_json({"status": "ready", "hz": hz, "regime": regime,
                        "max_hz": MAX_STREAM_HZ})

    # GBM tick simulator, dS = r*S*dt + sigma*S*dW, under the risk-neutral
    # drift. Only the spot moves, so the walk can leave the serving model's
    # box through moneyness alone.
    m_lo, m_hi = (ZERO_DTE_MONEYNESS if regime == "rough_bergomi_european"
                  else ASIAN_MONEYNESS)
    rng = np.random.default_rng()
    spot = config.spot
    tick = 0
    annual_dt = dt / (252 * 6.5 * 3600)  # seconds -> year fraction

    while True:
        t0 = time.perf_counter()

        dW = rng.standard_normal() * math.sqrt(annual_dt)
        spot *= math.exp((rate - 0.5 * sigma**2) * annual_dt + sigma * dW)
        tick += 1

        if not (m_lo <= spot / strike <= m_hi):
            # Outside the box the frame carries the spot and no price.
            await ws.send_json({"tick": tick, "spot": round(spot, 4),
                                "error": "out of domain"})
        else:
            try:
                result = await anyio.to_thread.run_sync(
                    eng.price_with_greeks,
                    spot, strike, maturity, sigma, rate, option_type)
            except Exception:
                logger.exception("stream pricing failed")
                await _refuse(ws, "pricing failed", code=1011)
                return
            frame = {
                "tick": tick,
                "spot": round(spot, 4),
                "price": round(result["price"], 4),
                "delta": round(result["greeks"]["delta"], 4),
                "gamma": round(result["greeks"]["gamma"], 6),
                "vega": round(result["greeks"]["vega"], 4),
                "theta": round(result["greeks"]["theta"], 4),
                "rho": round(result["greeks"]["rho"], 4),
                "clamped": result["clamped"],
                "latency_us": round((time.perf_counter() - t0) * 1e6, 0),
            }
            if regime == "rough_bergomi_european":
                # The REST path's floor, from the same function (1.12 us per
                # call against a 13.7 ms pricing call).
                bound = intrinsic_fields(
                    regime, result["raw_price"], spot, strike, maturity,
                    rate, option_type)
                frame["intrinsic"] = round(bound["intrinsic"], 4)
                frame["below_intrinsic"] = bound["below_intrinsic"]
            await ws.send_json(frame)

        # Pace to the target rate.
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(dt - elapsed, 0))


# The HTML, script and stylesheet names carry no content hash, so a browser
# revalidates them on every load (the ETag makes that a 304) and a deploy is
# seen at once. Images change rarely and are cached for an hour.
REVALIDATE = "no-cache"
IMAGE_CACHE = "public, max-age=3600"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"}


def cache_control_for(path: str | Path) -> str:
    return (IMAGE_CACHE if Path(path).suffix.lower() in _IMAGE_SUFFIXES
            else REVALIDATE)


class FrontendFiles(StaticFiles):
    """StaticFiles with a Cache-Control header on every file it serves."""

    def file_response(self, full_path, *args, **kwargs) -> Response:
        response = super().file_response(full_path, *args, **kwargs)
        response.headers["Cache-Control"] = cache_control_for(full_path)
        return response


class FrontendMount(Mount):
    """The dashboard mount at "/", which leaves /api paths to the router.

    A mount at "/" fully matches every path, so a method the API route does
    not take (GET /api/price, HEAD on a POST route) would reach StaticFiles
    and read as a missing file. Declining /api paths here lets the router
    answer 405 with an Allow header, or 404 for an unknown API path."""

    def matches(self, scope):
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == "/api"
                                        or path.startswith("/api/")):
            return Match.NONE, {}
        return super().matches(scope)


# The static dashboard is mounted last so the /api routes match first.
@app.api_route("/methodology", methods=["GET", "HEAD"],
               include_in_schema=False)
def methodology() -> FileResponse:
    """Clean URL for the static methodology page. StaticFiles(html=True) only
    maps directories to index.html, so /methodology would otherwise 404."""
    return FileResponse(FRONTEND / "methodology.html",
                        headers={"Cache-Control": REVALIDATE})


app.router.routes.append(
    FrontendMount("/", app=FrontendFiles(directory=str(FRONTEND), html=True),
                  name="frontend"))
