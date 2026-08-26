"""FastAPI layer serving the neural pricer, the Monte Carlo engine, and the
static dashboard.

Run from the repo root:
    python -m uvicorn backend.api.main:app --port 8000
"""

from __future__ import annotations

import anyio
import asyncio
import json
import math
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..quant.engine import PricingEngine, time_call
from ..quant.monte_carlo import MCResult

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

# Below this maturity the engine routes to the 0DTE rough-Bergomi surrogate,
# so the Monte Carlo benchmark must switch measure too. The rough-vol
# parameters must match backend/quant/dataset_0dte.py (the teacher).
#
# IMPORTANT: this cutoff separates two different CONTRACTS, not just two models.
# Above it the service prices an arithmetic-average ASIAN option under GBM;
# at or below it, a EUROPEAN option under rough Bergomi. Measured at S=K=100,
# sigma=0.25, r=0.04, the price jumps +45.6% (1.35400 -> 1.97183) across 8.1e-5
# years of maturity, about 42 minutes. Every response therefore carries an
# explicit `regime` field rather than leaving the switch invisible.
ZERO_DTE_CUTOFF = 12.0 / 252.0
ZERO_DTE_MIN_MATURITY = 1.0 / 252.0      # 0DTE surrogate's trained floor
ZERO_DTE_MONEYNESS = (0.85, 1.15)
ASIAN_MIN_MATURITY = 0.05                # main surrogate's trained floor
ASIAN_MONEYNESS = (0.5, 2.0)
ROUGH_ETA, ROUGH_RHO, ROUGH_H = 1.5, -0.7, 0.1

# Measured cost of one price+Greeks frame is ~13.7 ms median / 74.7 ms p95, so
# 60 Hz cannot be honoured (the socket actually delivered 28.8 Hz while starving
# the loop). 15 Hz leaves the event loop real headroom.
MAX_STREAM_HZ = 15

# One Monte Carlo / batch-inference job at a time. These handlers are sync
# `def`s, so Starlette dispatches each to a ~40-thread pool and nothing else
# serializes them: the dashboard's page load used to put /api/price,
# /api/convergence and /api/benchmark in flight simultaneously, and their
# combined path arrays OOM-killed the 512 MB free-tier container (exit 137).
# Queued requests just block their pool thread, which costs a few KB each;
# the timeout turns a pathological pile-up into a 503 instead of a hang.
_HEAVY_JOB_GATE = threading.Semaphore(1)
HEAVY_JOB_TIMEOUT_S = 120.0


@contextmanager
def heavy_job() -> Iterator[None]:
    if not _HEAVY_JOB_GATE.acquire(timeout=HEAVY_JOB_TIMEOUT_S):
        raise HTTPException(
            503, "Simulation queue is saturated; retry shortly",
            headers={"Retry-After": "10"})
    try:
        yield
    finally:
        _HEAVY_JOB_GATE.release()

app = FastAPI(title="Deep Learning for Options Pricing",
              description="Neural surrogate vs Monte Carlo for arithmetic "
                          "Asian options (maturity > 12/252) and European "
                          "options under rough Bergomi (maturity <= 12/252)")
# The dashboard is served by this same app, so browsers never need
# cross-origin access at all. The old wildcard existed for local file://
# testing and, once the app went public on a free-tier host, turned into an
# invitation: any third-party page could fan its visitors' browsers out
# against the Monte Carlo endpoints and burn the instance's CPU from anywhere.
# Local dev on a separate frontend port stays possible via the explicit
# localhost entries. Non-browser clients (curl, notebooks) are unaffected;
# CORS only governs browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000",
                   "https://neural-options-lab.onrender.com"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

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


try:
    from ..quant.hedging import HedgingEngine
    HEDGER: HedgingEngine | None = HedgingEngine()
except FileNotFoundError:
    HEDGER = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def regime_for(maturity: float) -> str:
    """Which surrogate (and therefore which CONTRACT) serves this maturity."""
    if maturity <= ZERO_DTE_CUTOFF and ENGINE is not None and ENGINE.has_0dte:
        return "rough_bergomi_european"
    return "asian_gbm"


def validate_domain(req: "OptionParams") -> str:
    """Reject anything outside a trained box, and name the serving regime.

    Previously only moneyness was checked, and only against the Asian box, so
    three failure modes reached the model as silent extrapolation:

      * maturity below the 0DTE floor. At S=K=100, sigma=0.25, r=0.04 the
        surrogate plateaus at ~0.399 as T -> 0 instead of decaying to the
        intrinsic value: T=1e-9 returned 0.39898 against a Monte Carlo
        reference of 0.00031, roughly 1300x the true value, with HTTP 200.
      * the dead band (12/252, 0.05). Above the 0DTE cutoff so it routes to the
        Asian net, but below that net's 0.05 training floor. Measured error
        there is 2.95-3.00 bps of strike versus 0.83 bps at T=1.0.
      * sigma below the trained floor of 0.05, which yields arbitrage-violating
        prices.

    Returns the regime string so handlers can label their response.
    """
    m = req.spot / req.strike
    regime = regime_for(req.maturity)

    if regime == "rough_bergomi_european":
        lo, hi = ZERO_DTE_MONEYNESS
        if not (lo <= m <= hi):
            raise HTTPException(
                422, f"0DTE engine covers moneyness S/K in [{lo}, {hi}]; "
                     f"got {m:.4f}")
        if req.maturity < ZERO_DTE_MIN_MATURITY:
            raise HTTPException(
                422, f"maturity {req.maturity:.6g} is below the 0DTE "
                     f"surrogate's trained floor of {ZERO_DTE_MIN_MATURITY:.6g} "
                     f"(1 trading day); the model extrapolates to a price far "
                     f"above intrinsic value there")
    else:
        lo, hi = ASIAN_MONEYNESS
        if not (lo <= m <= hi):
            raise HTTPException(
                422, f"moneyness S/K outside trained domain [{lo}, {hi}]; "
                     f"got {m:.4f}")
        if req.maturity < ASIAN_MIN_MATURITY:
            raise HTTPException(
                422, f"maturity {req.maturity:.6g} falls in the uncovered band "
                     f"({ZERO_DTE_CUTOFF:.6g}, {ASIAN_MIN_MATURITY:.6g}): above "
                     f"the 0DTE cutoff but below the Asian surrogate's trained "
                     f"floor. No surrogate is valid here.")

    if not (0.05 <= req.sigma <= 0.80):
        raise HTTPException(
            422, f"sigma {req.sigma:.4g} outside trained domain [0.05, 0.80]")
    if not (0.0 <= req.rate <= 0.10):
        raise HTTPException(
            422, f"rate {req.rate:.4g} outside trained domain [0.0, 0.10]")
    return regime


# Kept as an alias so existing call sites keep working.
def validate_moneyness(req: "OptionParams") -> None:
    validate_domain(req)


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

        # Benchmark under the dynamics the served 0DTE surrogate was
        # actually trained on (live-calibrated after a calibrate.py
        # --retrain cycle); constants are the pre-calibration fallback.
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


class OptionParams(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0, le=2.0)
    sigma: float = Field(0.25, gt=0, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    option_type: str = Field("call", pattern="^(call|put)$")


class PriceRequest(OptionParams):
    # Capped at 100k: the MC engines now run in fixed-size blocks so memory no
    # longer scales with the request, but wall-clock still does, and on the
    # free tier's 0.1 CPU a 500k-path run would hold the heavy-job gate (and
    # its pool thread) for tens of seconds.
    mc_paths: int = Field(50_000, ge=1_000, le=100_000)


class ConvergenceRequest(OptionParams):
    # Previously unbounded in value, sign AND length: a caller could ask for
    # [10**12] * 10**6 and the handler would try to honour it. Each entry costs
    # a full Monte Carlo run on the request thread.
    path_counts: list[int] = Field(
        default=[500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        min_length=1, max_length=12)

    @field_validator("path_counts")
    @classmethod
    def _bounded_paths(cls, v: list[int]) -> list[int]:
        # Same rationale as PriceRequest.mc_paths: memory is bounded by the
        # chunked MC engines, so these caps bound wall-clock on 0.1 CPU.
        for n in v:
            if not (100 <= n <= 100_000):
                raise ValueError(
                    f"path_counts entries must be in [100, 100000]; got {n}")
        if sum(v) > 600_000:
            raise ValueError(
                f"total simulated paths {sum(v):,} exceeds the 600,000 "
                f"budget for a single convergence request")
        return v


class SurfaceRequest(BaseModel):
    sigma: float = Field(0.25, gt=0, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    strike: float = Field(100.0, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    resolution: int = Field(45, ge=10, le=80)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": ENGINE is not None}


@app.get("/api/model-info")
def model_info() -> dict:
    meta = dict(engine().meta)
    eval_file = Path(__file__).resolve().parents[2] / "artifacts" / "eval.json"
    if eval_file.exists():
        report = json.loads(eval_file.read_text())
        meta["eval"] = {k: report[k] for k in
                        ("n_points", "ref_paths", "single", "ensemble")}
    return meta


@app.get("/api/error-distribution")
def error_distribution() -> dict:
    """Signed pricing errors (bps of strike) of the single model vs the
    ensemble, measured against high-precision MC references by
    backend.quant.evaluate."""
    eval_file = Path(__file__).resolve().parents[2] / "artifacts" / "eval.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=503,
            detail="No evaluation report. Run: "
                   "python -m backend.quant.evaluate"
        )
    return json.loads(eval_file.read_text())


@app.post("/api/price")
def price(req: PriceRequest) -> dict:
    eng = engine()
    validate_moneyness(req)

    nn_ms, nn_out = time_call(
        eng.price_with_greeks, req.spot, req.strike, req.maturity,
        req.sigma, req.rate, req.option_type)

    with heavy_job():
        t0 = time.perf_counter()
        mc, mc_engine = mc_reference(req, req.mc_paths)
        mc_ms = (time.perf_counter() - t0) * 1000.0

    diff = nn_out["price"] - mc.price
    return {
        "nn": {"price": nn_out["price"], "greeks": nn_out["greeks"],
               "latency_ms": nn_ms},
        "mc": {"price": mc.price, "std_error": mc.std_error,
               "ci_low": mc.ci_low, "ci_high": mc.ci_high,
               "n_paths": mc.n_paths, "n_steps": mc.n_steps,
               "engine": mc_engine, "latency_ms": mc_ms},
        "comparison": {
            "abs_diff": abs(diff),
            "diff_bps_of_spot": abs(diff) / req.spot * 1e4,
            "within_mc_ci": mc.ci_low <= nn_out["price"] <= mc.ci_high,
            "speedup": mc_ms / max(nn_ms, 1e-6),
        },
    }


@app.post("/api/convergence")
def convergence(req: ConvergenceRequest) -> dict:
    """MC estimate vs path count (with 95% CI) against the instant NN price."""
    eng = engine()
    validate_moneyness(req)
    points = []
    mc_engine = "asian_gbm_cv"
    with heavy_job():
        for n in sorted(set(req.path_counts)):
            t0 = time.perf_counter()
            mc, mc_engine = mc_reference(req, n, seed=42)
            points.append({"n_paths": n, "price": mc.price,
                           "ci_low": mc.ci_low, "ci_high": mc.ci_high,
                           "latency_ms": (time.perf_counter() - t0) * 1000.0})

        nn_ms, nn_out = time_call(
            eng.price_with_greeks, req.spot, req.strike, req.maturity,
            req.sigma, req.rate, req.option_type)

        # 100k, down from 400k: the CI-width gain of 400k (2x) is invisible on
        # the dashboard, while the extra 300k paths quadrupled the wall-clock
        # this endpoint holds the heavy-job gate on the free tier's 0.1 CPU.
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

        # 50k, down from 100k: a 100k-point batch through the 5-member
        # ensemble transiently holds ~150 MB of float32 activations, which is
        # most of the free tier's request headroom on its own. Throughput per
        # second is what the row reports, and that is batch-size-invariant
        # once the batch saturates the CPU.
        for b in (1_000, 50_000):
            rng = np.random.default_rng(0)
            spots = rng.uniform(60, 180, b)
            strikes = np.full(b, req.strike)
            mats = rng.uniform(0.1, 2.0, b)
            sigs = rng.uniform(0.1, 0.6, b)
            rates = np.full(b, req.rate)
            ms, _ = time_call(eng.price_batch, spots, strikes, mats, sigs,
                              rates, option_type=req.option_type, repeats=3)
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


class HedgeRequest(BaseModel):
    sigma: float = Field(0.25, gt=0, le=0.8)
    rate: float = Field(0.04, ge=0, le=0.1)
    cost: float = Field(0.01, ge=0, le=0.05)


@app.post("/api/hedge")
def hedge(req: HedgeRequest) -> dict:
    if HEDGER is None:
        raise HTTPException(503, "Hedging model not trained.")
    return HEDGER.compare(req.sigma, req.rate, req.cost)


@app.post("/api/explain")
def explain(req: OptionParams) -> dict:
    from ..quant.explain import integrated_gradients
    eng = engine()
    return integrated_gradients(
        eng, req.spot, req.strike, req.maturity, req.sigma, req.rate,
        req.option_type
    )


class RiskReportRequest(BaseModel):
    ticker: str
    nn_price: float
    bs_cvar: float
    deep_cvar: float
    attributions: dict


@app.post("/api/risk-report")
def risk_report(req: RiskReportRequest):
    from ..quant.llm import get_risk_report_stream
    return get_risk_report_stream(
        req.ticker, req.nn_price, req.bs_cvar, req.deep_cvar, req.attributions
    )


# ---------------------------------------------------------------------------
# WebSocket: real-time tick stream with neural pricing
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """Stream simulated market ticks with real-time neural net pricing.

    The client sends a JSON config on connect:
        {"spot": 100, "strike": 100, "sigma": 0.25, "rate": 0.04,
         "maturity": 1.0, "option_type": "call", "hz": 20}

    The server then pushes a JSON frame every 1/hz seconds containing the
    GBM-simulated spot, the neural net price, and all five Greeks.

    Two things this handler previously got wrong:

      * It called price_with_greeks synchronously on the event loop. That call
        is a 5-member ensemble forward plus a double backward for gamma -
        measured at 13.73 ms median, 74.71 ms p95, NOT the "sub-millisecond"
        the old comment claimed. At the formerly-permitted 60 Hz the loop was
        busy 82% of the time, and against a real uvicorn server one client
        moved GET /api/health from 1.79 ms median to 152.92 ms median with
        2050 ms p95 - an 85x inflation. Since render.yaml health-checks that
        endpoint, a single browser tab could mark the container unhealthy.
        The pricing call now runs in a worker thread.
      * It applied no validation at all, so sigma=-1 streamed a call delta of
        -4.49. The config is now validated against the same trained domain as
        the REST endpoints.

    hz is capped at MAX_STREAM_HZ, chosen so the measured per-frame cost leaves
    the loop real headroom rather than saturating it.
    """
    await ws.accept()
    eng = ENGINE
    if eng is None:
        await ws.send_json({"error": "model not loaded"})
        await ws.close()
        return

    try:
        config = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        return
    except Exception:
        await ws.send_json({"error": "config must be a JSON object"})
        await ws.close()
        return

    try:
        params = OptionParams(
            spot=float(config.get("spot", 100)),
            strike=float(config.get("strike", 100)),
            maturity=float(config.get("maturity", 1.0)),
            sigma=float(config.get("sigma", 0.25)),
            rate=float(config.get("rate", 0.04)),
            option_type=str(config.get("option_type", "call")))
        regime = validate_domain(params)
    except HTTPException as exc:
        await ws.send_json({"error": exc.detail})
        await ws.close()
        return
    except Exception as exc:
        await ws.send_json({"error": f"invalid config: {exc}"})
        await ws.close()
        return

    spot0, strike = params.spot, params.strike
    sigma, rate = params.sigma, params.rate
    maturity, option_type = params.maturity, params.option_type
    try:
        hz = int(config.get("hz", 10))
    except (TypeError, ValueError):
        hz = 10
    hz = max(1, min(hz, MAX_STREAM_HZ))
    dt = 1.0 / hz
    await ws.send_json({"status": "ready", "hz": hz, "regime": regime,
                        "max_hz": MAX_STREAM_HZ})

    # GBM tick simulator: dS = mu*S*dt + sigma*S*dW
    # We use the risk-neutral drift so the walk is financially coherent.
    rng = np.random.default_rng()
    spot = spot0
    tick = 0
    annual_dt = dt / (252 * 6.5 * 3600)  # seconds -> year fraction

    try:
        while True:
            t0 = time.perf_counter()

            # Simulate one GBM tick
            dW = rng.standard_normal() * math.sqrt(annual_dt)
            spot *= math.exp((rate - 0.5 * sigma**2) * annual_dt + sigma * dW)
            tick += 1

            # Price + Greeks off the event loop: this is ~13.7 ms of CPU
            # (ensemble forward + double backward for gamma), which would
            # otherwise block every other request on the loop thread.
            try:
                result = await anyio.to_thread.run_sync(
                    eng.price_with_greeks,
                    spot, strike, maturity, sigma, rate, option_type)
                frame = {
                    "tick": tick,
                    "spot": round(spot, 4),
                    "price": round(result["price"], 4),
                    "delta": round(result["greeks"]["delta"], 4),
                    "gamma": round(result["greeks"]["gamma"], 6),
                    "vega": round(result["greeks"]["vega"], 4),
                    "theta": round(result["greeks"]["theta"], 4),
                    "rho": round(result["greeks"]["rho"], 4),
                    "latency_us": round(
                        (time.perf_counter() - t0) * 1e6, 0),
                }
            except Exception:
                # Out of domain - send spot only
                frame = {"tick": tick, "spot": round(spot, 4),
                         "error": "out of domain"}

            await ws.send_json(frame)

            # Pace to target Hz
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(dt - elapsed, 0))

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


# Static dashboard - mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True),
          name="frontend")
