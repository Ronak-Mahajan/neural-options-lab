"""FastAPI layer serving the neural pricer, the Monte Carlo engine, and the
static dashboard.

Run from the repo root:
    python -m uvicorn backend.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..quant.engine import PricingEngine, time_call
from ..quant.monte_carlo import MCResult

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

# Below this maturity the engine routes to the 0DTE rough-Bergomi surrogate,
# so the Monte Carlo benchmark must switch measure too. The rough-vol
# parameters must match backend/quant/dataset_0dte.py (the teacher).
ZERO_DTE_CUTOFF = 12.0 / 252.0
ZERO_DTE_MONEYNESS = (0.85, 1.15)
ROUGH_ETA, ROUGH_RHO, ROUGH_H = 1.5, -0.7, 0.1

app = FastAPI(title="Deep Learning for Options Pricing",
              description="Neural surrogate vs Monte Carlo for arithmetic "
                          "Asian options")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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

def validate_moneyness(req: "OptionParams") -> None:
    """Domain check depends on which surrogate serves the maturity band."""
    m = req.spot / req.strike
    if req.maturity <= ZERO_DTE_CUTOFF and engine().has_0dte:
        lo, hi = ZERO_DTE_MONEYNESS
        if not (lo <= m <= hi):
            raise HTTPException(
                422, f"0DTE engine covers moneyness S/K in [{lo}, {hi}]")
    elif not (0.5 <= m <= 2.0):
        raise HTTPException(422, "moneyness S/K outside trained domain "
                                 "[0.5, 2.0]")


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
    mc_paths: int = Field(50_000, ge=1_000, le=500_000)


class ConvergenceRequest(OptionParams):
    path_counts: list[int] = Field(
        default=[500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000])


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
    for n in sorted(set(req.path_counts)):
        t0 = time.perf_counter()
        mc, mc_engine = mc_reference(req, n, seed=42)
        points.append({"n_paths": n, "price": mc.price,
                       "ci_low": mc.ci_low, "ci_high": mc.ci_high,
                       "latency_ms": (time.perf_counter() - t0) * 1000.0})

    nn_ms, nn_out = time_call(
        eng.price_with_greeks, req.spot, req.strike, req.maturity,
        req.sigma, req.rate, req.option_type)

    ref, _ = mc_reference(req, 400_000, seed=7)
    return {"mc_points": points, "engine": mc_engine,
            "nn": {"price": nn_out["price"], "latency_ms": nn_ms},
            "reference": {"price": ref.price, "std_error": ref.std_error,
                          "n_paths": ref.n_paths}}


@app.post("/api/surface")
def surface(req: SurfaceRequest) -> dict:
    """NN price surface over (moneyness x maturity) — thousands of prices
    in one batched forward pass to showcase surrogate throughput."""
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
    nn_rows.append({"label": "NN 1 price + Greeks", "latency_ms": single_ms})

    for b in (1_000, 100_000):
        rng = np.random.default_rng(0)
        spots = rng.uniform(60, 180, b)
        strikes = np.full(b, req.strike)
        mats = rng.uniform(0.1, 2.0, b)
        sigs = rng.uniform(0.1, 0.6, b)
        rates = np.full(b, req.rate)
        ms, _ = time_call(eng.price_batch, spots, strikes, mats, sigs, rates,
                          option_type=req.option_type, repeats=3)
        nn_rows.append({"label": f"NN batch {b:,} prices", "latency_ms": ms,
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
    GBM-simulated spot, the neural net price, and all five Greeks. This
    proves the surrogate can price at interactive frame rates — something
    Monte Carlo cannot do.
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

    spot0 = float(config.get("spot", 100))
    strike = float(config.get("strike", 100))
    sigma = float(config.get("sigma", 0.25))
    rate = float(config.get("rate", 0.04))
    maturity = float(config.get("maturity", 1.0))
    option_type = config.get("option_type", "call")
    hz = min(int(config.get("hz", 20)), 60)
    dt = 1.0 / max(hz, 1)

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

            # Neural net forward pass (sub-millisecond)
            try:
                result = eng.price_with_greeks(
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
                # Out of domain — send spot only
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


# Static dashboard — mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True),
          name="frontend")
