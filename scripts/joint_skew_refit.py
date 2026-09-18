"""Joint (H, eta) refit of rough Bergomi on SPY: smile fit + ATM skew term structure.

The question
------------
Every least-squares rough Bergomi fit of the SPY short-dated smile pins the
vol-of-vol eta at its 4.0 bound (artifacts/rough_calibration.json), and
docs/atm_skew_term_structure.md showed that at that eta the model's ATM skew
psi(T) = d sigma_imp/dk |_{k=0} steepens with exponent -0.321 +- 0.007 over
1-45 trading days while the SPY market's is -0.249 +- 0.033, close to the
textbook H - 1/2. Does a joint fit of (H, eta) - rho and xi free - exist that
matches the smiles acceptably AND the market's skew term structure, with eta
interior to its bounds, and what does it cost in smile RMSE?

What this script does (each stage checkpoints to JSON and resumes)
------------------------------------------------------------------
1. `skew-check`  Map-based ATM skew. psi(tau) is a central difference in
   log-moneyness of the pricing map's implied vols (MapPricer, the regionally
   validated surrogate of rough_bergomi_mc) on a five-point stencil k in
   {-2h, -h, 0, +h, +2h} centred on the forward (the market's and the MC
   stencil's ATM), h = max(0.25 atm_iv sqrt(tau), 0.002) like
   scripts/atm_skew_term_structure.py, at the capture's own expiries. It is
   validated against that script's Monte Carlo `model_skew_curve` at the
   served parameters; the agreement in psi and in exponent is what licenses
   using the map for the profile.
2. `profile`     For each cell of a 12 x 12 grid over H in [0.05, 0.45] and
   eta in [0.5, 4.0], (rho, xi) are optimised on MapCalibrator's smile loss
   (Powell, warm-started across the grid), for 3 trading-hour captures. Each
   cell records the smile RMSE, the model psi at the capture's expiries, the
   fitted exponent over those expiries, and the chi-square distance to the
   market psi(tau) using the market standard errors.
3. `joint`       J(theta) = smile Huber loss (MapCalibrator.loss, unchanged)
   + lambda * mean_e ((psi_model - psi_market) / SE_market)^2, minimised
   over (eta, rho, H, xi) by differential evolution + Powell exactly as
   MapCalibrator.fit does (plus Powell polishes from the previous lambda's
   optimum and from the profile's best cell; the best polish wins), for
   lambda on a log grid. lambda = 0 IS MapCalibrator.fit.
4. `eta-extend` The same profile strip and four-parameter fits with eta free
   up to the COMMITTED map's own training ceiling (pricing_map.pt's box,
   eta to 8.0) beside the same fits capped at the calibrator's 4.0, so that
   "is the smile optimum at eta 4, or is 4 where the optimiser stops?" is
   answered rather than assumed. The winning extended point is then checked
   against rough_bergomi_mc, which the map has no licence above eta 4.
5. `bootstrap`   Quote-resampling (stratified by expiry) uncertainty of the
   joint optimum at the recommended lambda.
6. `licence`     The map-vs-MC psi comparison repeated at profile cells with
   small eta (|psi| small), where the map's ABSOLUTE psi error becomes a
   large RELATIVE one: it bounds the region of the profile the map can be
   trusted in.
7. `mc`          Monte Carlo validation with the true engine: the skew ladder
   at the joint, served, pure-smile and smallest-interior-eta parameters,
   and the smile RMSE of each on one capture by pricing every quote with
   rough_bergomi_mc.
8. `report`      Assemble docs/joint_skew_refit.{json,png} and
   artifacts/rough_calibration_skewjoint.json (analysis, not served).

Polishing: scipy's Powell WITH bounds (what MapCalibrator.fit uses) minimises
each line search with fminbound over the whole feasible segment, which never
evaluates the current point; on a multimodal line it can return a WORSE
point, after which Powell's convergence test stops it. Measured at
lambda = 10 on every capture (e.g. the polish from the lambda = 3 optimum
moved xi from 0.017 to 0.35 and returned J = 117 from a start at J = 32). All
polishes here therefore also run an unbounded Powell in logit coordinates
(Brent line searches bracket outward from the current point) and keep the
best of {start, bounded, logit}; `polish` records which one won.

    python -m scripts.joint_skew_refit all --work-dir <scratch>      # everything, serial
    python -m scripts.joint_skew_refit profile --capture-index 1 ...  # one stage

The markdown (docs/joint_skew_refit.md) is written by hand from the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.calibrate import (BOUNDS, HUBER_DELTA, PIN_FRAC, Quote,  # noqa: E402
                                     implied_vol)
from backend.quant.calibrate_map import (MapCalibrator, MapPricer,  # noqa: E402
                                         quotes_from_capture)
from backend.quant.rough_vol import rough_bergomi_mc  # noqa: E402
from scripts import atm_skew_term_structure as ats  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DATA = ROOT / "data"
CAPTURE_GLOB = str(DATA / "surfaces" / "equity" / "spy_*.json.gz")
TRADING_DAYS = ats.TRADING_DAYS
LADDER = tuple(d for d in ats.DEFAULT_T_DAYS)          # 1 ... 126 trading days
SHORT_MAX_DAYS = ats.SHORT_FIT_MAX_DAYS                 # exponent window <= 45 d
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
H_GRID = np.linspace(0.05, 0.45, 12)
ETA_GRID = np.linspace(0.5, 4.0, 12)
JOINT_BOUNDS = [BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]
DE_KW = dict(maxiter=60, popsize=16, tol=1e-7, mutation=(0.4, 0.9),
             recombination=0.8, polish=False, init="sobol", updating="deferred")
POWELL_KW = {"xtol": 1e-6, "ftol": 1e-8, "maxfev": 4000}
#: relative "error" attached to a map-based psi so fit_power_law does an
#: unweighted fit in log space (the map has no path noise; this is a device)
MAP_PSI_REL_SE = 0.01
DEFAULT_WORK = Path(tempfile.gettempdir()) / "joint_skew_refit_work"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(p: Path, default):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def save_json(p: Path, obj) -> None:
    """Atomic-ish checkpoint write. Windows can refuse the rename for a few
    hundred ms while an indexer/AV scanner holds the fresh file, so retry
    before falling back to a direct write."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    text = json.dumps(obj, indent=1)
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(8):
        try:
            tmp.replace(p)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    p.write_text(text, encoding="utf-8")


def eta_interior(eta: float, lo: float = BOUNDS["eta"][0],
                 hi: float = BOUNDS["eta"][1], frac: float = PIN_FRAC) -> bool:
    """Interior by calibrate.quality_gate's rule: more than PIN_FRAC of the
    bound's width away from either end."""
    w = frac * (hi - lo)
    return bool(lo + w < eta < hi - w)


# ── (1) map-based ATM skew ─────────────────────────────────────────────────
class SkewStencil:
    """Five-point strike stencils, one per expiry, priced on the map.

    The map's native log-moneyness is ln(K / fwd_pv) with fwd_pv = F e^{-r tau}
    (calibrate.Quote), whereas the market extractor (market_skew_from_quotes)
    and the Monte Carlo stencil (price_stencil, strikes = F e^{k}) both take
    k against the FORWARD F. The stencil is therefore centred at K = F, i.e.
    at map log-moneyness k0 = r tau, with strikes e^{k0 + m h} on fwd_pv = 1:
    "ATM" means the same strike in all three. Centring on fwd_pv instead
    (k0 = 0) shifts psi by about 2 c r tau (c the smile curvature), measured
    at 0.02-0.08 on the SPY captures, i.e. one to several market standard
    errors - not negligible for the chi2 term.
    """

    def __init__(self, taus, h, k0=None):
        self.taus = np.asarray(taus, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.k0 = (np.zeros_like(self.taus) if k0 is None
                   else np.asarray(k0, dtype=float))
        if self.taus.shape != self.h.shape or self.taus.shape != self.k0.shape:
            raise ValueError("taus, h and k0 must align")
        self.quotes: list[Quote] = []
        for tau, hh, kk in zip(self.taus, self.h, self.k0):
            for m in ats.STENCIL:
                self.quotes.append(Quote(
                    tau=float(tau), strike=math.exp(kk + m * hh), mid_call=0.0,
                    iv=0.0, vega=0.0, kind="C", expiry=f"tau={tau:.6f}",
                    fwd_pv=1.0))
        self.n = len(self.taus)

    @classmethod
    def for_market_rows(cls, rows: list[dict], rate: float, h_scale: float = 0.25,
                        h_floor: float = 0.002) -> "SkewStencil":
        """h from the MARKET atm vol so the discretisation is fixed while the
        parameters move (the objective must not carry a moving stencil);
        centred on the forward (k0 = rate * tau)."""
        taus = [r["tau"] for r in rows]
        h = [max(h_scale * r["atm_iv"] * math.sqrt(r["tau"]), h_floor)
             for r in rows]
        return cls(taus, h, [rate * t for t in taus])

    @classmethod
    def for_model(cls, taus, xi: float, rate: float, h_scale: float = 0.25,
                  h_floor: float = 0.002) -> "SkewStencil":
        """h exactly as model_skew_curve chooses it (from sqrt(xi)), centred
        on the forward exactly as price_stencil is."""
        return cls(taus, [ats.moneyness_step(xi, float(t), h_scale, h_floor)
                          for t in taus], [rate * float(t) for t in taus])

    def ivs(self, pricer: MapPricer, theta) -> np.ndarray:
        eta, rho, H, xi = map(float, theta[:4])
        return pricer.ivs(self.quotes, eta, rho, H, xi).reshape(self.n, 5)

    def skew(self, pricer: MapPricer, theta) -> dict:
        iv = self.ivs(pricer, theta)
        h = self.h
        psi = (iv[:, 3] - iv[:, 1]) / (2.0 * h)
        psi_2h = (iv[:, 4] - iv[:, 0]) / (4.0 * h)
        return {"psi": psi, "psi_2h": psi_2h,
                "psi_richardson": (4.0 * psi - psi_2h) / 3.0,
                "truncation": np.abs(psi - psi_2h) / 3.0,
                "atm_iv": iv[:, 2], "h": h}

    def psi(self, pricer: MapPricer, theta) -> np.ndarray:
        iv = self.ivs(pricer, theta)
        return (iv[:, 3] - iv[:, 1]) / (2.0 * self.h)


def model_exponent(taus, psi) -> dict:
    """Power-law exponent of a map-based psi over the capture's expiries:
    unweighted in log space (constant relative 'SE'); the returned se_b is the
    least-squares scatter SE, not a sampling error."""
    psi = np.asarray(psi, dtype=float)
    return ats.fit_power_law(taus, psi, MAP_PSI_REL_SE * np.abs(psi))


def skew_chi2(psi_model, psi_market, se_market) -> float:
    """mean_e ((psi_model - psi_market) / SE_market)^2"""
    z = (np.asarray(psi_model) - np.asarray(psi_market)) / np.asarray(se_market)
    return float(np.mean(z ** 2))


# ── batched map evaluation ─────────────────────────────────────────────────
class MapBatch:
    """MapCalibrator.loss and the stencil psi for MANY parameter vectors in
    one forward pass of the map.

    The input layout is exactly MapPricer.ivs's (eta, rho, H, xi, lam = 0,
    mu_j = 0, sig_j = 0.01, tau, k), so for a single theta the smile loss is
    bit-identical to MapCalibrator.loss (asserted in the tests); the point of
    the batch is that differential evolution can evaluate a whole population
    (64 vectors x ~620 rows) in one call, which measured ~2x cheaper per
    vector than one call per member.
    """

    def __init__(self, cal: MapCalibrator, stencil: SkewStencil | None = None):
        self.pricer = cal.pricer
        self.n_smile = len(cal.quotes)
        self.stencil = stencil
        qs = list(cal.quotes) + (list(stencil.quotes) if stencil is not None else [])
        self.n = len(qs)
        self.tau = np.array([q.tau for q in qs], dtype=float)
        self.k = np.array([math.log(q.strike / q.fwd_pv) for q in qs], dtype=float)
        self.mids = np.array(cal.mids_iv, dtype=float)
        self.n_evals = 0

    def ivs(self, thetas: np.ndarray) -> np.ndarray:
        """(S, 4) parameter vectors -> (S, n) implied vols."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        S, n = thetas.shape[0], self.n
        self.n_evals += S
        X = np.empty((S * n, 9))
        X[:, :4] = np.repeat(thetas[:, :4], n, axis=0)
        X[:, 4] = 0.0
        X[:, 5] = 0.0
        X[:, 6] = 0.01
        X[:, 7] = np.tile(self.tau, S)
        X[:, 8] = np.tile(self.k, S)
        Xt = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            out = self.pricer.model(self.pricer._features(Xt)).numpy().astype(np.float64)
        return out.reshape(S, n)

    def smile_loss_from_ivs(self, ivs: np.ndarray) -> np.ndarray:
        e = ivs[:, :self.n_smile] - self.mids
        d = HUBER_DELTA
        hub = np.where(np.abs(e) <= d, 0.5 * e ** 2, d * (np.abs(e) - 0.5 * d))
        return hub.mean(axis=1) * 1e4

    def psi_from_ivs(self, ivs: np.ndarray) -> np.ndarray:
        st = ivs[:, self.n_smile:].reshape(ivs.shape[0], self.stencil.n, 5)
        return (st[:, :, 3] - st[:, :, 1]) / (2.0 * self.stencil.h)

    def evaluate(self, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        ivs = self.ivs(thetas)
        psi = self.psi_from_ivs(ivs) if self.stencil is not None else None
        return self.smile_loss_from_ivs(ivs), psi

    def smile_loss(self, theta) -> float:
        return float(self.smile_loss_from_ivs(self.ivs(np.asarray(theta)[None, :4]))[0])

    def rmse_volpts(self, theta) -> float:
        ivs = self.ivs(np.asarray(theta)[None, :4])[0, :self.n_smile]
        return float(np.sqrt(np.mean((ivs - self.mids) ** 2))) * 100.0


# ── captures ───────────────────────────────────────────────────────────────
class CaptureSet:
    """One capture's quotes, market skew rows and stencil, ready for fitting.

    Build from a capture file, or from an explicit quote list (tests, the
    bootstrap) with `CaptureSet.from_quotes`.
    """

    def __init__(self, path: Path | None, pricer: MapPricer | None = None,
                 quotes: list[Quote] | None = None, rate: float | None = None,
                 meta: dict | None = None, expiry_order: list[str] | None = None):
        self.path = Path(path) if path is not None else Path("quotes")
        if quotes is None:
            self.quotes_all, self.rate, self.meta = quotes_from_capture(self.path)
        else:
            self.quotes_all, self.rate, self.meta = list(quotes), float(rate), dict(meta or {})
        self.pricer = pricer or MapPricer()
        self.cal = MapCalibrator(self.quotes_all, market="SPY", pricer=self.pricer)
        rows = ats.market_skew_from_quotes(self.cal.quotes, self.rate)
        if expiry_order is not None:
            by = {r["expiry"]: r for r in rows}
            rows = [by[e] for e in expiry_order if e in by]
        self.market_rows = rows
        if not self.market_rows:
            raise ValueError("no expiry with enough ATM quotes for a market skew")
        self.taus = np.array([r["tau"] for r in self.market_rows])
        self.psi_mkt = np.array([r["psi"] for r in self.market_rows])
        self.se_mkt = np.array([r["se"] for r in self.market_rows])
        self.stencil = SkewStencil.for_market_rows(self.market_rows, self.rate)
        self.batch = MapBatch(self.cal, self.stencil)
        self.market_fit = ats.fit_power_law(self.taus, self.psi_mkt, self.se_mkt)

    @classmethod
    def from_quotes(cls, quotes: list[Quote], rate: float, meta: dict | None = None,
                    pricer: MapPricer | None = None, expiry_order: list[str] | None = None,
                    name: str = "quotes") -> "CaptureSet":
        cs = cls(None, pricer, quotes=quotes, rate=rate, meta=meta, expiry_order=expiry_order)
        cs.path = Path(name)
        return cs

    def info(self) -> dict:
        return {"capture": self.path.name,
                "pricing_time": self.meta.get("pricing_time") or self.meta.get("as_of"),
                "spot": self.meta.get("spot"), "rate": self.rate,
                "n_quotes": len(self.quotes_all), "n_fitted": len(self.cal.quotes),
                "n_out_of_box": self.cal.n_out_of_box,
                "expiries": [r["expiry"] for r in self.market_rows],
                "taus": self.taus.tolist(),
                "T_days_equiv": (self.taus * TRADING_DAYS).tolist(),
                "psi_market": self.psi_mkt.tolist(), "se_market": self.se_mkt.tolist(),
                "atm_iv_market": [r["atm_iv"] for r in self.market_rows],
                "stencil_h": self.stencil.h.tolist(),
                "stencil_k0": self.stencil.k0.tolist(),
                "market_b": self.market_fit.get("b"),
                "market_se_b": self.market_fit.get("se_b")}

    def evaluate(self, theta) -> dict:
        """Everything the profile/joint stages record about a parameter vector."""
        theta = np.asarray(theta, dtype=float)
        sk = self.stencil.skew(self.pricer, theta)
        f = model_exponent(self.taus, sk["psi"])
        fr = model_exponent(self.taus, sk["psi_richardson"])
        return {"theta": theta.tolist(),
                "smile_loss": self.batch.smile_loss(theta),
                "rmse_volpts": self.batch.rmse_volpts(theta),
                "psi": sk["psi"].tolist(), "psi_2h": sk["psi_2h"].tolist(),
                "psi_richardson": sk["psi_richardson"].tolist(),
                "atm_iv": sk["atm_iv"].tolist(),
                "b_expiries": f.get("b"), "b_expiries_scatter_se": f.get("se_b"),
                "b_expiries_richardson": fr.get("b"),
                "chi2": skew_chi2(sk["psi"], self.psi_mkt, self.se_mkt),
                "chi2_richardson": skew_chi2(sk["psi_richardson"], self.psi_mkt,
                                             self.se_mkt),
                "eta_interior": eta_interior(float(theta[0]))}


def select_captures(n: int = 3, explicit: list[str] | None = None) -> list[Path]:
    if explicit:
        return [Path(c) for c in explicit]
    return ats.trading_hour_captures(CAPTURE_GLOB, n)


def served_params() -> dict:
    return ats.load_params(ARTIFACTS / "rough_calibration.json")


def served_theta() -> np.ndarray:
    p = served_params()
    return np.array([p["eta"], p["rho"], p["H"], p["xi"]])


# ── (2) profile over (H, eta) ──────────────────────────────────────────────
def fit_rho_xi(cs: "CaptureSet", H: float, eta: float, x0=(-0.6, math.log(0.013)),
               maxfev: int = 800) -> dict:
    """Powell over (rho, ln xi) at fixed (H, eta) on MapCalibrator's loss."""
    lo_xi, hi_xi = BOUNDS["xi"]
    bounds = [BOUNDS["rho"], (math.log(lo_xi), math.log(hi_xi))]
    x0 = np.array([min(max(x0[0], bounds[0][0] + 1e-6), bounds[0][1] - 1e-6),
                   min(max(x0[1], bounds[1][0] + 1e-6), bounds[1][1] - 1e-6)])
    n0 = cs.batch.n_evals

    def f(z):
        return cs.batch.smile_loss(np.array([eta, z[0], H, math.exp(z[1])]))

    r = minimize(f, x0, method="Powell", bounds=bounds,
                 options={"xtol": 1e-4, "ftol": 1e-7, "maxfev": maxfev})
    return {"rho": float(r.x[0]), "xi": float(math.exp(r.x[1])),
            "loss": float(r.fun), "nfev": cs.batch.n_evals - n0}


def profile_capture(cs: CaptureSet, ckpt: Path, log, H_grid=H_GRID,
                    eta_grid=ETA_GRID) -> dict:
    """Snake-ordered sweep with warm starts; every cell saved to `ckpt`."""
    state = load_json(ckpt, {"capture": cs.path.name, "H_grid": list(map(float, H_grid)),
                             "eta_grid": list(map(float, eta_grid)), "cells": {}})
    cells = state["cells"]
    served = served_theta()
    x_prev = (float(served[1]), math.log(float(served[3])))
    t_all = time.perf_counter()
    for i, H in enumerate(H_grid):
        js = range(len(eta_grid)) if i % 2 == 0 else range(len(eta_grid) - 1, -1, -1)
        for j in js:
            key = f"{i}_{j}"
            if key in cells:
                c = cells[key]
                x_prev = (c["rho"], math.log(c["xi"]))
                continue
            eta = float(eta_grid[j])
            t0 = time.perf_counter()
            starts = [x_prev]
            if i == 0 and j in (0, len(eta_grid) - 1):
                starts.append((float(served[1]), math.log(float(served[3]))))
            best = None
            for x0 in starts:
                r = fit_rho_xi(cs, float(H), eta, x0)
                if best is None or r["loss"] < best["loss"]:
                    best = r
            ev = cs.evaluate([eta, best["rho"], float(H), best["xi"]])
            cells[key] = {"i": i, "j": j, "H": float(H), "eta": eta,
                          "rho": best["rho"], "xi": best["xi"],
                          "nfev": best["nfev"], "seconds": time.perf_counter() - t0,
                          **{k: v for k, v in ev.items() if k != "theta"}}
            x_prev = (best["rho"], math.log(best["xi"]))
            save_json(ckpt, state)
            log(f"  [{cs.path.name[4:19]}] cell {key} H={H:.3f} eta={eta:.3f} "
                f"rho={best['rho']:+.3f} sqrt(xi)={math.sqrt(best['xi']):.4f} "
                f"rmse={ev['rmse_volpts']:.3f} b={ev['b_expiries']:+.3f} "
                f"chi2={ev['chi2']:.1f} ({best['nfev']} ev, "
                f"{cells[key]['seconds']:.1f}s, {len(cells)}/{len(H_grid)*len(eta_grid)})")
    state["seconds_total"] = time.perf_counter() - t_all
    state["done"] = len(cells) == len(H_grid) * len(eta_grid)
    save_json(ckpt, state)
    return state


def profile_best_cell(state: dict, lam: float) -> dict:
    """Grid cell minimising smile_loss + lam * chi2."""
    return min(state["cells"].values(),
               key=lambda c: c["smile_loss"] + lam * c["chi2"])


# ── (2b) eta above the calibrator's ceiling ────────────────────────────────
def map_eta_box(pricer: MapPricer | None = None) -> tuple[float, float]:
    """The eta interval the committed map was TRAINED on, read from the
    checkpoint's own metadata.

    The 4.0 ceiling every SPY fit sits on is an optimiser bound
    (calibrate.BOUNDS, exposed as calibrate_map.ETA_MAX['SPY']), not a
    property of the surrogate: artifacts/pricing_map.pt was banked over
    gen_pricing_map.BOX with eta in (0.5, 8.0), and the BTC path already
    drives the same map to 8. So "is the smile optimum AT eta 4, or is 4
    where the optimiser stops?" is answerable from committed artifacts, with
    nothing regenerated - which is what this stage does.
    """
    lo, hi = (pricer or MapPricer()).box["eta"]
    return float(lo), float(hi)


def extended_eta_grid(eta_hi: float, eta_lo: float = BOUNDS["eta"][1],
                      step: float = 0.5) -> np.ndarray:
    """eta from the calibrator's ceiling up to the map's box top, inclusive.
    Starting AT the ceiling keeps the strip self-contained: the eta = 4
    column is the anchor every extended cell is compared against."""
    n = int(math.floor((float(eta_hi) - float(eta_lo)) / step + 1e-9)) + 1
    return float(eta_lo) + step * np.arange(max(n, 1), dtype=float)


def extended_bounds(eta_hi: float) -> list:
    """JOINT_BOUNDS with eta opened to the map's box."""
    return [(BOUNDS["eta"][0], float(eta_hi))] + list(JOINT_BOUNDS[1:])


# ── (3) joint fit ──────────────────────────────────────────────────────────
class JointObjective:
    """smile Huber loss (MapCalibrator.loss, same formula on the same map)
    + lam * mean_e ((psi_model - psi_market) / SE_market)^2."""

    def __init__(self, cs: "CaptureSet", lam: float):
        self.cs, self.lam = cs, float(lam)
        self.n_evals = 0

    def batch(self, thetas: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(thetas)
        self.n_evals += thetas.shape[0]
        smile, psi = self.cs.batch.evaluate(thetas)
        if self.lam == 0.0:
            return smile
        z = (psi - self.cs.psi_mkt) / self.cs.se_mkt
        return smile + self.lam * np.mean(z ** 2, axis=1)

    def __call__(self, theta) -> float:
        return float(self.batch(np.asarray(theta, dtype=float)[None, :])[0])

    def vectorized(self, X: np.ndarray) -> np.ndarray:
        """scipy's vectorized convention: X has shape (N, S)."""
        return self.batch(np.asarray(X, dtype=float).T)


def _bounds_arrays(bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo, hi


def to_logit(x, bounds=JOINT_BOUNDS, eps: float = 1e-6) -> np.ndarray:
    lo, hi = _bounds_arrays(bounds)
    u = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), eps, 1.0 - eps)
    return np.log(u / (1.0 - u))


def from_logit(z, bounds=JOINT_BOUNDS) -> np.ndarray:
    lo, hi = _bounds_arrays(bounds)
    return lo + (hi - lo) / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def polish(obj, x0, bounds=JOINT_BOUNDS, powell_kw: dict | None = None) -> dict:
    """Powell polish that cannot end worse than where it started.

    Runs scipy's bounded Powell (MapCalibrator.fit's polisher) AND an
    unbounded Powell in logit-transformed coordinates from the same start,
    and returns the best of {start, bounded, logit}. See the module
    docstring for why the bounded one alone is not safe.
    """
    powell_kw = dict(POWELL_KW, **(powell_kw or {}))
    lo, hi = _bounds_arrays(bounds)
    x0 = np.clip(np.asarray(x0, dtype=float), lo + 1e-9, hi - 1e-9)
    f0 = float(obj(x0))
    cands = [("start", x0, f0, 0)]
    r = minimize(obj, x0, method="Powell", bounds=bounds, options=powell_kw)
    cands.append(("bounded", np.clip(np.asarray(r.x, dtype=float), lo, hi), float(r.fun), int(r.nfev)))
    rz = minimize(lambda z: obj(from_logit(z, bounds)), to_logit(x0, bounds),
                  method="Powell", options=powell_kw)
    cands.append(("logit", from_logit(rz.x, bounds), float(rz.fun), int(rz.nfev)))
    name, x, fun, _ = min(cands, key=lambda c: c[2])
    return {"x": np.asarray(x, dtype=float), "fun": fun, "method": name,
            "nfev": int(sum(c[3] for c in cands)), "start_fun": f0,
            "bounded_fun": cands[1][2], "logit_fun": cands[2][2],
            "bounded_worse_than_start": bool(cands[1][2] > f0 * (1 + 1e-12))}


def joint_fit(obj: JointObjective, seed: int = 7, extra_starts=(),
              de_kw: dict | None = None, powell_kw: dict | None = None,
              polish_extra: int = 1, bounds=JOINT_BOUNDS) -> dict:
    """DE + Powell with MapCalibrator.fit's settings and seed (the population
    is evaluated in one batched call per generation, which does not change
    the deferred-updating trajectory), plus Powell polishes from the best
    `polish_extra` of `extra_starts` (ranked by objective); best polish wins.

    `bounds` defaults to the calibrator's box; the eta-extend stage passes the
    same box with eta opened to the map's own training range."""
    de_kw = dict(DE_KW, **(de_kw or {}))
    powell_kw = dict(POWELL_KW, **(powell_kw or {}))
    t0 = time.perf_counter()
    n0 = obj.n_evals
    de = differential_evolution(obj.vectorized, bounds, seed=seed,
                                vectorized=True, **de_kw)
    nfev_de = obj.n_evals - n0
    starts = [("de", np.asarray(de.x, dtype=float))]
    extra = [(name, np.asarray(x, dtype=float)) for name, x in extra_starts]
    if extra and polish_extra > 0:
        vals = obj.batch(np.array([x for _, x in extra]))
        order = np.argsort(vals)[:polish_extra]
        starts += [extra[i] for i in order]
    best, winner, polished = None, None, []
    for name, x0 in starts:
        r = polish(obj, x0, bounds, powell_kw)
        polished.append({"start": name, "x0": np.asarray(x0, dtype=float).tolist(),
                         "fun": r["fun"], "x": r["x"].tolist(), "nfev": r["nfev"],
                         "method": r["method"], "start_fun": r["start_fun"],
                         "bounded_fun": r["bounded_fun"], "logit_fun": r["logit_fun"],
                         "bounded_worse_than_start": r["bounded_worse_than_start"]})
        if best is None or r["fun"] < best["fun"]:
            best, winner = r, name
    return {"theta": np.asarray(best["x"], dtype=float), "J": float(best["fun"]),
            "de_x": np.asarray(de.x, dtype=float).tolist(), "de_fun": float(de.fun),
            "nfev_de": nfev_de, "nfev_total": obj.n_evals - n0,
            "winner": winner, "polished": polished,
            "seconds": time.perf_counter() - t0}


def joint_capture(cs: CaptureSet, ckpt: Path, profile_state: dict | None, log,
                  lambdas=LAMBDAS, seed: int = 7) -> dict:
    state = load_json(ckpt, {"capture": cs.path.name, "seed": seed, "fits": {}})
    fits = state["fits"]
    prev = None
    for lam in lambdas:
        key = f"{lam:g}"
        if key in fits:
            prev = np.array(fits[key]["theta"])
            continue
        extra = []
        if prev is not None:
            extra.append(("prev_lambda", prev))
        if profile_state is not None and profile_state.get("cells"):
            c = profile_best_cell(profile_state, lam)
            extra.append(("profile_best", np.array([c["eta"], c["rho"], c["H"], c["xi"]])))
        extra.append(("served", served_theta()))
        obj = JointObjective(cs, lam)
        r = joint_fit(obj, seed=seed, extra_starts=extra)
        ev = cs.evaluate(r["theta"])
        fits[key] = {"lambda": lam, **ev, "J": r["J"], "de_x": r["de_x"],
                     "de_fun": r["de_fun"], "nfev_de": r["nfev_de"],
                     "nfev_total": r["nfev_total"], "winner": r["winner"],
                     "polished": r["polished"], "seconds": r["seconds"]}
        prev = r["theta"]
        save_json(ckpt, state)
        th = r["theta"]
        log(f"  [{cs.path.name[4:19]}] lambda={lam:g}: eta={th[0]:.4f} rho={th[1]:.4f} "
            f"H={th[2]:.4f} sqrt(xi)={math.sqrt(th[3]):.4f}  rmse={ev['rmse_volpts']:.3f} "
            f"b={ev['b_expiries']:+.3f} chi2={ev['chi2']:.2f} J={r['J']:.3f} "
            f"(winner {r['winner']}, {r['nfev_total']} ev, {r['seconds']:.0f}s)")
    state["done"] = all(f"{lam:g}" in fits for lam in lambdas)
    save_json(ckpt, state)
    return state


# ── Pareto front ───────────────────────────────────────────────────────────
def _kneedle(x, y) -> int | None:
    """Index of the point of maximum perpendicular distance from the chord
    joining the first and last points, each axis scaled to [0, 1]; None when
    either axis is degenerate."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.max() <= x.min() or y.max() <= y.min():
        return None
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    d = np.array([u[-1] - u[0], v[-1] - v[0]])
    d /= np.linalg.norm(d)
    dist = np.abs((u - u[0]) * d[1] - (v - v[0]) * d[0])
    return int(np.argmax(dist))


def pareto_table(rows: list[dict], b_market: float, se_b_market: float,
                 tol: float = 1e-6) -> dict:
    """Sort joint fits by lambda, check the monotonicity exact optima must
    obey (smile loss non-decreasing, skew chi2 non-increasing in lambda), and
    locate the knee.

    Knee: the point of maximum perpendicular distance from the chord joining
    the lambda = 0 and largest-lambda points in the (smile RMSE, |b - b_mkt|)
    plane with each axis scaled to [0, 1] (the usual "kneedle" rule).
    Also reported: the smallest lambda whose exponent is within one market SE
    of the market exponent.
    """
    rows = sorted(rows, key=lambda r: r["lambda"])
    rmse = np.array([r["rmse_volpts"] for r in rows])
    loss = np.array([r["smile_loss"] for r in rows])
    chi2 = np.array([r["chi2"] for r in rows])
    berr = np.array([abs(r["b_expiries"] - b_market) for r in rows])
    smile_mono = bool(np.all(np.diff(loss) >= -tol * np.maximum(1.0, np.abs(loss[:-1]))))
    chi2_mono = bool(np.all(np.diff(chi2) <= tol * np.maximum(1.0, np.abs(chi2[:-1]))))
    knee = _kneedle(rmse, berr) if len(rows) >= 3 else None
    # the same rule in the (smile RMSE, log chi2) plane: the chi2 front spans
    # decades, so its knee is located on log chi2
    knee_chi2 = _kneedle(rmse, np.log(np.maximum(chi2, 1e-12))) if len(rows) >= 3 else None
    within = [i for i, e in enumerate(berr) if e <= se_b_market]
    # recommendation rule: the knee; if the front is too short for a knee, the
    # first lambda within one market SE; failing that, the largest lambda
    rec = knee if knee is not None else (within[0] if within else len(rows) - 1)
    return {"recommended_index": rec, "recommended_lambda": rows[rec]["lambda"],
            "rows": [{"lambda": r["lambda"], "rmse_volpts": r["rmse_volpts"],
                      "smile_loss": r["smile_loss"], "chi2": r["chi2"],
                      "b_expiries": r["b_expiries"], "b_error": float(e),
                      "theta": r["theta"], "eta_interior": r.get("eta_interior")}
                     for r, e in zip(rows, berr)],
            "smile_monotone": smile_mono, "chi2_monotone": chi2_mono,
            "knee_index": knee,
            "knee_lambda": rows[knee]["lambda"] if knee is not None else None,
            "knee_chi2_index": knee_chi2,
            "knee_chi2_lambda": rows[knee_chi2]["lambda"] if knee_chi2 is not None else None,
            "first_within_1se_index": within[0] if within else None,
            "first_within_1se_lambda": rows[within[0]]["lambda"] if within else None,
            "b_market": b_market, "se_b_market": se_b_market}


# ── (4) bootstrap ──────────────────────────────────────────────────────────
def resample_quotes(quotes: list[Quote], rng: np.random.Generator) -> list[Quote]:
    by: dict[str, list[Quote]] = {}
    for q in quotes:
        by.setdefault(q.expiry, []).append(q)
    out = []
    for qs in by.values():
        idx = rng.integers(0, len(qs), size=len(qs))
        out.extend(qs[i] for i in idx)
    return out


def bootstrap_capture(cs: CaptureSet, lam: float, theta_hat, ckpt: Path, log,
                      reps: int = 8, rep_offset: int = 0, seed: int = 20260910) -> dict:
    state = load_json(ckpt, {"capture": cs.path.name, "lambda": lam,
                             "theta_hat": list(map(float, theta_hat)), "reps": {}})
    for r in range(rep_offset, rep_offset + reps):
        key = str(r)
        if key in state["reps"]:
            continue
        rng = np.random.default_rng(seed + r)
        t0 = time.perf_counter()
        sub = resample_quotes(cs.cal.quotes, rng)
        boot = CaptureSet.from_quotes(sub, cs.rate, cs.meta, cs.pricer,
                                      expiry_order=[x["expiry"] for x in cs.market_rows],
                                      name=cs.path.name)
        rows = boot.market_rows
        obj = JointObjective(boot, lam)
        res = polish(obj, theta_hat, JOINT_BOUNDS)
        th = np.asarray(res["x"], dtype=float)
        ev = boot.evaluate(th)
        state["reps"][key] = {"theta": th.tolist(), "J": float(res["fun"]),
                              "polish_method": res["method"],
                              "bounded_worse_than_start": res["bounded_worse_than_start"],
                              "rmse_volpts": ev["rmse_volpts"], "chi2": ev["chi2"],
                              "b_expiries": ev["b_expiries"], "n_expiries": len(rows),
                              "nfev": obj.n_evals, "seconds": time.perf_counter() - t0}
        save_json(ckpt, state)
        log(f"  boot {r}: eta={th[0]:.4f} rho={th[1]:.4f} H={th[2]:.4f} "
            f"sqrt(xi)={math.sqrt(th[3]):.4f} rmse={ev['rmse_volpts']:.3f} "
            f"b={ev['b_expiries']:+.3f} ({obj.n_evals} ev, {state['reps'][key]['seconds']:.0f}s)")
    return state


def summarise_bootstrap(states: list[dict]) -> dict:
    reps = [r for s in states for r in s["reps"].values()]
    th = np.array([r["theta"] for r in reps])
    if len(reps) == 0:
        return {"n": 0}
    names = ["eta", "rho", "H", "xi"]
    out = {"n": len(reps), "theta_hat": states[0]["theta_hat"], "lambda": states[0]["lambda"]}
    for i, n in enumerate(names):
        col = th[:, i]
        out[n] = {"mean": float(col.mean()), "sd": float(col.std(ddof=1)) if len(col) > 1 else None,
                  "p2.5": float(np.percentile(col, 2.5)), "p97.5": float(np.percentile(col, 97.5))}
    for k in ("rmse_volpts", "chi2", "b_expiries"):
        col = np.array([r[k] for r in reps])
        out[k] = {"mean": float(col.mean()), "sd": float(col.std(ddof=1)) if len(col) > 1 else None}
    out["frac_eta_interior"] = float(np.mean([eta_interior(t[0]) for t in th]))
    return out


# ── diagnostics ────────────────────────────────────────────────────────────
def residual_bands(cs: "CaptureSet", theta, edges=(0.0, 0.5, 1.0, 2.0, float("inf"))) -> dict:
    """Map IV residuals (model - quoted, vol points) of the fitted quotes by
    normalised moneyness z = |k| / (atm_iv sqrt(tau)) (k against the forward,
    atm_iv the market's per expiry) and by expiry: where a skew-constrained
    fit pays its smile RMSE, and with which sign."""
    theta = np.asarray(theta, dtype=float)
    ivs = cs.pricer.ivs(cs.cal.quotes, *map(float, theta[:4]))
    e = (ivs - cs.cal.mids_iv) * 100.0
    atm = {r["expiry"]: r["atm_iv"] for r in cs.market_rows}
    z = np.array([abs(math.log(q.strike / q.fwd_pv) - cs.rate * q.tau)
                  / (atm.get(q.expiry, q.iv) * math.sqrt(q.tau)) for q in cs.cal.quotes])
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (z >= lo) & (z < hi)
        if sel.any():
            bands.append({"z_lo": lo, "z_hi": (hi if np.isfinite(hi) else None),
                          "n": int(sel.sum()),
                          "rmse_volpts": float(np.sqrt(np.mean(e[sel] ** 2))),
                          "mean_volpts": float(np.mean(e[sel]))})
    per_exp = {}
    for ex in sorted(set(q.expiry for q in cs.cal.quotes)):
        sel = np.array([q.expiry == ex for q in cs.cal.quotes])
        per_exp[ex] = {"n": int(sel.sum()), "rmse_volpts": float(np.sqrt(np.mean(e[sel] ** 2))),
                       "mean_volpts": float(np.mean(e[sel]))}
    return {"theta": theta.tolist(), "bands": bands, "per_expiry": per_exp,
            "rmse_volpts": float(np.sqrt(np.mean(e ** 2)))}


def contour_bests(prof: dict, b_mkt: float, se_mkt: float,
                  eta_caps=(4.01, 3.5, 2.5, 1.5)) -> dict:
    """Cheapest cell (smile loss) among those whose exponent is within one
    market SE of the market's, with eta capped: the profile's own answer to
    'what does matching the exponent cost at a given vol-of-vol'."""
    cells = list(prof["cells"].values())
    keys = ("H", "eta", "rho", "xi", "rmse_volpts", "b_expiries", "chi2", "smile_loss")
    best = min(cells, key=lambda c: c["smile_loss"])
    on = [c for c in cells if abs(c["b_expiries"] - b_mkt) <= se_mkt]
    out = {"best_smile_cell": {k: best[k] for k in keys}, "n_cells_within_1se": len(on),
           "by_eta_cap": {}}
    for cap in eta_caps:
        sub = [c for c in on if c["eta"] <= cap]
        if sub:
            bb = min(sub, key=lambda c: c["smile_loss"])
            out["by_eta_cap"][f"{cap:g}"] = {**{k: bb[k] for k in keys},
                                             "rmse_cost_volpts": bb["rmse_volpts"] - best["rmse_volpts"]}
    return out


def interior_candidate(fits: dict, lambdas=LAMBDAS) -> dict | None:
    """Smallest lambda > 0 whose optimum has eta interior (the gate's rule)."""
    for lam in lambdas:
        f = fits.get(f"{lam:g}")
        if f and lam > 0 and f.get("eta_interior"):
            return {"lambda": lam, **{k: f[k] for k in ("theta", "rmse_volpts", "b_expiries",
                                                         "chi2", "smile_loss", "eta_interior")}}
    return None


# ── (5) Monte Carlo validation ─────────────────────────────────────────────
def mc_skew_on_taus(theta, taus, spot: float, rate: float, n_paths: int,
                    n_reps: int, seed: int, log) -> list[dict]:
    params = {"eta": float(theta[0]), "rho": float(theta[1]), "H": float(theta[2]),
              "xi": float(theta[3]), "spot": float(spot), "rate": float(rate)}
    T_days = [float(t) * TRADING_DAYS for t in taus]      # T = d / 252 == tau
    return ats.model_skew_curve(params, T_days, n_paths, n_reps, seed, log=log)


def mc_smile_rmse(cs: CaptureSet, theta, n_paths: int, seed: int, log,
                  seeds=(0, 1), edges=(0.0, 0.5, 1.0, 2.0, float("inf"))) -> dict:
    """Smile RMSE of `theta` under the true engine, scored exactly as
    calibrate.iv_fit_report scores a calibration: every fitted quote is
    priced with rough_bergomi_mc (grouped per expiry on one path set,
    starting at fwd_pv as calibrate.Calibrator does) and inverted with
    implied_vol; a quote whose MC price lands inside the no-arbitrage band
    is NOT dropped but scored by the vega-linearised price error. Two
    independent seeds give the MC noise floor per quote, and everything is
    broken down by normalised moneyness z = |k| / (atm_iv sqrt(tau)),
    because in the far wing (z >= 2, half the quotes) plain MC on 400k
    paths cannot resolve a deep-ITM call's time value and the inversion is
    noise-dominated, while inside z < 2 the map-vs-MC agreement is the
    regionally validated 0.03-0.10 vp. The map's RMSE on the same quotes is reported
    alongside (its own error at these parameters)."""
    from backend.quant.calibrate import iv_fit_report
    eta, rho, H, xi = map(float, theta[:4])
    groups: dict[str, list[Quote]] = {}
    for q in cs.cal.quotes:
        groups.setdefault(q.expiry, []).append(q)
    quotes = [q for qs in groups.values() for q in qs]
    mids = np.array([q.iv for q in quotes])
    atm = {r["expiry"]: r["atm_iv"] for r in cs.market_rows}
    z = np.array([abs(math.log(q.strike / q.fwd_pv) - cs.rate * q.tau)
                  / (atm.get(q.expiry, q.iv) * math.sqrt(q.tau)) for q in quotes])
    e_map = (cs.pricer.ivs(quotes, eta, rho, H, xi) - mids) * 100.0
    t0 = time.perf_counter()
    reports, errs = [], []
    for s_ in seeds:
        prices = []
        for expiry, qs in groups.items():
            b = len(qs)
            prices.append(rough_bergomi_mc(
                torch.full((b,), qs[0].fwd_pv), torch.tensor([q.strike for q in qs], dtype=torch.float32),
                torch.full((b,), qs[0].tau), torch.full((b,), xi), torch.full((b,), eta),
                torch.full((b,), rho), torch.full((b,), cs.rate), n_paths=n_paths,
                n_steps=ats.N_STEPS, H=H, seed=seed + s_).double().numpy())
        r = iv_fit_report(quotes, np.concatenate(prices), cs.rate)
        reports.append(r)
        errs.append(np.array(r["errors_volpts"], dtype=float))
        log(f"    seed {seed + s_}: RMSE {r['rmse_volpts']:.3f} vp (priceable only "
            f"{r['rmse_volpts_priceable_only']:.3f}, {r['n_unpriceable']} scored by vega-linearised "
            f"error) ({time.perf_counter() - t0:.0f}s)")
    e1, e2 = errs[0], errs[1] if len(errs) > 1 else errs[0]
    noise = np.sqrt(np.mean((e1 - e2) ** 2) / 2.0)

    def block(sel):
        return {"n": int(sel.sum()),
                "rmse_mc_volpts": float(np.mean([np.sqrt(np.mean(e[sel] ** 2)) for e in errs])),
                "rmse_mc_by_seed": [float(np.sqrt(np.mean(e[sel] ** 2))) for e in errs],
                "rmse_map_volpts": float(np.sqrt(np.mean(e_map[sel] ** 2))),
                "map_minus_mc_rms_volpts": float(np.sqrt(np.mean((e_map[sel] - e1[sel]) ** 2))),
                "map_minus_mc_mean_volpts": float(np.mean(e_map[sel] - e1[sel])),
                "mc_noise_volpts": float(np.sqrt(np.mean((e1[sel] - e2[sel]) ** 2) / 2.0)),
                "n_unpriceable_seed0": int(np.isin(np.where(sel)[0], reports[0]["unpriceable_idx"]).sum())}

    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (z >= lo) & (z < hi)
        if sel.any():
            bands.append({"z_lo": lo, "z_hi": (hi if np.isfinite(hi) else None), **block(sel)})
    per_exp = {}
    for e in groups:
        sel = np.array([q.expiry == e for q in quotes])
        per_exp[e] = block(sel)
    core = block(z < 2.0)
    allq = block(np.ones(len(quotes), dtype=bool))
    return {"theta": list(map(float, theta[:4])), "n_paths": n_paths, "seeds": [seed + s_ for s_ in seeds],
            "n_quotes": len(quotes), "scoring": "calibrate.iv_fit_report (vega-linearised error "
            "for quotes whose MC price is inside the no-arbitrage band; nothing dropped)",
            "rmse_mc_volpts": allq["rmse_mc_volpts"], "rmse_mc_by_seed": allq["rmse_mc_by_seed"],
            "rmse_mc_priceable_only_by_seed": [r["rmse_volpts_priceable_only"] for r in reports],
            "n_unpriceable_by_seed": [r["n_unpriceable"] for r in reports],
            "n_unpriceable": reports[0]["n_unpriceable"], "n_priced": len(quotes),
            "rmse_map_volpts": allq["rmse_map_volpts"],
            "rmse_map_all_quotes_volpts": cs.cal.rmse_volpts(np.asarray(theta, dtype=float)),
            "mc_noise_floor_volpts": float(noise),
            "map_minus_mc_rms_volpts": allq["map_minus_mc_rms_volpts"],
            "map_minus_mc_mean_volpts": allq["map_minus_mc_mean_volpts"],
            "map_minus_mc_max_volpts": float(np.max(np.abs(e_map - e1))),
            "z_lt_2": core, "bands": bands, "per_expiry": per_exp,
            "seconds": time.perf_counter() - t0}


def compare_map_vs_mc(cs: CaptureSet, theta, mc_rows: list[dict]) -> dict:
    """Map psi against MC psi on the same taus, with the MC's own h."""
    taus = [r["T"] for r in mc_rows]
    st = SkewStencil.for_model(taus, float(theta[3]), cs.rate)
    sk = st.skew(cs.pricer, theta)
    psi_mc = np.array([r["psi"] for r in mc_rows])
    se_mc = np.array([r["se"] for r in mc_rows])
    diff = sk["psi"] - psi_mc
    f_map = model_exponent(taus, sk["psi"])
    f_mc = ats.fit_power_law(taus, psi_mc, se_mc)
    f_mc_unw = model_exponent(taus, psi_mc)
    return {"taus": list(map(float, taus)), "h_mc": [r["h"] for r in mc_rows],
            "psi_map": sk["psi"].tolist(), "psi_map_richardson": sk["psi_richardson"].tolist(),
            "psi_mc": psi_mc.tolist(), "se_mc": se_mc.tolist(),
            "psi_mc_richardson": [r["psi_richardson"] for r in mc_rows],
            "diff": diff.tolist(), "pull": (diff / se_mc).tolist(),
            "diff_rms": float(np.sqrt(np.mean(diff ** 2))),
            "diff_max_abs": float(np.max(np.abs(diff))),
            "diff_rel_rms": float(np.sqrt(np.mean((diff / psi_mc) ** 2))),
            "atm_iv_map": sk["atm_iv"].tolist(), "atm_iv_mc": [r["atm_iv"] for r in mc_rows],
            "b_map": f_map.get("b"), "b_mc": f_mc.get("b"), "se_b_mc": f_mc.get("se_b"),
            "b_mc_unweighted": f_mc_unw.get("b"),
            "b_diff_sigmas": ((f_map["b"] - f_mc["b"]) / f_mc["se_b"]
                              if np.isfinite(f_mc.get("se_b", np.nan)) and f_mc.get("se_b", 0) > 0 else None)}


# ── (6) report / figure / artifact ─────────────────────────────────────────
BG, PANEL, BLUE, AMBER, MINT, WHITE, GRID, GREY = (
    "#000000", "#1c1c1e", "#0A84FF", "#FF9F0A", "#30D158", "#FFFFFF", "#3a3a3c", "#8E8E93")


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=WHITE, labelsize=11, which="both")
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)


def grid_matrix(states: list[dict], key: str) -> np.ndarray:
    """Cell value averaged over captures, shape (n_eta, n_H)."""
    nH, nE = len(states[0]["H_grid"]), len(states[0]["eta_grid"])
    acc = np.zeros((nE, nH))
    cnt = np.zeros((nE, nH))
    for s in states:
        for c in s["cells"].values():
            v = c.get(key)
            if v is not None and np.isfinite(v):
                acc[c["j"], c["i"]] += v
                cnt[c["j"], c["i"]] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def make_figure(payload: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], left=0.05, right=0.955,
                          top=0.875, bottom=0.07, wspace=0.22, hspace=0.38)
    prof = payload["profile"]
    Hs, Es = np.array(prof["H_grid"]), np.array(prof["eta_grid"])
    rmse = np.array(prof["rmse_volpts_mean"])
    bmat = np.array(prof["b_expiries_mean"])
    b_mkt = payload["market"]["pooled_8_captures"]["b"]
    served = payload["served"]["theta"]
    joint = payload["recommended"]["theta"]
    pure = payload["pure_smile"]["theta"]
    extent = [Hs[0] - (Hs[1] - Hs[0]) / 2, Hs[-1] + (Hs[1] - Hs[0]) / 2,
              Es[0] - (Es[1] - Es[0]) / 2, Es[-1] + (Es[1] - Es[0]) / 2]

    def marks(ax):
        ax.plot(served[2], served[0], marker="*", ms=17, color=BLUE, mec=WHITE, mew=0.8,
                ls="none", label="served calibration")
        ax.plot(pure[2], pure[0], marker="o", ms=9, color=BLUE, mec=WHITE, mew=0.8,
                ls="none", label="pure smile optimum (lambda = 0)")
        ax.plot(joint[2], joint[0], marker="*", ms=17, color=MINT, mec=WHITE, mew=0.8,
                ls="none", label=f"joint optimum (lambda = {payload['recommended']['lambda']:g})")
        cs = ax.contour(Hs, Es, bmat, levels=[b_mkt], colors=[AMBER], linewidths=2.0,
                        linestyles="--")
        ax.clabel(cs, fmt={b_mkt: f"b = {b_mkt:+.3f} (market)"}, colors=[AMBER], fontsize=10)
        ax.set_xlabel("H")
        ax.set_ylabel("eta")

    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1)
    im1 = ax1.imshow(rmse, origin="lower", extent=extent, aspect="auto", cmap="magma_r",
                     vmin=np.nanmin(rmse), vmax=min(np.nanmax(rmse), np.nanmin(rmse) + 1.5))
    cb = fig.colorbar(im1, ax=ax1, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE)
    cb.set_label("smile RMSE (vol points), mean of 3 captures", color=WHITE)
    marks(ax1)
    ax1.set_title("Smile RMSE with (rho, xi) optimised at each (H, eta)")
    ax1.legend(loc="lower right", fontsize=9, facecolor=BG, edgecolor=GRID,
               labelcolor=WHITE, framealpha=0.85)

    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    span = max(abs(np.nanmin(bmat) - b_mkt), abs(np.nanmax(bmat) - b_mkt))
    im2 = ax2.imshow(bmat, origin="lower", extent=extent, aspect="auto", cmap="RdBu",
                     norm=TwoSlopeNorm(vcenter=b_mkt, vmin=b_mkt - span, vmax=b_mkt + span))
    cb = fig.colorbar(im2, ax=ax2, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE)
    cb.set_label("model skew exponent b over the capture expiries (map)", color=WHITE)
    eta_lic = payload.get("licence_eta_min")
    if eta_lic is not None:
        ax2.fill_between([extent[0], extent[1]], extent[2], eta_lic, facecolor="none",
                         edgecolor=GREY, hatch="///", linewidth=0.0, alpha=0.9)
        ax2.axhline(eta_lic, color=GREY, lw=1.0, ls="-")
        ax2.text(extent[0] + 0.005, eta_lic + 0.04, "map psi not licensed vs MC below this eta",
                 color=WHITE, fontsize=8.5, va="bottom")
    marks(ax2)
    ax2.set_title(f"ATM skew exponent b (2-10 d); amber dashes: market b = {b_mkt:+.3f}")

    # Pareto front
    ax3 = fig.add_subplot(gs[0, 1])
    _style(ax3)
    ax3.grid(True, color=GRID, lw=0.6, alpha=0.8)
    pf = payload["pareto"]["mean_over_captures"]
    x = np.array([r["rmse_volpts"] for r in pf["rows"]])
    y = np.array([r["b_error"] for r in pf["rows"]])
    c2 = np.array([r["chi2"] for r in pf["rows"]])
    ax3.plot(x, y, "-o", color=BLUE, lw=1.8, ms=7, label="joint fits, lambda = 0 ... 10 (mean of 3 captures)")
    for r, xi_, yi_ in zip(pf["rows"], x, y):
        ax3.annotate(f"{r['lambda']:g}", (xi_, yi_), textcoords="offset points",
                     xytext=(6, 5), color=WHITE, fontsize=9)
    k = pf["knee_index"]
    if k is not None:
        ax3.plot(x[k], y[k], marker="*", ms=18, color=MINT, mec=WHITE, mew=0.8, ls="none",
                 label=f"knee: lambda = {pf['rows'][k]['lambda']:g}")
    se_m = payload["market"]["pooled_8_captures"]["se_b"]
    ax3.axhline(se_m, color=AMBER, ls=":", lw=1.5, label=f"one market SE on b ({se_m:.3f})")
    sv = payload["served"]
    ax3.plot(sv["rmse_volpts_capture_mean"], abs(sv["b_expiries_capture_mean"] - b_mkt),
             marker="*", ms=16, color=BLUE, mec=WHITE, mew=0.8, ls="none",
             label="served calibration (map, same captures)")
    ax3.set_xlabel("smile RMSE (vol points, map)")
    ax3.set_ylabel("|b_model - b_market| over the capture expiries")
    ax3.set_title("Pareto front: smile fit vs skew-exponent error")
    ax3b = ax3.twinx()
    ax3b.plot(x, c2, "--", color=GREY, lw=1.2, marker="s", ms=4, label="skew chi2 / expiry (right axis)")
    ax3b.set_yscale("log")
    ax3b.tick_params(colors=GREY, labelsize=10)
    ax3b.set_ylabel("skew chi2 per expiry", color=GREY)
    for s in ax3b.spines.values():
        s.set_color(GRID)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, facecolor=BG,
               edgecolor=GRID, labelcolor=WHITE, framealpha=0.85)

    # skew term structure
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4)
    ax4.grid(True, which="both", color=GRID, lw=0.5, alpha=0.7)
    mk = payload["market"]["rows_8_captures"]
    Tm = np.array([r["tau"] for r in mk])
    pm = np.array([r["psi"] for r in mk])
    sm = np.array([r["se"] for r in mk])
    neg = pm < 0
    ax4.errorbar(Tm[neg], -pm[neg], yerr=sm[neg], fmt="D", color=AMBER, ms=4.5, alpha=0.85,
                 ecolor=AMBER, elinewidth=0.8, capsize=0,
                 label=f"market, 8 captures x expiries (n = {int(neg.sum())})")
    fk = payload["market"]["pooled_8_captures"]
    Tf = np.geomspace(fk["T_min"], fk["T_max"], 40)
    ax4.plot(Tf, fk["C"] * Tf ** fk["b"], "--", color=AMBER, lw=1.6,
             label=f"market fit: b = {fk['b']:+.3f} +- {fk['se_b']:.3f}")
    mc = payload["mc"]
    lam_int = (mc.get("sets", {}).get("interior") or {}).get("lambda")
    for key, col, name, fmt in (("served_ladder", BLUE, "served (MC)", "-o"),
                                ("joint_ladder", MINT, f"joint, lambda = {mc.get('lambda', float('nan')):g} (MC)", "-o"),
                                ("interior_ladder", MINT, f"interior eta, lambda = {lam_int} (MC)", "--s"),
                                ("pure_ladder", BLUE, "pure smile, lambda = 0 (MC)", ":^")):
        rows = mc.get(key)
        if not rows:
            continue
        T = np.array([r["T"] for r in rows])
        p = np.abs([r["psi"] for r in rows])
        s = np.array([r["se"] for r in rows])
        f = mc.get(key + "_fit", {})
        ax4.errorbar(T, p, yerr=s, fmt=fmt, color=col, lw=1.6 if fmt != "-o" else 2.0, ms=4.5,
                     ecolor=col, capsize=2, alpha=0.8 if fmt != "-o" else 1.0,
                     label=f"{name}: b(T<=45d) = {f.get('b', float('nan')):+.3f} +- {f.get('se_b', float('nan')):.3f}")
    for key, col, name in (("served_map_curve", BLUE, "served (map)"), ("joint_map_curve", MINT, "joint (map)")):
        rows = mc.get(key)
        if rows:
            ax4.plot([r["tau"] for r in rows], np.abs([r["psi"] for r in rows]), ls="none", marker="x",
                     color=col, ms=6, mew=1.5, label=f"{name}, capture expiries")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("T (years, log)")
    ax4.set_ylabel("|psi(T)| = |d sigma_imp / dk| at k = 0")
    ax4.set_title("ATM skew term structure: market, served and joint parameters")
    # the legend sits in the empty band above the 1-day points (ylim leaves
    # room for it) so that no market point or ladder point is covered
    ax4.legend(loc="upper center", ncol=2, fontsize=8, facecolor=BG, edgecolor=GRID,
               labelcolor=WHITE, framealpha=0.9, columnspacing=1.0, handlelength=2.2)
    ax4.set_ylim(0.2, 18.0)
    rec = payload["recommended"]
    fig.suptitle("Joint (H, eta) refit of rough Bergomi on SPY: smile fit vs ATM skew term structure",
                 color=WHITE, fontsize=18, y=0.985)
    th = rec["theta"]
    fig.text(0.5, 0.945, (f"Recommended lambda = {rec['lambda']:g}: eta = {th[0]:.3f}, rho = {th[1]:.3f}, "
                         f"H = {th[2]:.3f}, sqrt(xi) = {math.sqrt(th[3]):.3f}; smile RMSE {rec['rmse_volpts']:.3f} vp "
                         f"vs {payload['pure_smile']['rmse_volpts']:.3f} pure smile; MC ladder exponent "
                         f"{mc.get('joint_ladder_fit', {}).get('b', float('nan')):+.3f} +- "
                         f"{mc.get('joint_ladder_fit', {}).get('se_b', float('nan')):.3f} "
                         f"(served {mc.get('served_ladder_fit', {}).get('b', float('nan')):+.3f}, market {b_mkt:+.3f})"),
             color=GREY, fontsize=11, ha="center", va="top")
    fig.savefig(out_png, dpi=100, facecolor=BG)
    plt.close(fig)


def mean_rows(list_of_fit_dicts: list[dict], lambdas=LAMBDAS) -> list[dict]:
    """Average per-lambda rows across captures (theta averaged too)."""
    out = []
    for lam in lambdas:
        key = f"{lam:g}"
        rs = [d["fits"][key] for d in list_of_fit_dicts if key in d["fits"]]
        if not rs:
            continue
        out.append({"lambda": lam,
                    "rmse_volpts": float(np.mean([r["rmse_volpts"] for r in rs])),
                    "smile_loss": float(np.mean([r["smile_loss"] for r in rs])),
                    "chi2": float(np.mean([r["chi2"] for r in rs])),
                    "b_expiries": float(np.mean([r["b_expiries"] for r in rs])),
                    "theta": np.mean([r["theta"] for r in rs], axis=0).tolist(),
                    "theta_sd": (np.std([r["theta"] for r in rs], axis=0, ddof=1).tolist()
                                 if len(rs) > 1 else None),
                    "eta_interior": all(r["eta_interior"] for r in rs),
                    "n_captures": len(rs)})
    return out


# ── driver ─────────────────────────────────────────────────────────────────
def make_logger(log_file: Path | None):
    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    return log


def stage_skew_check(work: Path, caps: list[Path], n_paths: int, n_reps: int, log) -> dict:
    """Map psi vs MC psi at the served parameters on capture[1]'s expiries."""
    cs = CaptureSet(caps[1] if len(caps) > 1 else caps[0])
    th = served_theta()
    out = load_json(work / "skew_check.json", {})
    if out.get("done"):
        return out
    log(f"[skew-check] {cs.path.name}: MC {n_reps} x {n_paths:,} paths at served params on "
        f"{len(cs.taus)} expiries")
    rows = mc_skew_on_taus(th, cs.taus, cs.meta["spot"], cs.rate, n_paths, n_reps, 20260910, log)
    cmp_mc = compare_map_vs_mc(cs, th, rows)
    ev = cs.evaluate(th)
    out = {"capture": cs.info(), "theta": th.tolist(), "mc_rows": rows, "compare": cmp_mc,
           "map_eval": ev, "n_paths": n_paths, "n_reps": n_reps, "done": True, "generated": now()}
    save_json(work / "skew_check.json", out)
    log(f"[skew-check] psi map vs MC: rms diff {cmp_mc['diff_rms']:.4f}, max |diff| "
        f"{cmp_mc['diff_max_abs']:.4f}, pulls {np.round(cmp_mc['pull'], 2).tolist()}; "
        f"b map {cmp_mc['b_map']:+.4f} vs MC {cmp_mc['b_mc']:+.4f} +- {cmp_mc['se_b_mc']:.4f}")
    return out


def stage_profile(work: Path, caps: list[Path], idx: list[int], log,
                  H_grid=H_GRID, eta_grid=ETA_GRID) -> None:
    pricer = MapPricer()
    for i in idx:
        cs = CaptureSet(caps[i], pricer)
        log(f"[profile] capture {i}: {cs.path.name} ({len(cs.cal.quotes)} quotes, "
            f"{len(cs.taus)} expiries, market b {cs.market_fit.get('b'):+.4f})")
        st = profile_capture(cs, work / f"profile_{i}.json", log, H_grid, eta_grid)
        log(f"[profile] capture {i} done: {len(st['cells'])} cells")


def stage_joint(work: Path, caps: list[Path], idx: list[int], log, lambdas=LAMBDAS) -> None:
    pricer = MapPricer()
    for i in idx:
        cs = CaptureSet(caps[i], pricer)
        prof = load_json(work / f"profile_{i}.json", None)
        log(f"[joint] capture {i}: {cs.path.name}; profile cells available: "
            f"{len(prof['cells']) if prof else 0}")
        joint_capture(cs, work / f"joint_{i}.json", prof, log, lambdas)


def stage_eta_extend(work: Path, caps: list[Path], idx: list[int], log,
                     eta_max: float | None = None, H_grid=H_GRID,
                     lambdas=(0.0, 0.1), step: float = 0.5, seed: int = 7,
                     mc_capture: int = 1, mc_paths: int = 200_000,
                     mc_reps: int = 4, mc_smile_paths: int = 0) -> dict:
    """Does the smile optimum sit AT eta 4, or is 4 where the optimiser stops?

    Re-runs the profile strip and the joint fits with eta free up to the
    COMMITTED map's own training ceiling (map_eta_box, 8.0) beside the same
    fits capped at the calibrator's 4.0, on the same captures, same seed and
    same map. Nothing is regenerated and nothing served changes; the map is
    deterministic, so the only cost is a few minutes of CPU.

    The map has no Monte Carlo licence above eta 4 (stage_licence covers the
    profile's own box), so the winning extended point is checked against
    rough_bergomi_mc on the capture's expiries before anything is concluded
    from it; `--smile-paths` > 0 additionally reprices the whole capture with
    the true engine, which is what decides whether a map RMSE that improves
    above eta 4 is real or is the surrogate's own error. That check earns its
    cost: on spy_20260821T150017Z the pure-smile arm's 0.02 vp map gain at
    eta 8 does NOT survive the engine (1.53 -> 1.55 vp over all 585 quotes,
    1.34 -> 1.44 inside z < 2), while the lambda = 0.1 knee's 0.08 vp gain
    does (2.12 -> 2.01, 1.64 -> 1.60). Keep --smile-paths > 0 before quoting
    any extended smile RMSE. docs/joint_skew_refit.md section 3.5.
    """
    pricer = MapPricer()
    eta_lo_box, eta_hi_box = map_eta_box(pricer)
    eta_hi = float(eta_max) if eta_max is not None else eta_hi_box
    eta_grid = extended_eta_grid(eta_hi, step=step)
    ext_bounds = extended_bounds(eta_hi)
    out_path = work / "eta_extend.json"
    out = load_json(out_path, {})
    out.update({"map_box_eta": [eta_lo_box, eta_hi_box],
                "calibrator_eta_bounds": list(BOUNDS["eta"]),
                "map_file": "artifacts/pricing_map.pt",
                "eta_grid": eta_grid.tolist(), "H_grid": list(map(float, H_grid)),
                "lambdas": list(lambdas), "bounds_extended": [list(b) for b in ext_bounds],
                "seed": seed})
    log(f"[eta-extend] map box eta {eta_lo_box}-{eta_hi_box}; calibrator bound "
        f"{BOUNDS['eta'][1]}; strip eta {eta_grid[0]:g}..{eta_grid[-1]:g} "
        f"({len(eta_grid)} x {len(H_grid)} cells per capture)")
    per_capture = out.get("per_capture", {})
    for i in idx:
        cs = CaptureSet(caps[i], pricer)
        log(f"[eta-extend] capture {i}: {cs.path.name}")
        strip = profile_capture(cs, work / f"eta_extend_profile_{i}.json", log,
                                H_grid, eta_grid)
        cells = list(strip["cells"].values())
        best_cell = min(cells, key=lambda c: c["smile_loss"])
        anchor = min((c for c in cells if abs(c["eta"] - BOUNDS["eta"][1]) < 1e-9),
                     key=lambda c: c["smile_loss"])
        # the four-parameter fits, capped and extended, from the same seed
        fits_ckpt = work / f"eta_extend_joint_{i}.json"
        fstate = load_json(fits_ckpt, {"capture": cs.path.name, "seed": seed, "fits": {}})
        for lam in lambdas:
            for tag, bnds in (("capped", JOINT_BOUNDS), ("extended", ext_bounds)):
                key = f"{lam:g}|{tag}"
                if key in fstate["fits"]:
                    continue
                obj = JointObjective(cs, lam)
                starts = [("served", served_theta()),
                          ("strip_best", np.array([best_cell["eta"], best_cell["rho"],
                                                   best_cell["H"], best_cell["xi"]]))]
                r = joint_fit(obj, seed=seed, extra_starts=starts, bounds=bnds)
                ev = cs.evaluate(r["theta"])
                fstate["fits"][key] = {
                    "lambda": lam, "bounds": tag, **ev, "J": r["J"],
                    "eta_interior_map_box": eta_interior(float(r["theta"][0]), eta_lo_box, eta_hi),
                    "winner": r["winner"], "nfev_total": r["nfev_total"],
                    "seconds": r["seconds"]}
                save_json(fits_ckpt, fstate)
                th = r["theta"]
                log(f"  [{cs.path.name[4:19]}] lambda={lam:g} {tag}: eta={th[0]:.4f} "
                    f"rho={th[1]:.4f} H={th[2]:.4f} sqrt(xi)={math.sqrt(th[3]):.4f} "
                    f"rmse={ev['rmse_volpts']:.3f} b={ev['b_expiries']:+.3f} "
                    f"chi2={ev['chi2']:.2f} J={r['J']:.3f} ({r['seconds']:.0f}s)")
        keep = ("H", "eta", "rho", "xi", "rmse_volpts", "smile_loss", "b_expiries", "chi2")
        per_capture[str(i)] = {
            "capture": cs.path.name,
            "best_cell_extended": {k: best_cell[k] for k in keep},
            "best_cell_at_bound": {k: anchor[k] for k in keep},
            "rmse_gain_volpts": anchor["rmse_volpts"] - best_cell["rmse_volpts"],
            "best_cell_by_eta": {f"{e:g}": {k: min((c for c in cells if abs(c["eta"] - e) < 1e-9),
                                                  key=lambda c: c["smile_loss"])[k] for k in keep}
                                 for e in eta_grid},
            "fits": fstate["fits"],
            "market_b": cs.market_fit.get("b"), "market_se_b": cs.market_fit.get("se_b")}
        out["per_capture"] = per_capture
        save_json(out_path, out)
        log(f"[eta-extend] capture {i}: best strip cell eta {best_cell['eta']:g}, "
            f"H {best_cell['H']:.3f}, RMSE {best_cell['rmse_volpts']:.3f} vp "
            f"(at the 4.0 bound {anchor['rmse_volpts']:.3f} vp, H {anchor['H']:.3f}); "
            f"b {best_cell['b_expiries']:+.3f} vs {anchor['b_expiries']:+.3f}")
    # Monte Carlo check of the map where it has no licence (eta > 4): both
    # ends of every capped/extended pair on one capture, because the extended
    # arm's whole claim - that the ridge is flat in eta and the knee moves off
    # the bound - is a claim about the surrogate until the true engine says so.
    mc_check = out.get("mc_check", {})
    if per_capture.get(str(mc_capture)) and mc_paths:
        cs = CaptureSet(caps[mc_capture], pricer)
        blk = per_capture[str(mc_capture)]
        pairs = [(f"{lam:g}|{tag}", np.array(blk["fits"][f"{lam:g}|{tag}"]["theta"], dtype=float))
                 for lam in lambdas for tag in ("capped", "extended")
                 if f"{lam:g}|{tag}" in blk["fits"]]
        for name, th in pairs:
            if name not in mc_check:
                log(f"[eta-extend] MC check {name}: theta {np.round(th, 4).tolist()}, "
                    f"{mc_reps} x {mc_paths:,} paths on {len(cs.taus)} expiries")
                rows = mc_skew_on_taus(th, cs.taus, cs.meta["spot"], cs.rate, mc_paths,
                                       mc_reps, 20260917, log)
                cmp_mc = compare_map_vs_mc(cs, th, rows)
                psi_mc = np.array([r["psi"] for r in rows])
                mc_check[name] = {"theta": th.tolist(), "compare": cmp_mc,
                                  "chi2_mc": skew_chi2(psi_mc, cs.psi_mkt, cs.se_mkt),
                                  "b_expiries_mc": ats.fit_power_law(
                                      cs.taus, psi_mc, [r["se"] for r in rows]).get("b")}
                log(f"[eta-extend] MC {name}: psi rel rms diff {cmp_mc['diff_rel_rms']:.3f}, "
                    f"b map {cmp_mc['b_map']:+.4f} vs MC {cmp_mc['b_mc']:+.4f} "
                    f"+- {cmp_mc['se_b_mc']:.4f}")
                out["mc_check"] = mc_check
                save_json(out_path, out)
            if mc_smile_paths and "smile" not in mc_check[name]:
                log(f"[eta-extend] MC smile RMSE {name}: {len(cs.cal.quotes)} quotes, "
                    f"{mc_smile_paths:,} paths")
                mc_check[name]["smile"] = mc_smile_rmse(cs, th, mc_smile_paths, 4242, log)
                out["mc_check"] = mc_check
                save_json(out_path, out)
    out["generated"] = now()
    out["done"] = True
    save_json(out_path, out)
    return out


def recommended_lambda(work: Path, n_caps: int, b_market: float, se_b_market: float) -> tuple[float, dict]:
    joints = [load_json(work / f"joint_{i}.json", None) for i in range(n_caps)]
    joints = [j for j in joints if j]
    rows = mean_rows(joints)
    pf = pareto_table(rows, b_market, se_b_market)
    return pf["recommended_lambda"], pf


def market_8_captures() -> dict:
    caps8 = ats.trading_hour_captures(CAPTURE_GLOB, 8)
    all_rows, per_cap = [], []
    for c in caps8:
        rows, info = ats.market_skew_from_capture(c)
        f = ats.fit_power_law([r["tau"] for r in rows], [r["psi"] for r in rows], [r["se"] for r in rows])
        per_cap.append({"capture": c.name, "pricing_time": info["pricing_time"],
                        "b": f.get("b"), "se_b": f.get("se_b"), "n": f.get("n")})
        all_rows += [dict(r, capture=c.name) for r in rows]
    pooled = ats.fit_power_law([r["tau"] for r in all_rows], [r["psi"] for r in all_rows],
                               [r["se"] for r in all_rows],
                               [r["T_days_equiv"] <= SHORT_MAX_DAYS for r in all_rows])
    return {"captures": [c.name for c in caps8], "rows_8_captures": all_rows,
            "per_capture": per_cap, "pooled_8_captures": pooled}


def stage_bootstrap(work: Path, caps: list[Path], cap_index: int, lam: float | None,
                    reps: int, rep_offset: int, log) -> None:
    pricer = MapPricer()
    cs = CaptureSet(caps[cap_index], pricer)
    if lam is None:
        mk = market_8_captures()["pooled_8_captures"]
        lam, _ = recommended_lambda(work, len(caps), mk["b"], mk["se_b"])
    joint = load_json(work / f"joint_{cap_index}.json", None)
    theta_hat = np.array(joint["fits"][f"{lam:g}"]["theta"])
    log(f"[bootstrap] capture {cap_index} lambda={lam:g} reps {rep_offset}..{rep_offset + reps - 1} "
        f"from theta_hat {np.round(theta_hat, 4).tolist()}")
    bootstrap_capture(cs, lam, theta_hat, work / f"boot_{cap_index}_{rep_offset}.json", log,
                      reps, rep_offset)


def mc_parameter_sets(work: Path, cap_index: int, lam: float) -> dict:
    """Named parameter vectors the MC stage validates on capture `cap_index`:
    the joint optimum at `lam`, the served calibration, the pure-smile
    optimum (lambda = 0) and the smallest-lambda optimum with eta interior."""
    fits = load_json(work / f"joint_{cap_index}.json", None)["fits"]
    sets = {"joint": {"theta": list(map(float, fits[f"{lam:g}"]["theta"])), "lambda": lam},
            "served": {"theta": served_theta().tolist(), "lambda": None},
            "pure": {"theta": list(map(float, fits["0"]["theta"])), "lambda": 0.0}}
    # the smallest-lambda interior-eta optimum, or, when that is the joint
    # optimum itself, the next interior one (a clearly interior point)
    ic = interior_candidate(fits)
    if ic is not None and ic["lambda"] == lam:
        ic = interior_candidate(fits, lambdas=[l for l in LAMBDAS if l > lam])
    if ic is not None:
        sets["interior"] = {"theta": list(map(float, ic["theta"])), "lambda": ic["lambda"]}
    return sets


MC_SEEDS = {"joint": 20260911, "served": 20260912, "pure": 20260913, "interior": 20260914}


def stage_mc(work: Path, caps: list[Path], cap_index: int, lam: float | None,
             n_paths_ladder: int, n_reps_ladder: int, n_paths_smile: int, log,
             smile_sets=("joint", "served", "pure", "interior")) -> dict:
    out_path = work / "mc.json"
    out = load_json(out_path, {})
    cs = CaptureSet(caps[cap_index])
    if lam is None:
        mk = market_8_captures()["pooled_8_captures"]
        lam, _ = recommended_lambda(work, len(caps), mk["b"], mk["se_b"])
    sets = mc_parameter_sets(work, cap_index, lam)
    out.update({"capture": cs.info(), "lambda": lam, "sets": sets,
                "theta_joint": sets["joint"]["theta"], "theta_served": sets["served"]["theta"],
                "n_paths_ladder": n_paths_ladder, "n_reps_ladder": n_reps_ladder,
                "n_paths_smile": n_paths_smile, "ladder_T_days": list(LADDER)})
    spot, rate = cs.meta["spot"], cs.rate
    for name, blk in sets.items():
        th = np.array(blk["theta"], dtype=float)
        seed = MC_SEEDS[name]
        if f"{name}_ladder" not in out:
            log(f"[mc] {name} ladder: {n_reps_ladder} x {n_paths_ladder:,} paths, {len(LADDER)} maturities, "
                f"theta {np.round(th, 4).tolist()}")
            params = {"eta": th[0], "rho": th[1], "H": th[2], "xi": th[3], "spot": spot, "rate": rate}
            rows = ats.model_skew_curve(params, LADDER, n_paths_ladder, n_reps_ladder, seed, log=log)
            short = [r["T_days"] <= SHORT_MAX_DAYS for r in rows]
            f = ats.fit_power_law([r["T"] for r in rows], [r["psi"] for r in rows], [r["se"] for r in rows], short)
            fr = ats.fit_power_law([r["T"] for r in rows], [r["psi_richardson"] for r in rows],
                                   [r["se_richardson"] for r in rows], short)
            out[f"{name}_ladder"] = rows
            out[f"{name}_ladder_fit"] = f
            out[f"{name}_ladder_fit_richardson"] = fr
            out[f"{name}_ladder_local"] = ats.local_exponents(rows)
            save_json(out_path, out)
            log(f"[mc] {name} ladder b(T<=45d) = {f['b']:+.4f} +- {f['se_b']:.4f} "
                f"(Richardson {fr['b']:+.4f}); H - 1/2 = {th[2] - 0.5:+.4f}")
        if f"{name}_expiries" not in out:
            log(f"[mc] {name} on the capture's {len(cs.taus)} expiries")
            rows = mc_skew_on_taus(th, cs.taus, spot, rate, n_paths_ladder, n_reps_ladder, seed + 5, log)
            out[f"{name}_expiries"] = rows
            out[f"{name}_map_vs_mc"] = compare_map_vs_mc(cs, th, rows)
            st = cs.stencil.skew(cs.pricer, th)
            out[f"{name}_map_curve"] = [{"tau": float(t), "psi": float(p)} for t, p in zip(cs.taus, st["psi"])]
            psi_mc = np.array([r["psi"] for r in rows])
            out[f"{name}_chi2_mc"] = skew_chi2(psi_mc, cs.psi_mkt, cs.se_mkt)
            out[f"{name}_chi2_map"] = skew_chi2(st["psi"], cs.psi_mkt, cs.se_mkt)
            fe = ats.fit_power_law(cs.taus, psi_mc, [r["se"] for r in rows])
            out[f"{name}_b_expiries_mc"] = fe.get("b")
            out[f"{name}_b_expiries_mc_se"] = fe.get("se_b")
            out[f"{name}_b_expiries_map"] = model_exponent(cs.taus, st["psi"]).get("b")
            save_json(out_path, out)
            c = out[f"{name}_map_vs_mc"]
            log(f"[mc] {name} map vs MC on expiries: rms diff {c['diff_rms']:.4f} "
                f"(rel {c['diff_rel_rms']:.3f}), b map {c['b_map']:+.4f} vs MC {c['b_mc']:+.4f} "
                f"+- {c['se_b_mc']:.4f}; chi2 map {out[f'{name}_chi2_map']:.2f} vs MC {out[f'{name}_chi2_mc']:.2f}")
    for name, blk in sets.items():
        if name not in smile_sets or f"{name}_smile" in out:
            continue
        th = np.array(blk["theta"], dtype=float)
        log(f"[mc] {name} smile RMSE by MC: {len(cs.cal.quotes)} quotes, {n_paths_smile:,} paths")
        out[f"{name}_smile"] = mc_smile_rmse(cs, th, n_paths_smile, 4242, log)
        save_json(out_path, out)
        sm = out[f"{name}_smile"]
        log(f"[mc] {name} smile: MC RMSE {sm['rmse_mc_volpts']:.3f} vp (noise floor "
            f"{sm['mc_noise_floor_volpts']:.3f}), map RMSE {sm['rmse_map_volpts']:.3f} vp; z<2: MC "
            f"{sm['z_lt_2']['rmse_mc_volpts']:.3f} map {sm['z_lt_2']['rmse_map_volpts']:.3f} map-MC rms "
            f"{sm['z_lt_2']['map_minus_mc_rms_volpts']:.3f} noise {sm['z_lt_2']['mc_noise_volpts']:.3f}")
    out["done"] = True
    out["generated"] = now()
    save_json(out_path, out)
    return out


def stage_licence(work: Path, caps: list[Path], n_paths: int, n_reps: int, log,
                  H_target: float = 0.268, etas=(0.5, 1.45, 2.41)) -> dict:
    """Map-vs-MC psi at profile cells on the market-exponent ridge (H near
    0.27) with small eta: where |psi| is small the map's absolute error is a
    large relative error and the profile's exponent there is not licensed."""
    out_path = work / "skew_check_cells.json"
    out = load_json(out_path, {"cells": {}})
    cs = CaptureSet(caps[1] if len(caps) > 1 else caps[0])
    prof = load_json(work / "profile_1.json", None)
    if prof is None:
        return out
    Hs, Es = np.array(prof["H_grid"]), np.array(prof["eta_grid"])
    i = int(np.argmin(np.abs(Hs - H_target)))
    for eta in etas:
        j = int(np.argmin(np.abs(Es - eta)))
        key = f"{i}_{j}"
        if key in out["cells"]:
            continue
        c = prof["cells"][key]
        th = np.array([c["eta"], c["rho"], c["H"], c["xi"]])
        log(f"[licence] cell {key} H={c['H']:.3f} eta={c['eta']:.3f} rho={c['rho']:+.3f} "
            f"sqrt(xi)={math.sqrt(c['xi']):.4f}: MC {n_reps} x {n_paths:,} on {len(cs.taus)} expiries")
        rows = mc_skew_on_taus(th, cs.taus, cs.meta["spot"], cs.rate, n_paths, n_reps, 20260920 + j, log)
        cmp_ = compare_map_vs_mc(cs, th, rows)
        st = cs.stencil.skew(cs.pricer, th)
        psi_mc = np.array([r["psi"] for r in rows])
        out["cells"][key] = {
            "theta": th.tolist(), "H": c["H"], "eta": c["eta"], "rho": c["rho"], "xi": c["xi"],
            "mc_rows": rows, "compare": cmp_,
            "psi_map_market_stencil": st["psi"].tolist(),
            "b_profile": c["b_expiries"], "chi2_profile": c["chi2"], "rmse_profile": c["rmse_volpts"],
            "b_mc": cmp_["b_mc"], "se_b_mc": cmp_["se_b_mc"],
            "chi2_mc": skew_chi2(psi_mc, cs.psi_mkt, cs.se_mkt),
            "psi_mc_mean_abs": float(np.mean(np.abs(psi_mc)))}
        save_json(out_path, out)
        log(f"[licence] cell {key}: psi map {np.round(cmp_['psi_map'], 3).tolist()} vs MC "
            f"{np.round(cmp_['psi_mc'], 3).tolist()}; rel rms {cmp_['diff_rel_rms']:.3f}; "
            f"b map {cmp_['b_map']:+.4f} vs MC {cmp_['b_mc']:+.4f} +- {cmp_['se_b_mc']:.4f}")
    out["capture"] = cs.info()
    out["n_paths"], out["n_reps"] = n_paths, n_reps
    out["done"] = True
    save_json(out_path, out)
    return out


def stage_report(work: Path, caps: list[Path], out_dir: Path, artifact: Path, log) -> dict:
    n = len(caps)
    pricer = MapPricer()
    css = [CaptureSet(c, pricer) for c in caps]
    market = market_8_captures()
    b_mkt, se_mkt = market["pooled_8_captures"]["b"], market["pooled_8_captures"]["se_b"]
    profiles = [load_json(work / f"profile_{i}.json", None) for i in range(n)]
    joints = [load_json(work / f"joint_{i}.json", None) for i in range(n)]
    skew_check = load_json(work / "skew_check.json", {})
    mc = load_json(work / "mc.json", {})
    boots = [load_json(p, None) for p in sorted(work.glob("boot_*.json"))]
    boots = [b for b in boots if b]

    # market over the 3 captures used for fitting
    rows3 = [dict(r, capture=cs.path.name) for cs in css for r in cs.market_rows]
    pooled3 = ats.fit_power_law([r["tau"] for r in rows3], [r["psi"] for r in rows3],
                                [r["se"] for r in rows3])

    # profile summaries
    prof_summary = {"H_grid": profiles[0]["H_grid"], "eta_grid": profiles[0]["eta_grid"],
                    "rmse_volpts_mean": grid_matrix(profiles, "rmse_volpts").tolist(),
                    "b_expiries_mean": grid_matrix(profiles, "b_expiries").tolist(),
                    "chi2_mean": grid_matrix(profiles, "chi2").tolist(),
                    "smile_loss_mean": grid_matrix(profiles, "smile_loss").tolist(),
                    "per_capture": []}
    for i, p in enumerate(profiles):
        cells = list(p["cells"].values())
        best = min(cells, key=lambda c: c["smile_loss"])
        on_contour = [c for c in cells if abs(c["b_expiries"] - b_mkt) <= se_mkt]
        best_on = min(on_contour, key=lambda c: c["smile_loss"]) if on_contour else None
        prof_summary["per_capture"].append({
            "capture": p["capture"], "n_cells": len(cells),
            "seconds": p.get("seconds_total"),
            "best_smile_cell": {k: best[k] for k in ("H", "eta", "rho", "xi", "rmse_volpts", "b_expiries", "chi2")},
            "n_cells_within_1se_of_market_b": len(on_contour),
            "best_smile_cell_within_1se_of_market_b": (
                {k: best_on[k] for k in ("H", "eta", "rho", "xi", "rmse_volpts", "b_expiries", "chi2")}
                if best_on else None),
            "rmse_range": [min(c["rmse_volpts"] for c in cells), max(c["rmse_volpts"] for c in cells)],
            "b_range": [min(c["b_expiries"] for c in cells), max(c["b_expiries"] for c in cells)],
            "nfev_mean": float(np.mean([c["nfev"] for c in cells]))})

    # pareto per capture and mean
    pareto = {"per_capture": [], "mean_over_captures": None}
    for j in joints:
        rows = list(j["fits"].values())
        pareto["per_capture"].append({"capture": j["capture"], **pareto_table(rows, b_mkt, se_mkt)})
    mean_r = mean_rows(joints)
    pareto["mean_over_captures"] = pareto_table(mean_r, b_mkt, se_mkt)
    lam_rec = pareto["mean_over_captures"]["recommended_lambda"]
    rec_i = 1 if n > 1 else 0                         # the 11:00 capture, same time as the served fit
    rec = dict(joints[rec_i]["fits"][f"{lam_rec:g}"])
    rec["capture"] = joints[rec_i]["capture"]
    rec["lambda"] = lam_rec
    pure = dict(joints[rec_i]["fits"]["0"])
    pure["capture"] = joints[rec_i]["capture"]
    served_rows = [cs.evaluate(served_theta()) for cs in css]
    served = {"theta": served_theta().tolist(), "params": served_params(),
              "per_capture": [{"capture": cs.path.name, **ev} for cs, ev in zip(css, served_rows)],
              "rmse_volpts_capture_mean": float(np.mean([e["rmse_volpts"] for e in served_rows])),
              "b_expiries_capture_mean": float(np.mean([e["b_expiries"] for e in served_rows])),
              "chi2_capture_mean": float(np.mean([e["chi2"] for e in served_rows]))}
    boot = summarise_bootstrap(boots) if boots else {"n": 0}
    lam_rows = {f"{lam:g}": {"per_capture": [j["fits"].get(f"{lam:g}") for j in joints]} for lam in LAMBDAS}
    licence = load_json(work / "skew_check_cells.json", {})
    # licence summary: a cell passes when the map's psi is within 5% (rms,
    # relative) of MC and its exponent within 0.035 (about one market SE)
    lic_rows = []
    for key, v in licence.get("cells", {}).items():
        c = v["compare"]
        lic_rows.append({"cell": key, "H": v["H"], "eta": v["eta"], "rho": v["rho"], "xi": v["xi"],
                         "rmse_profile": v["rmse_profile"], "rel_rms": c["diff_rel_rms"],
                         "diff_rms": c["diff_rms"], "b_map": c["b_map"], "b_mc": c["b_mc"],
                         "se_b_mc": c["se_b_mc"], "b_profile_market_stencil": v["b_profile"],
                         "chi2_profile": v["chi2_profile"], "chi2_mc": v["chi2_mc"],
                         "mean_abs_psi_mc": v["psi_mc_mean_abs"],
                         "passes": bool(c["diff_rel_rms"] < 0.05 and abs(c["b_map"] - c["b_mc"]) < 0.035)})
    if skew_check.get("compare"):
        c = skew_check["compare"]
        lic_rows.append({"cell": "served", "H": skew_check["theta"][2], "eta": skew_check["theta"][0],
                         "rho": skew_check["theta"][1], "xi": skew_check["theta"][3],
                         "rmse_profile": skew_check["map_eval"]["rmse_volpts"], "rel_rms": c["diff_rel_rms"],
                         "diff_rms": c["diff_rms"], "b_map": c["b_map"], "b_mc": c["b_mc"],
                         "se_b_mc": c["se_b_mc"], "b_profile_market_stencil": skew_check["map_eval"]["b_expiries"],
                         "chi2_profile": skew_check["map_eval"]["chi2"], "chi2_mc": None,
                         "mean_abs_psi_mc": float(np.mean(np.abs(c["psi_mc"]))),
                         "passes": bool(c["diff_rel_rms"] < 0.05 and abs(c["b_map"] - c["b_mc"]) < 0.035)})
    lic_rows.sort(key=lambda r: r["eta"])
    passing = [r["eta"] for r in lic_rows if r["passes"]]
    failing = [r["eta"] for r in lic_rows if not r["passes"]]
    licence_summary = {"rows": lic_rows, "eta_min_passing": (min(passing) if passing else None),
                       "eta_max_failing": (max(failing) if failing else None),
                       "rule": "rel rms |psi_map - psi_mc| / |psi_mc| < 0.05 and |b_map - b_mc| < 0.035"}
    prof_summary["contour_bests"] = [{"capture": p_["capture"], **contour_bests(p_, b_mkt, se_mkt)}
                                     for p_ in profiles]
    interior = {"per_capture": [{"capture": j["capture"], **(interior_candidate(j["fits"]) or {"lambda": None})}
                                for j in joints],
                "mean_rows": interior_candidate({f"{r['lambda']:g}": r for r in mean_r})}
    residuals = {"per_capture": []}
    for cs, j in zip(css, joints):
        # the same "interior eta" alternative the MC stage validates: the
        # smallest-lambda interior optimum, or the next one when that is the
        # recommended point itself
        ic = interior_candidate(j["fits"])
        if ic is not None and ic["lambda"] == lam_rec:
            ic = interior_candidate(j["fits"], lambdas=[l for l in LAMBDAS if l > lam_rec])
        residuals["per_capture"].append({
            "capture": cs.path.name,
            "pure": residual_bands(cs, j["fits"]["0"]["theta"]),
            "recommended": residual_bands(cs, j["fits"][f"{lam_rec:g}"]["theta"]),
            "served": residual_bands(cs, served_theta()),
            "interior": residual_bands(cs, ic["theta"]) if ic else None})
    polish_stats = {"n_polishes": 0, "n_bounded_worse_than_start": 0, "winners": {}}
    for j in joints:
        for f in j["fits"].values():
            for pz in f.get("polished", []):
                polish_stats["n_polishes"] += 1
                polish_stats["n_bounded_worse_than_start"] += int(bool(pz.get("bounded_worse_than_start")))
                m = pz.get("method", "bounded(legacy)")
                polish_stats["winners"][m] = polish_stats["winners"].get(m, 0) + 1

    payload = {
        "generated": now(),
        "protocol": {
            "captures": [cs.info() for cs in css], "H_grid": H_GRID.tolist(),
            "eta_grid": ETA_GRID.tolist(), "lambdas": list(LAMBDAS),
            "h_rule": "max(0.25 atm_iv_market sqrt(tau), 0.002) (profile/joint); "
                      "max(0.25 sqrt(xi) sqrt(T), 0.002) for the map-vs-MC comparison",
            "stencil_centre": "the forward F = fwd_pv e^{r tau} (map log-moneyness k0 = r tau), "
                              "as in market_skew_from_quotes and price_stencil",
            "inner_optimiser": "Powell over (rho, ln xi), warm-started along a snake path",
            "joint_optimiser": "DE (maxiter 60, popsize 16, sobol, seed 7) + Powell, as MapCalibrator.fit, "
                               "plus Powell from the previous lambda's optimum, the profile's best cell "
                               "and the served point; best polish wins",
            "objective": "MapCalibrator.loss (IV-space Huber, delta 2 vp, x1e4) + lambda * "
                         "mean_e ((psi_model - psi_market)/SE_market)^2",
            "exponent_window_expiries": "power law over the capture's listed expiries (2-10 trading days), "
                                        "unweighted in log space for map psi",
            "mc": {k: mc.get(k) for k in ("n_paths_ladder", "n_reps_ladder", "n_paths_smile", "ladder_T_days")},
            "bounds": {k: list(v) for k, v in BOUNDS.items()}, "pin_frac": PIN_FRAC,
            "torch_threads": torch.get_num_threads(),
        },
        "market": {**market, "rows_3_captures": rows3, "pooled_3_captures": pooled3},
        "skew_check": skew_check,
        "profile": prof_summary,
        "joint": {"per_capture": joints, "mean_rows": mean_r, "by_lambda": lam_rows},
        "pareto": pareto,
        "served": served,
        "pure_smile": pure,
        "recommended": rec,
        "bootstrap": boot,
        "eta_extension": load_json(work / "eta_extend.json", None),
        "licence_cells": licence,
        "licence_summary": licence_summary,
        "licence_eta_min": licence_summary["eta_min_passing"],
        "interior_eta": interior,
        "residuals": residuals,
        "polish_stats": polish_stats,
        "mc": mc,
        "cost_benefit": {
            "rmse_cost_vs_pure_smile_volpts": rec["rmse_volpts"] - pure["rmse_volpts"],
            "rmse_cost_vs_pure_smile_mean_3_captures": (
                mean_r[[r["lambda"] for r in mean_r].index(lam_rec)]["rmse_volpts"] - mean_r[0]["rmse_volpts"]),
            "b_expiries_pure": pure["b_expiries"], "b_expiries_joint": rec["b_expiries"],
            "chi2_pure": pure["chi2"], "chi2_joint": rec["chi2"],
            "eta_interior_pure": pure["eta_interior"], "eta_interior_joint": rec["eta_interior"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "joint_skew_refit.json", payload)
    make_figure(payload, out_dir / "joint_skew_refit.png")

    # artifact in the rough_calibration.json schema
    base = json.loads((ARTIFACTS / "rough_calibration.json").read_text(encoding="utf-8"))
    cs = css[rec_i]
    th = rec["theta"]
    fits_l = mc.get("joint_ladder_fit", {})
    art = dict(base)
    art.update({
        "as_of": cs.info()["pricing_time"], "run_at": now(), "spot": cs.meta.get("spot"),
        "rate": cs.rate, "rate_source": cs.meta.get("rate_source", base.get("rate_source")),
        "eta": round(th[0], 4), "rho": round(th[1], 4), "H": round(th[2], 4),
        "xi": round(th[3], 6), "sqrt_xi": round(math.sqrt(th[3]), 4),
        "n_quotes": len(cs.quotes_all), "n_scored": len(cs.cal.quotes),
        "n_unpriceable": mc.get("joint_smile", {}).get("n_unpriceable", 0),
        "expiries": [r["expiry"] for r in cs.market_rows],
        "forwards": {e: {"F": round(q0.fwd_pv * math.exp(cs.rate * q0.tau), 4), "source": "capture",
                         "fwd_pv": q0.fwd_pv, "tau": q0.tau}
                     for e in [r["expiry"] for r in cs.market_rows]
                     for q0 in [next(q for q in cs.cal.quotes if q.expiry == e)]},
        "iv_rmse_volpts": round(rec["rmse_volpts"], 3),
        "iv_rmse_volpts_mc": (round(mc["joint_smile"]["rmse_mc_volpts"], 3) if "joint_smile" in mc else None),
        "iv_rmse_volpts_priceable_only": (round(float(np.mean(mc["joint_smile"]["rmse_mc_priceable_only_by_seed"])), 3)
                                          if "joint_smile" in mc else None),
        "objective": "map IV Huber (MapCalibrator.loss) + lambda * skew chi2 per expiry",
        "objective_value": rec["J"], "objective_mc_sd": 0.0, "objective_noise_reps": 0,
        "search_paths": None, "polish_paths": None, "final_paths": None,
        "pricing_engine": "pricing_map.pt (deterministic surrogate); MC validation in docs/joint_skew_refit.json",
        "quote_source": cs.meta.get("quote_source"),
        # half_spread_iv already arrives in vol points: calibrate.py:633 divides
        # the half-spread by vega and multiplies by 100 before it is stored, and
        # the reference writer at calibrate.py:1302 records it with no further
        # scaling. Scaling again here put this artifact 100x off the one it is
        # meant to be comparable with, on the same 585 quotes.
        "median_half_spread_iv_volpts": round(float(np.median([q.half_spread_iv for q in cs.cal.quotes
                                                              if np.isfinite(q.half_spread_iv)])), 4),
        "kernel": base.get("kernel"),
        "accepted": None,
        "reject_reasons": [],
        "note": "ANALYSIS ARTIFACT from scripts/joint_skew_refit.py, not the served calibration: "
                "parameters minimise smile loss + lambda * ATM-skew chi2 (docs/joint_skew_refit.md). "
                "accepted is null because calibrate.quality_gate was not run on it and it is not "
                "wired into the pricing service or the 0DTE surrogate.",
        "skew_term_structure": {
            "lambda": lam_rec,
            "exponent_ladder_mc": fits_l.get("b"), "exponent_ladder_mc_se": fits_l.get("se_b"),
            "exponent_ladder_window_days": SHORT_MAX_DAYS,
            "exponent_expiries_map": rec["b_expiries"],
            "exponent_expiries_mc": mc.get("joint_b_expiries_mc"),
            "market_exponent": b_mkt, "market_exponent_se": se_mkt,
            "served_exponent_ladder_mc": mc.get("served_ladder_fit", {}).get("b"),
            "chi2": rec["chi2"], "chi2_mc": mc.get("joint_chi2_mc"),
            "chi2_pure_smile": pure["chi2"],
            "eta_interior": rec["eta_interior"],
            "eta_bounds": list(BOUNDS["eta"]), "pin_frac": PIN_FRAC,
            "rmse_cost_vs_pure_smile_volpts": payload["cost_benefit"]["rmse_cost_vs_pure_smile_volpts"],
            "pure_smile_theta": pure["theta"],
            "bootstrap": {k: boot.get(k) for k in ("n", "eta", "rho", "H", "xi", "frac_eta_interior")},
            "interior_eta_alternative": interior["per_capture"][rec_i],
            "interior_eta_alternative_mc": {
                "exponent_ladder_mc": mc.get("interior_ladder_fit", {}).get("b"),
                "exponent_ladder_mc_se": mc.get("interior_ladder_fit", {}).get("se_b"),
                "rmse_mc_volpts": mc.get("interior_smile", {}).get("rmse_mc_volpts")},
        },
    })
    for k in ("xi_curve", "skipped_exdiv", "ex_dividend_check", "session_date",
              "snapshot_spread_min", "day_count", "min_tau_hours", "rmse_over_half_spread"):
        art.setdefault(k, base.get(k))
    art["rmse_over_half_spread"] = (round(rec["rmse_volpts"] / art["median_half_spread_iv_volpts"], 3)
                                    if art["median_half_spread_iv_volpts"] else None)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(art, indent=2), encoding="utf-8")
    log(f"[report] wrote {out_dir / 'joint_skew_refit.json'}, {out_dir / 'joint_skew_refit.png'}, {artifact}")
    log(f"[report] recommended lambda {lam_rec:g}: theta {np.round(th, 4).tolist()} rmse {rec['rmse_volpts']:.3f} "
        f"(pure {pure['rmse_volpts']:.3f}) b_exp {rec['b_expiries']:+.3f} chi2 {rec['chi2']:.2f} "
        f"eta_interior {rec['eta_interior']}")
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("skew-check", "profile", "joint", "eta-extend", "bootstrap",
                                     "licence", "mc", "report", "all"))
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--captures", nargs="*", default=None)
    p.add_argument("--n-captures", type=int, default=3)
    p.add_argument("--capture-index", type=int, nargs="*", default=None,
                   help="which of the selected captures this process handles")
    p.add_argument("--lambda", dest="lam", type=float, default=None)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--rep-offset", type=int, default=0)
    p.add_argument("--paths", type=int, default=200_000)
    p.add_argument("--mc-reps", type=int, default=4)
    p.add_argument("--smile-paths", type=int, default=400_000)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--grid-n", type=int, default=None, help="smoke runs: coarser (H, eta) grid")
    p.add_argument("--eta-max", type=float, default=None,
                   help="eta-extend: ceiling for the extended strip and bounds "
                        "(default: the committed map's own box top)")
    p.add_argument("--eta-step", type=float, default=0.5, help="eta-extend: strip spacing")
    p.add_argument("--lambdas", nargs="*", type=float, default=None, help="override the lambda grid")
    p.add_argument("--out-dir", type=Path, default=DOCS)
    p.add_argument("--artifact", type=Path, default=ARTIFACTS / "rough_calibration_skewjoint.json")
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    log = make_logger(args.log_file)
    caps = select_captures(args.n_captures, args.captures)
    idx = args.capture_index if args.capture_index is not None else list(range(len(caps)))
    log(f"[{args.stage}] captures: {[c.name for c in caps]}; work {work}; threads {torch.get_num_threads()}")
    t0 = time.perf_counter()
    if args.stage in ("skew-check", "all"):
        stage_skew_check(work, caps, args.paths, args.mc_reps, log)
    H_grid, eta_grid = H_GRID, ETA_GRID
    if args.grid_n:
        H_grid = np.linspace(H_GRID[0], H_GRID[-1], args.grid_n)
        eta_grid = np.linspace(ETA_GRID[0], ETA_GRID[-1], args.grid_n)
    lambdas = tuple(args.lambdas) if args.lambdas is not None else LAMBDAS
    if args.stage in ("profile", "all"):
        stage_profile(work, caps, idx, log, H_grid, eta_grid)
    if args.stage in ("joint", "all"):
        stage_joint(work, caps, idx, log, lambdas)
    if args.stage in ("eta-extend", "all"):
        stage_eta_extend(work, caps, idx, log, eta_max=args.eta_max, H_grid=H_grid,
                         step=args.eta_step, mc_paths=args.paths, mc_reps=args.mc_reps,
                         mc_smile_paths=args.smile_paths)
    if args.stage in ("bootstrap", "all"):
        stage_bootstrap(work, caps, idx[0] if args.stage == "bootstrap" else 1, args.lam,
                        args.reps, args.rep_offset, log)
    if args.stage in ("licence", "all"):
        stage_licence(work, caps, args.paths, args.mc_reps, log)
    if args.stage in ("mc", "all"):
        stage_mc(work, caps, idx[0] if args.stage == "mc" else 1, args.lam, args.paths,
                 args.mc_reps, args.smile_paths, log)
    if args.stage in ("report", "all"):
        stage_report(work, caps, args.out_dir, args.artifact, log)
    log(f"[{args.stage}] finished in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
