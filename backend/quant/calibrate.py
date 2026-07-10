"""Live market calibration of the rough Bergomi 0DTE engine.

Fits (eta, rho, H) — plus the forward-variance nuisance parameter xi — to
the live short-dated (<= 3 DTE) SPY option smile, using the same PyTorch
Monte Carlo teacher (rough_vol.rough_bergomi_mc) that generates the 0DTE
surrogate's training data.

Methodology
-----------
Objective  min  sum_i Huber( (P_model_i - P_mid_i) / vega_i ; delta )

* Vega-weighting: to first order dP = vega * d(sigma_iv), so vega-normalized
  price errors ARE implied-vol errors. This keeps every strike on the smile
  equally informative (raw price MSE is dominated by ATM options and fits
  the skew poorly) without inverting Black-Scholes on a *noisy MC price*
  at every optimizer step.
* Huber (delta = 2 vol points): stale or crossed quotes get linear, not
  quadratic, influence.
* Common random numbers: a frozen seed makes the MC objective a
  deterministic, near-smooth function of the parameters — the optimizer
  never chases resampling noise.
* Joint fit over all expiries <= 3 DTE: the Hurst index H is identified by
  the *term structure* of the skew (ATM skew ~ rho*eta*tau^{H-1/2}), not by
  any single smile.
* Optimization: differential evolution (global, bounded, derivative-free)
  followed by a Powell polish, both on the CRN objective.

Data hygiene: OTM options only (puts mapped to synthetic calls via exact
European parity so all quotes live on one call surface), bid > $0.02,
relative spread <= 40%, volume or open interest floor, mids below intrinsic
dropped, Yahoo IVs discarded in favor of our own bisection inversion.

Usage (from the repo root):
    python -m backend.quant.calibrate                    # calibrate SPY live
    python -m backend.quant.calibrate --ticker QQQ --max-dte 2
    python -m backend.quant.calibrate --retrain          # + regenerate the
                                                         # 0DTE dataset and
                                                         # retrain the
                                                         # surrogate ensemble

Writes artifacts/rough_calibration.json, which dataset_0dte.py picks up
automatically on the next dataset build.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

# Windows consoles often default to cp1252, which cannot encode the rules
# and Greek letters in the log; force UTF-8 rather than crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from .rough_vol import rough_bergomi_mc

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
CAL_FILE = ARTIFACTS / "rough_calibration.json"
BOUNDS = {"eta": (0.5, 4.0), "rho": (-1.0, 0.0), "H": (0.01, 0.5),
          "xi": (0.05 ** 2, 1.2 ** 2)}
HUBER_DELTA = 0.02          # 2 vol points
CRN_SEED = 1234
NY = ZoneInfo("America/New_York")

# ── terminal aesthetics ────────────────────────────────────────────────
CY, MG, VI, GN, RD, DIM, BOLD, RS = ("\x1b[38;5;51m", "\x1b[38;5;205m",
                                     "\x1b[38;5;141m", "\x1b[38;5;84m",
                                     "\x1b[38;5;203m", "\x1b[2m",
                                     "\x1b[1m", "\x1b[0m")


def rule(title: str = "") -> None:
    pad = f"═══ {BOLD}{title}{RS} " if title else ""
    print(f"{DIM}{pad}{'═' * max(8, 74 - len(title))}{RS}")


# ── Black-Scholes helpers (calls only — puts are parity-mapped) ────────
def bs_call(spot, strike, tau, sigma, rate):
    sd = max(sigma, 1e-8) * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    d2 = d1 - sd
    return spot * norm.cdf(d1) - strike * math.exp(-rate * tau) * norm.cdf(d2)


def bs_vega(spot, strike, tau, sigma, rate):
    sd = max(sigma, 1e-8) * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / sd
    return spot * norm.pdf(d1) * math.sqrt(tau)


def implied_vol(price, spot, strike, tau, rate,
                lo: float = 1e-3, hi: float = 5.0) -> float | None:
    """Bisection BS inversion; None if the quote is out of no-arb range."""
    if price <= max(spot - strike * math.exp(-rate * tau), 0.0) + 1e-10:
        return None
    if price >= spot:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_call(spot, strike, tau, mid, rate) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── market data ingestion ──────────────────────────────────────────────
@dataclass
class Quote:
    tau: float          # year fraction (ACT/365, intraday-accurate)
    strike: float
    mid_call: float     # call-equivalent mid (puts parity-mapped)
    iv: float           # our own BS inversion of the mid
    vega: float
    kind: str           # original instrument: "C" or "P"
    expiry: str


def fetch_calibration_set(ticker: str, max_dte: int,
                          min_tau_hours: float = 1.0
                          ) -> tuple[float, float, list[Quote]]:
    import yfinance as yf

    from .market_data import _fetch_risk_free

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["last_price"])
    rate, rate_src = _fetch_risk_free(yf)
    now = datetime.now(tz=NY)

    quotes: list[Quote] = []
    kept_expiries = []
    used_last_trade = [False]
    for expiry in tk.options:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d") \
            .replace(hour=16, minute=0, tzinfo=NY)
        tau = (exp_dt - now).total_seconds() / (365.0 * 24 * 3600)
        dte = (exp_dt.date() - now.date()).days
        if tau * 365 * 24 < min_tau_hours or dte > max_dte:
            continue
        kept_expiries.append(expiry)

        chain = tk.option_chain(expiry)
        fwd = spot * math.exp(rate * tau)
        for df, kind in ((chain.calls, "C"), (chain.puts, "P")):
            for row in df.itertuples():
                def _safe_float(v):
                    try:
                        f = float(v)
                        return 0.0 if math.isnan(f) else f
                    except (TypeError, ValueError):
                        return 0.0

                k = float(row.strike)
                bid = _safe_float(row.bid)
                ask = _safe_float(row.ask)
                last = _safe_float(row.lastPrice)
                vol = _safe_float(row.volume)
                oi = _safe_float(row.openInterest)
                # OTM only — the liquid half of each smile wing
                if kind == "C" and k <= fwd:
                    continue
                if kind == "P" and k >= fwd:
                    continue
                if k / spot < 0.9 or k / spot > 1.1:
                    continue                     # surrogate/liquidity band

                # Quote selection. Live market: bid/ask mid with spread and
                # liquidity filters. Closed market (Yahoo zeroes the book):
                # fall back to the last session trade, with a stricter
                # volume floor since spreads cannot be checked. The run is
                # flagged as stale either way.
                mid = None
                if bid > 0.02 and ask >= bid:
                    cand = 0.5 * (bid + ask)
                    if (ask - bid) / cand <= 0.40 and (vol >= 10 or oi >= 100):
                        mid = cand
                elif last > 0.05 and vol >= 50:
                    mid = last
                    used_last_trade[0] = True
                if mid is None:
                    continue
                # parity-map puts onto the call surface (exact, European)
                mid_call = mid if kind == "C" else \
                    mid + spot - k * math.exp(-rate * tau)
                iv = implied_vol(mid_call, spot, k, tau, rate)
                if iv is None:
                    continue                     # stale/arb quote
                quotes.append(Quote(tau=tau, strike=k, mid_call=mid_call,
                                    iv=iv, vega=bs_vega(spot, k, tau, iv,
                                                        rate),
                                    kind=kind, expiry=expiry))

    print(f"  spot {BOLD}${spot:,.2f}{RS}  ·  r {rate:.2%} ({rate_src})  ·  "
          f"expiries kept: {', '.join(kept_expiries) or 'none'}")
    if used_last_trade[0]:
        print(f"  {RD}market closed: using last session trades as mids "
              f"(no live book; treat the fit as indicative){RS}")
    return spot, rate, quotes, used_last_trade[0]


# ── calibration engine ─────────────────────────────────────────────────
class Calibrator:
    def __init__(self, spot: float, rate: float, quotes: list[Quote],
                 n_paths: int):
        self.spot, self.rate = spot, rate
        self.quotes = quotes
        self.n_paths = n_paths
        self.n_evals = 0
        # group quotes by expiry so each MC call shares one maturity
        self.groups: dict[str, list[Quote]] = {}
        for q in quotes:
            self.groups.setdefault(q.expiry, []).append(q)

    def model_prices(self, eta: float, rho: float, H: float, xi: float,
                     n_paths: int | None = None) -> np.ndarray:
        out = []
        for qs in self.groups.values():
            b = len(qs)
            prices = rough_bergomi_mc(
                torch.full((b,), self.spot),
                torch.tensor([q.strike for q in qs], dtype=torch.float32),
                torch.full((b,), qs[0].tau),
                torch.full((b,), xi),
                torch.full((b,), eta), torch.full((b,), rho),
                torch.full((b,), self.rate),
                n_paths=n_paths or self.n_paths, n_steps=50, H=H,
                seed=CRN_SEED)
            out.append(prices.numpy())
        return np.concatenate(out)

    def loss(self, theta: np.ndarray) -> float:
        eta, rho, H, xi = map(float, theta)
        self.n_evals += 1
        model = self.model_prices(eta, rho, H, xi)
        mids = np.array([q.mid_call for q in self.quotes])
        vegas = np.array([q.vega for q in self.quotes])
        e = (model - mids) / np.maximum(vegas, 1e-4)   # ≈ IV error
        d = HUBER_DELTA
        hub = np.where(np.abs(e) <= d, 0.5 * e ** 2,
                       d * (np.abs(e) - 0.5 * d))
        return float(hub.mean()) * 1e4                 # scaled for optimizer

    def rmse_volpts(self, theta: np.ndarray, n_paths: int) -> float:
        model = self.model_prices(*map(float, theta), n_paths=n_paths)
        mids = np.array([q.mid_call for q in self.quotes])
        vegas = np.array([q.vega for q in self.quotes])
        return float(np.sqrt((((model - mids) / vegas) ** 2).mean()) * 100)


def calibrate(ticker: str, max_dte: int, search_paths: int,
              final_paths: int, seed: int = 7) -> dict:
    rule(f"ROUGH BERGOMI LIVE CALIBRATION · {ticker}")
    spot, rate, quotes, stale = fetch_calibration_set(ticker, max_dte)
    if len(quotes) < 8:
        raise SystemExit(f"{RD}only {len(quotes)} clean quotes survived "
                         f"filtering — market closed or chain illiquid; "
                         f"try --max-dte 5{RS}")
    n_c = sum(q.kind == "C" for q in quotes)
    taus = sorted({round(q.tau * 365, 2) for q in quotes})
    print(f"  {BOLD}{len(quotes)}{RS} clean quotes "
          f"({n_c} OTM calls / {len(quotes) - n_c} OTM puts) across "
          f"DTE {taus} calendar days")

    cal = Calibrator(spot, rate, quotes, n_paths=search_paths)
    bounds = [BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]

    rule("STAGE 1 · DIFFERENTIAL EVOLUTION (global search)")
    print(f"  {DIM}{'gen':>4} {'eta':>7} {'rho':>7} {'H':>7} "
          f"{'sqrt(xi)':>9} {'loss':>10}{RS}")
    gen = [0]
    t0 = time.perf_counter()

    def cb(xk, convergence=None):
        gen[0] += 1
        print(f"  {gen[0]:>4} {CY}{xk[0]:>7.3f}{RS} {MG}{xk[1]:>7.3f}{RS} "
              f"{VI}{xk[2]:>7.3f}{RS} {math.sqrt(xk[3]):>8.1%} "
              f"{cal.loss(xk):>10.4f}", flush=True)

    de = differential_evolution(
        cal.loss, bounds, seed=seed, maxiter=12, popsize=6, tol=1e-3,
        mutation=(0.4, 0.9), recombination=0.8, polish=False, callback=cb,
        init="sobol", updating="deferred")

    rule("STAGE 2 · POWELL POLISH (local refinement)")
    res = minimize(cal.loss, de.x, method="Powell", bounds=bounds,
                   options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    eta, rho, H, xi = map(float, res.x)
    elapsed = time.perf_counter() - t0
    print(f"  converged: {CY}eta={eta:.3f}{RS}  {MG}rho={rho:.3f}{RS}  "
          f"{VI}H={H:.3f}{RS}  sqrt(xi)={math.sqrt(xi):.1%}  "
          f"({cal.n_evals} MC objective evals · {elapsed:.0f}s)")

    rule("FIT QUALITY · SMILE (high-precision repricing)")
    model = cal.model_prices(eta, rho, H, xi, n_paths=final_paths)
    print(f"  {DIM}{'expiry':>11} {'K':>8} {'type':>4} {'mkt mid':>9} "
          f"{'model':>9} {'mkt IV':>7} {'mdl IV':>7} {'err':>8}{RS}")
    iv_errs = []
    for q, p in zip(quotes, model):
        iv_m = implied_vol(float(p), spot, q.strike, q.tau, rate)
        iv_err = (iv_m - q.iv) * 100 if iv_m else float("nan")
        iv_errs.append(iv_err)
        col = GN if abs(iv_err) < 1.5 else (RD if abs(iv_err) > 4 else "")
        print(f"  {q.expiry:>11} {q.strike:>8.0f} {q.kind:>4} "
              f"{q.mid_call:>9.2f} {p:>9.2f} {q.iv:>6.1%} "
              f"{(iv_m or float('nan')):>6.1%} "
              f"{col}{iv_err:>+7.2f}vp{RS}")
    iv_arr = np.array([e for e in iv_errs if math.isfinite(e)])
    rmse = float(np.sqrt((iv_arr ** 2).mean()))
    print(f"\n  {BOLD}implied-vol RMSE across the smile: "
          f"{(GN if rmse < 2 else '')}{rmse:.2f} vol points{RS} "
          f"({len(iv_arr)} quotes, {final_paths:,} paths)")

    result = {
        "ticker": ticker, "as_of": datetime.now(tz=NY).isoformat(),
        "spot": spot, "rate": rate,
        "eta": round(eta, 4), "rho": round(rho, 4), "H": round(H, 4),
        "xi": round(xi, 6), "sqrt_xi": round(math.sqrt(xi), 4),
        "n_quotes": len(quotes), "expiries": sorted(cal.groups),
        "iv_rmse_volpts": round(rmse, 3),
        "search_paths": search_paths, "final_paths": final_paths,
        "objective": "vega-weighted price Huber (CRN)",
        "quote_source": "last_trade_market_closed" if stale else "live_mid",
        # Quality gate: downstream retraining only adopts a calibration that
        # fits reasonably and did not pin a parameter at its bound (a bound
        # hit signals misspecification or an ill-conditioned fit).
        "accepted": bool(
            rmse < 3.0
            and BOUNDS["eta"][0] + 1e-3 < eta < BOUNDS["eta"][1] - 1e-3
            and BOUNDS["rho"][0] + 1e-3 < rho < BOUNDS["rho"][1] - 1e-3
            and BOUNDS["H"][0] + 1e-3 < H < BOUNDS["H"][1] - 1e-3),
    }
    CAL_FILE.write_text(json.dumps(result, indent=2))
    print(f"  {DIM}saved -> {CAL_FILE}{RS}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--max-dte", type=int, default=3)
    p.add_argument("--search-paths", type=int, default=8_000)
    p.add_argument("--final-paths", type=int, default=64_000)
    p.add_argument("--retrain", action="store_true",
                   help="after calibrating, regenerate the 0DTE dataset and "
                        "retrain the surrogate ensemble on the new dynamics")
    args = p.parse_args()

    result = calibrate(args.ticker, args.max_dte,
                       args.search_paths, args.final_paths)

    if args.retrain:
        rule("REGIME SYNC · dataset regeneration + surrogate retrain")
        from .dataset_0dte import generate_0dte_dataset
        generate_0dte_dataset(eta=result["eta"], rho=result["rho"],
                              H=result["H"])
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "backend.quant.train_0dte",
                        "--ensemble", "5", "--epochs", "500"], check=True,
                       cwd=Path(__file__).resolve().parents[2])
        print(f"{GN}0DTE surrogate resynced to calibrated dynamics — "
              f"restart the API server to serve it.{RS}")


if __name__ == "__main__":
    main()
