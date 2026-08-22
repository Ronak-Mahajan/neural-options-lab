"""Calibrate the rough Bergomi 0DTE engine to a live Deribit BTC option surface.

Why a second calibration path
-----------------------------
`calibrate.py` fits SPY through yfinance. That works, but it is only usable for
the six and three quarter hours a day the US equity option market is open, and
it carries three approximations this venue simply does not have:

* SPY is American-exercise; Deribit options are European, which is exactly what
  `rough_vol.rough_bergomi_mc` prices. No early-exercise premium to wave away.
* SPY pays a 1.1-1.3% dividend. The put-call parity map is applied only to puts,
  so a dividend error is one-sided - a pure skew distortion, and rho is the
  parameter that absorbs skew. An audit measured 0.37/0.28/0.31 vol points of
  bias at 0.5/1.5/3.0 DTE against a total fit residual of 1.969, i.e. 15-20% of
  the residual was a systematic shift of the put wing. Deribit BTC options have
  no dividend and the forward is published per expiry.
* yfinance serves mids; outside market hours it serves last-session prints stamped
  with the current time. Deribit publishes live two-sided books 24/7.

So this path is both always-available and better-posed. The trade is that it
calibrates the model to BTC volatility dynamics, not equity-index dynamics -
which is why the market is recorded in the calibration file, in the dataset and
in the served checkpoint, and why the two calibrations live in separate files
that cannot overwrite one another.

What is reused
--------------
Everything that matters: `Calibrator` (the CRN vega-weighted Huber objective),
the three-stage differential-evolution + Powell + re-polish schedule, and
`quality_gate`. Only the market snapshot differs. That is deliberate - two fits
that share an objective and a gate are comparable; two that do not are not.

Domain
------
Quotes are restricted to the 0DTE surrogate's trained box, because a calibration
outside it configures a model that will never be asked to price there:
moneyness in [0.85, 1.15], maturity in [1/252, 12/252] years (1.45 to 17.4 days),
sigma in [0.05, 0.80]. On the committed snapshot that admits 4 expiries, which
matters: H is identified by the TERM STRUCTURE of the skew, not by any single
smile, so a single-expiry fit cannot identify it at all.

Deribit reports `interest_rate = 0` and publishes a per-expiry forward that
already carries the carry, so the Monte Carlo starts at F with r = 0 and model
and market agree on E[S_T] by construction.

Usage
-----
    python -m backend.quant.calibrate_deribit                 # live fetch
    python -m backend.quant.calibrate_deribit --offline       # committed fixture
    python -m backend.quant.calibrate_deribit --retrain       # + resync surrogate
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .calibrate import (ARTIFACTS, BOUNDS, CRN_SEED, PIN_FRAC, REPORT_SEED,
                        Calibrator, Quote, quality_gate)
from .dataset_0dte import KERNEL_ID
from .deribit import DeribitClient, DeribitError, latest_snapshot
from .surface import build_surface, implied_vol

MARKET = "BTC-DERIBIT"
CAL_FILE = ARTIFACTS / "rough_calibration_btc.json"

#: The 0DTE surrogate's trained box (dataset_0dte.py, engine.py).
BOX_MONEYNESS = (0.85, 1.15)
BOX_MATURITY = (1.0 / 252.0, 12.0 / 252.0)
BOX_SIGMA = (0.05, 0.80)
MIN_QUOTES = 8
MIN_EXPIRIES = 2

#: BTC needs more vol-of-vol headroom than an equity index. The first live fit
#: ran eta to 3.936 against calibrate.BOUNDS' ceiling of 4.0 - 98.2% of the
#: range - which is a bound hit, not a fit. Crypto smiles are far more convex
#: than SPY's, so the equity-tuned ceiling is the wrong prior here. Widening it
#: is only legitimate if the fit then lands INTERIOR; if it runs to the new
#: ceiling too, that is model misspecification and the gate must say so rather
#: than the bound being widened again.
ETA_MAX_BTC = 8.0

#: A fit is only meaningful RELATIVE TO THE WIDTH OF THE MARKET it is fitting.
#: calibrate.MAX_RMSE_VOLPTS is an absolute 3.0 vol points, which is
#: market-independent and therefore blind: 3 vol points is a tight fit to a
#: market quoting 8 points wide and a bad one to a market quoting 1 point wide.
# The market-width criterion lives inside calibrate.quality_gate now (see the
# call in main()), and it is one-directional: the wide crypto book RAISES the
# RMSE ceiling above the 3 vp floor, a tight book cannot lower it. A local
# symmetric copy used to live here; it was the same mistake the shared gate
# corrected against live SPY data, kept alive in a second place.


def quotes_from_surface(surface) -> tuple[list[Quote], dict]:
    """Convert clean Deribit quotes into calibrate.Quote, OTM only.

    Puts are parity-mapped onto the call surface. Deribit's r is 0 and the
    forward is published per expiry, so the map is exactly C = P + (F - K) with
    no discounting and no dividend assumption.
    """
    out: list[Quote] = []
    spreads: list[float] = []
    drops = {"not_clean": 0, "moneyness": 0, "maturity": 0, "sigma": 0,
             "itm": 0, "no_iv": 0, "no_vega": 0}
    for q in surface.quotes:
        if not q.is_clean:
            drops["not_clean"] += 1
            continue
        m = q.strike / q.forward
        if not (BOX_MONEYNESS[0] <= m <= BOX_MONEYNESS[1]):
            drops["moneyness"] += 1
            continue
        if not (BOX_MATURITY[0] <= q.tenor <= BOX_MATURITY[1]):
            drops["maturity"] += 1
            continue
        if not q.is_otm:
            drops["itm"] += 1          # OTM carries the information; ITM is parity
            continue
        mid = q.mid_usd
        if not math.isfinite(mid) or mid <= 0.0:
            drops["no_iv"] += 1
            continue
        # parity onto the call surface (r = 0, F published)
        mid_call = mid if q.right == "call" else mid + (q.forward - q.strike)
        if mid_call <= 0.0:
            drops["no_iv"] += 1
            continue
        iv = implied_vol(mid_call, q.forward, q.strike, q.tenor, "call")
        if not math.isfinite(iv) or not (BOX_SIGMA[0] <= iv <= BOX_SIGMA[1]):
            drops["sigma"] += 1
            continue
        if not math.isfinite(q.vega) or q.vega <= 0.0:
            drops["no_vega"] += 1
            continue
        # The market's own width on exactly the quotes that enter the fit -
        # short-dated OTM books are wider than the surface median, so this must
        # be measured on the fitted subset, not on the whole chain.
        if math.isfinite(q.iv_spread):
            spreads.append(q.iv_spread * 100.0)
        out.append(Quote(tau=float(q.tenor), strike=float(q.strike),
                         mid_call=float(mid_call), iv=float(iv),
                         vega=float(q.vega),
                         kind="C" if q.right == "call" else "P",
                         expiry=q.expiry_iso, fwd_pv=float(q.forward)))
    finite = [x for x in spreads if math.isfinite(x)]
    drops["median_iv_spread_volpts"] = (float(np.median(finite))
                                       if finite else float("nan"))
    return out, drops


def fit(quotes: list[Quote], *, search_paths: int, polish_paths: int,
        final_paths: int, noise_reps: int, seed: int = CRN_SEED) -> dict:
    """The same three-stage schedule calibrate.py uses, on Deribit quotes."""
    cal = Calibrator(rate=0.0, quotes=quotes, n_paths=search_paths, seed=seed)
    eta_bounds = (BOUNDS["eta"][0], ETA_MAX_BTC)
    bounds = [eta_bounds, BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]

    t0 = time.perf_counter()
    print(f"stage 1  differential evolution @ {search_paths:,} paths ...")
    de = differential_evolution(cal.loss, bounds, seed=seed, maxiter=12,
                                popsize=6, tol=1e-3, mutation=(0.4, 0.9),
                                recombination=0.8, polish=False, init="sobol",
                                updating="deferred")
    print(f"  eta={de.x[0]:.3f} rho={de.x[1]:+.3f} H={de.x[2]:.3f} "
          f"sqrt(xi)={math.sqrt(de.x[3]):.1%}  loss {de.fun:.4f}")

    print(f"stage 2  Powell polish @ {search_paths:,} paths ...")
    res = minimize(cal.loss, de.x, method="Powell", bounds=bounds,
                   options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    print(f"  eta={res.x[0]:.3f} rho={res.x[1]:+.3f} H={res.x[2]:.3f} "
          f"sqrt(xi)={math.sqrt(res.x[3]):.1%}  loss {res.fun:.4f}")

    # H is not identifiable on the cheap objective, so re-polish at the
    # precision the fit is reported and gated at.
    print(f"stage 3  Powell re-polish @ {polish_paths:,} paths ...")
    cal.n_paths = polish_paths
    res3 = minimize(cal.loss, res.x, method="Powell", bounds=bounds,
                    options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    eta, rho, H, xi = map(float, res3.x)
    print(f"  eta={eta:.3f} rho={rho:+.3f} H={H:.3f} "
          f"sqrt(xi)={math.sqrt(xi):.1%}  loss {res3.fun:.4f}"
          f"  ({time.perf_counter() - t0:.1f}s)")

    # The objective's own Monte Carlo noise, so a reader can tell whether a
    # loss difference is signal or resampling.
    noise = [float(np.mean(np.abs(
        cal.model_prices(eta, rho, H, xi, n_paths=polish_paths,
                         seed=CRN_SEED + 1000 + k) - cal.mids) / cal.vegas))
        for k in range(max(noise_reps, 2))]
    noise_sd = float(statistics.stdev(noise))

    # Honest scoring at final_paths on an out-of-sample path set: every quote is
    # scored, unpriceable ones via the vega-linearised error rather than dropped.
    model = cal.model_prices(eta, rho, H, xi, n_paths=final_paths,
                             seed=REPORT_SEED)
    errs, n_unpriceable = [], 0
    for q, p in zip(cal.quotes, model):
        iv_model = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, "call")
        if math.isfinite(iv_model):
            errs.append((iv_model - q.iv) * 100.0)
        else:
            n_unpriceable += 1
            errs.append((float(p) - q.mid_call) / max(q.vega, 1e-4) * 100.0)
    err = np.array(errs)
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"eta": eta, "rho": rho, "H": H, "xi": xi, "rmse": rmse,
            "n_unpriceable": n_unpriceable, "objective": float(res3.fun),
            "objective_mc_sd": noise_sd, "seconds": time.perf_counter() - t0,
            "eta_bounds": eta_bounds}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true",
                   help="use the committed snapshot instead of fetching")
    p.add_argument("--currency", default="BTC")
    p.add_argument("--search-paths", type=int, default=8_000)
    p.add_argument("--polish-paths", type=int, default=64_000)
    p.add_argument("--final-paths", type=int, default=200_000)
    p.add_argument("--noise-reps", type=int, default=4)
    p.add_argument("--retrain", action="store_true",
                   help="regenerate the 0DTE dataset and retrain the surrogate")
    args = p.parse_args()

    try:
        if args.offline:
            snap = latest_snapshot()
            print(f"offline snapshot {snap.currency} @ {snap.captured_at_iso}")
        else:
            print(f"fetching live {args.currency} chain ...")
            snap = DeribitClient().snapshot(currency=args.currency)
    except DeribitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    surface = build_surface(snap)
    quotes, drops = quotes_from_surface(surface)
    expiries = sorted({q.expiry for q in quotes})
    print(f"index {snap.index_price:,.2f} | {len(surface.quotes)} raw | "
          f"{len(surface.clean())} clean | {len(quotes)} in the surrogate's box "
          f"across {len(expiries)} expiries")
    print(f"  dropped: {drops}")

    if len(quotes) < MIN_QUOTES or len(expiries) < MIN_EXPIRIES:
        print(f"ERROR: need >= {MIN_QUOTES} quotes across >= {MIN_EXPIRIES} "
              f"expiries; H is identified by the term structure of the skew, "
              f"so a single-expiry fit cannot identify it.", file=sys.stderr)
        return 1

    r = fit(quotes, search_paths=args.search_paths,
            polish_paths=args.polish_paths, final_paths=args.final_paths,
            noise_reps=args.noise_reps)

    # The shared gate keys eta on calibrate.BOUNDS; this path widened it, so
    # check eta against the bound actually used and let the gate handle the rest.
    # The spread criterion lives INSIDE the shared gate now, and it is
    # one-directional: the ceiling is max(3 vp, 1.5x median half-spread), so a
    # wide crypto book loosens it and a tight book cannot tighten it. This
    # call site used to re-implement the criterion symmetrically -- the exact
    # mistake the main gate corrected on live SPY data -- and it also passed
    # no n_quotes, which silently disabled the unpriceable-fraction check
    # (the gate skips it when n_quotes is 0).
    half_spread = drops.get("median_iv_spread_volpts", float("nan")) / 2.0
    accepted, reasons = quality_gate(
        rmse=r["rmse"], eta=2.0, rho=r["rho"], H=r["H"], xi=r["xi"],
        stale=False,                      # Deribit books are live 24/7
        n_unpriceable=r["n_unpriceable"], n_quotes=len(quotes),
        median_half_spread_iv=(half_spread if math.isfinite(half_spread)
                               and half_spread > 0 else None))

    lo_e, hi_e = r["eta_bounds"]
    tol_e = PIN_FRAC * (hi_e - lo_e)
    if not (lo_e + tol_e < r["eta"] < hi_e - tol_e):
        accepted = False
        reasons = list(reasons) + [
            f"eta = {r['eta']:.4f} is pinned at its bound [{lo_e}, {hi_e}] "
            f"({(r['eta'] - lo_e) / (hi_e - lo_e) * 100:.1f}% of range) - the "
            f"model cannot reach this surface without extreme vol-of-vol"]

    payload = {
        "market": MARKET, "currency": snap.currency,
        "as_of": snap.captured_at_iso,
        "venue": "deribit-public-v2",
        "index_price": snap.index_price,
        "eta": round(r["eta"], 4), "rho": round(r["rho"], 4),
        "H": round(r["H"], 4), "xi": round(r["xi"], 6),
        "sqrt_xi": round(math.sqrt(r["xi"]), 4),
        "n_quotes": len(quotes), "n_scored": len(quotes),
        "n_unpriceable": r["n_unpriceable"],
        "expiries": expiries,
        "iv_rmse_volpts": round(r["rmse"], 3),
        "objective_value": round(r["objective"], 6),
        "objective_mc_sd": round(r["objective_mc_sd"], 6),
        "search_paths": args.search_paths, "polish_paths": args.polish_paths,
        "final_paths": args.final_paths,
        "day_count": "ACT/365 from the venue clock",
        "quote_source": "live_two_sided_book",
        "rate": 0.0,
        "rate_note": "Deribit reports interest_rate=0; the per-expiry forward "
                     "carries the carry, and the MC starts at F.",
        "kernel": KERNEL_ID,
        "eta_bounds": list(r["eta_bounds"]),
        "median_iv_half_spread_volpts": round(half_spread, 4),
        "rmse_over_half_spread": (round(r["rmse"] / half_spread, 3)
                                  if math.isfinite(half_spread) and half_spread > 0
                                  else None),
        "accepted": accepted, "reject_reasons": reasons,
        "fit_seconds": round(r["seconds"], 1),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    CAL_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\niv RMSE {r['rmse']:.3f} vol points over {len(quotes)} quotes "
          f"({r['n_unpriceable']} unpriceable) | objective MC sd "
          f"{r['objective_mc_sd']:.4f}")
    print(f"wrote {CAL_FILE}")
    if accepted:
        print("ACCEPTED by the quality gate")
    else:
        print("REJECTED by the quality gate:")
        for why in reasons:
            print(f"  · {why}")

    if not args.retrain:
        return 0
    if not accepted:
        print("refusing to regenerate the dataset or retrain on a rejected fit")
        return 2

    from .dataset_0dte import generate_0dte_dataset
    generate_0dte_dataset(eta=r["eta"], rho=r["rho"], H=r["H"])
    import subprocess
    subprocess.run([sys.executable, "-m", "backend.quant.train_0dte",
                    "--ensemble", "5", "--epochs", "500"], check=True,
                   cwd=Path(__file__).resolve().parents[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
