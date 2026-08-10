"""Where does the rough-Bergomi fit fail, and is that failure bias or scatter?

A single RMSE says a fit is 1.2 vol points wrong. It does not say WHERE, or
whether the model is systematically unable to make a shape (bias) as opposed to
being buffeted by noise and wide quotes (scatter). Only the first is a
modelling problem. This decomposes the residual three ways:

    by raw log-moneyness ln(K/F)      -- where in strike the error lives
    by expiry                          -- where in maturity it lives
    by STANDARDISED moneyness          -- the two above are confounded, and
      d = ln(K/F) / (sigma * sqrt(tau))   this is what separates them

The third is the one that matters. A strike 6% below the forward is a routine
one-sigma strike at 11 days and a ~6-sigma strike at 2 days, so a "deep wing"
bucket in raw moneyness is largely just the near-dated expiries wearing a
different label. On the standardised scale a skew failure is comparable across
maturities and a level failure separates cleanly from it.

For each bucket the important column is |mean| / rmse. Near 1 the errors all
share a sign and the model cannot make that shape. Near 0 they cancel and the
bucket is noise.

WHAT THIS FOUND (live SPY, 2026-08-10, 679 quotes across 8 expiries, 2 to 11
days, fit RMSE 1.24 vp). Two INDEPENDENT failures:

  (a) a far-left-tail skew failure, at every maturity.
      At d in [-3, -2] the model sits 2.12 vp BELOW the market with
      |mean|/rmse = 0.86 -- almost pure bias -- and it is negative at every
      single expiry (2d -1.84, 4d -3.19, 7d -0.75, 11d -1.51). The call wing is
      fine (-0.11 vp). Single-factor rough Bergomi at rho = -0.39 cannot make a
      left tail this fat. This is genuine misspecification and it is OPEN.

  (b) a short-maturity LEVEL failure, which was an artefact of parameterising
      the forward variance as one number.
      At MATCHED standardised moneyness the <=4 day expiries sat below the rest
      in every band (-0.55, -0.62, -0.71, -0.80, -1.25). Uniform across the
      smile means level, not skew -- and one scalar xi cannot give the 2-4 day
      expiries more variance than the 7-11 day ones, so it splits the
      difference. Holding (eta, rho, H) fixed and swapping scalar xi for a
      per-expiry curve cut the objective 0.500 -> 0.378 and the RMSE
      1.098 -> 0.886, and the 4-day expiry's mean error from -1.69 to -0.57 vp.

Run it after any calibration change. If (a) ever moves, that is a real result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from backend.quant.calibrate import (BOUNDS, CAL_FILE, Calibrator,
                                     MIN_TAU_HOURS, fetch_calibration_set,
                                     implied_vol, iv_fit_report)

#: Bands of standardised moneyness d = ln(K/F)/(sigma*sqrt(tau)).
D_BANDS = [(-3.0, -2.0), (-2.0, -1.0), (-1.0, -0.25), (-0.25, 0.25),
           (0.25, 1.0), (1.0, 3.0)]
#: Bands of raw log-moneyness, for the confounded view the standardised one
#: is meant to replace.
K_BANDS = [(-0.12, -0.05), (-0.05, -0.025), (-0.025, -0.005), (-0.005, 0.005),
           (0.005, 0.025), (0.025, 0.09)]
NEAR_DATED_DAYS = 4.5
MIN_BUCKET = 4


def _residuals(cal: Calibrator, rate: float, model: np.ndarray) -> list[dict]:
    rows = []
    for q, p in zip(cal.quotes, model):
        iv_m = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, rate)
        if iv_m is None:                       # outside no-arb: no IV to compare
            continue
        rows.append(dict(
            k=math.log(q.strike / q.fwd_pv),
            d=math.log(q.strike / q.fwd_pv) / (q.iv * math.sqrt(q.tau)),
            err=(iv_m - q.iv) * 100.0,         # vol points, + = model above mkt
            tau_d=q.tau * 365.0,
            expiry=q.expiry,
            vega=q.vega,
            hs=q.half_spread_iv,
        ))
    return rows


def _table(title: str, note: str, labels, masks, err, rows) -> None:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'bucket':>16} {'n':>5} {'mean':>8} {'rmse':>7} "
          f"{'|mean|/rmse':>12} {'med vega':>9} {'med h/s':>8}")
    print(f"  {'-' * 16} {'-' * 5} {'-' * 8} {'-' * 7} {'-' * 12} "
          f"{'-' * 9} {'-' * 8}")
    for lab, m in zip(labels, masks):
        if m.sum() < MIN_BUCKET:
            continue
        e = err[m]
        rmse = math.sqrt(float(np.mean(e ** 2)))
        idx = np.where(m)[0]
        print(f"  {lab:>16} {m.sum():>5} {e.mean():>+8.3f} {rmse:>7.3f} "
              f"{abs(float(e.mean())) / rmse if rmse else float('nan'):>12.2f} "
              f"{np.median([rows[i]['vega'] for i in idx]):>9.1f} "
              f"{np.median([rows[i]['hs'] for i in idx]):>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--max-dte", type=int, default=17)
    p.add_argument("--report-paths", type=int, default=200_000,
                   help="repricing precision; well above the 64k the gate uses "
                        "so the buckets are not reading Monte Carlo noise")
    p.add_argument("--no-refit", action="store_true",
                   help="reprice the committed artifact's parameters instead of "
                        "re-polishing them on this snapshot. Faster, but the "
                        "parameters then come from a DIFFERENT moment than the "
                        "market they are scored against, and intraday drift "
                        "lands in the residual. Refitting is the default for "
                        "that reason.")
    args = p.parse_args()

    snap = fetch_calibration_set(args.ticker, args.max_dte,
                                 min_tau_hours=MIN_TAU_HOURS)
    print(f"\nsnapshot {snap.pricing_time.isoformat()}  spot {snap.spot:.2f}  "
          f"{len(snap.quotes)} quotes  {len(snap.expiries)} expiries  "
          f"stale={snap.stale}")

    prev = json.loads(Path(CAL_FILE).read_text())
    x0 = np.array([prev["eta"], prev["rho"], prev["H"], prev["xi"]])
    cal = Calibrator(snap.rate, snap.quotes, n_paths=64_000)
    if args.no_refit:
        eta, rho, H, xi = map(float, x0)
        print(f"  repricing committed fit from {prev['as_of']}")
    else:
        print(f"  re-polishing {np.round(x0, 4)} on this snapshot ...",
              flush=True)
        res = minimize(cal.loss, x0, method="Powell",
                       bounds=[BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"],
                               BOUNDS["xi"]],
                       options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 80})
        eta, rho, H, xi = map(float, res.x)
    print(f"  eta={eta:.4f} rho={rho:.4f} H={H:.4f} "
          f"sqrt(xi)={math.sqrt(xi):.2%}")

    model = cal.model_prices(eta, rho, H, xi, n_paths=args.report_paths,
                             seed=20260810)
    rep = iv_fit_report(cal.quotes, model, snap.rate)
    print(f"  RMSE {rep['rmse_volpts']:.3f} vp at {args.report_paths:,} paths "
          f"({rep['n_scored']} scored, {rep['n_unpriceable']} unpriceable)")

    rows = _residuals(cal, snap.rate, model)
    err = np.array([r["err"] for r in rows])
    d = np.array([r["d"] for r in rows])
    k = np.array([r["k"] for r in rows])

    _table("IV ERROR BY RAW LOG-MONEYNESS ln(K/F)   [+ = model above market]",
           "confounded with maturity -- see the standardised table below",
           [f"{lo:+.3f}..{hi:+.3f}" for lo, hi in K_BANDS],
           [(k >= lo) & (k < hi) for lo, hi in K_BANDS], err, rows)

    exps = sorted({r["expiry"] for r in rows})
    _table("IV ERROR BY EXPIRY", "also confounded -- short expiries are where "
           "the deep-wing strikes live",
           exps, [np.array([r["expiry"] == e for r in rows]) for e in exps],
           err, rows)

    _table("IV ERROR BY STANDARDISED MONEYNESS d = ln(K/F)/(sigma*sqrt(tau))",
           "THIS is the one to read: comparable across maturities",
           [f"{lo:+.2f}..{hi:+.2f}" for lo, hi in D_BANDS],
           [(d >= lo) & (d < hi) for lo, hi in D_BANDS], err, rows)

    # The decisive split: hold standardised moneyness fixed and vary maturity.
    # A gap that survives here is a LEVEL failure at the short end; one that
    # vanishes means the expiry table was only ever showing the skew again.
    print("\nNEAR-DATED (<= 4d) MINUS REST, AT MATCHED STANDARDISED MONEYNESS")
    print("  a level failure shows as the same-signed gap in EVERY band")
    print(f"  {'d band':>14} {'n near':>7} {'mean':>8} {'n rest':>7} "
          f"{'mean':>8} {'gap':>8}")
    near = np.array([r["tau_d"] <= NEAR_DATED_DAYS for r in rows])
    for lo, hi in D_BANDS:
        mb = (d >= lo) & (d < hi)
        a, b = mb & near, mb & ~near
        if a.sum() < MIN_BUCKET or b.sum() < MIN_BUCKET:
            continue
        print(f"  {f'{lo:+.2f}..{hi:+.2f}':>14} {a.sum():>7} "
              f"{err[a].mean():>+8.3f} {b.sum():>7} {err[b].mean():>+8.3f} "
              f"{err[a].mean() - err[b].mean():>+8.3f}")

    # Body vs tail. The search minimises a Huber loss (linear past 2 vol points)
    # while the gate scores RMSE (quadratic). They can disagree, and when they
    # do it is because a change helped the body and cost the tail.
    a = np.abs(err)
    print("\nBODY vs TAIL -- why the search objective and the gate can disagree")
    print(f"  {'median |err|':>16} {np.median(a):>8.3f} vp")
    print(f"  {'p75 |err|':>16} {np.percentile(a, 75):>8.3f} vp")
    print(f"  {'p90 |err|':>16} {np.percentile(a, 90):>8.3f} vp")
    print(f"  {'p99 |err|':>16} {np.percentile(a, 99):>8.3f} vp")
    print(f"  {'RMSE':>16} {math.sqrt(float(np.mean(err ** 2))):>8.3f} vp"
          f"   <- gate scores this; squaring makes it tail-dominated")
    print(f"  {'worst |err|':>16} {a.max():>8.3f} vp")


if __name__ == "__main__":
    main()
