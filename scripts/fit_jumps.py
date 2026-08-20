"""Does adding jumps close the left-tail bias? A paired, controlled fit.

fit_diagnostics located the diffusive model's one genuine misspecification: at
2-3 sigma into the put wing the model sits ~2.1 vol points BELOW the market,
with |mean|/rmse = 0.86 (almost pure bias), at every maturity. The standard
explanation at 2-11 days is missing JUMP risk: a continuous-path model cannot
make the left tail fat enough because crash risk does not scale with
sqrt(tau).

This measures the hypothesis instead of asserting it. Both arms run against
ONE snapshot (two arms on different snapshots at too few paths is exactly how
this project drew wrong conclusions about the xi curve, twice):

    A  diffusive rough Bergomi        (eta, rho, H, xi)            4 params
    B  + compensated Merton jumps     + (lam, mu_j, sig_j)         7 params

both reported at REPORT_PATHS with the same seed, then bucketed by
standardised moneyness d = ln(K/F)/(sigma*sqrt(tau)). The success criterion
is fixed in advance: arm B closes the d in [-3,-2] bucket's mean bias toward
zero WITHOUT giving it back elsewhere (the body staying within noise), and the
jump parameters land at plausible values rather than a bound. RMSE alone
cannot arbitrate this -- a 7-param model beating a 4-param model on in-sample
RMSE is guaranteed and means nothing.

A jump fit is a DIFFERENT simulated driver. Its artifact (if ever written) must
carry a different kernel stamp so dataset_0dte.load_calibrated_dynamics()
refuses to adopt it into the jump-free surrogate; this script writes no
artifact at all -- it is a measurement, not a calibration path.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.optimize import differential_evolution, minimize

from backend.quant.calibrate import (BOUNDS, Calibrator, MIN_TAU_HOURS,
                                     fetch_calibration_set, implied_vol,
                                     iv_fit_report)

SEARCH, POLISH, REPORT = 8_000, 64_000, 200_000
SEED, REPORT_SEED = 7, 20260820
#: Per-market bounds. lam jumps/year, mu_j mean log-jump (negative: crashes),
#: sig_j log-jump dispersion. SPY: small frequent jumps. BTC: crypto realizes
#: 5-10% daily moves routinely, so the jump-size box must reach -25% or the
#: optimizer rails against an assumption instead of the data. BTC also gets
#: the widened eta ceiling calibrate_deribit uses: the diffusive fit on
#: 2026-08-20 railed BOTH eta (at 4.0) and H (at 0.5) trying to reach the
#: surface, which is the motivating observation for this experiment.
JUMP_BOUNDS = {
    "SPY": {"lam": (0.1, 60.0), "mu_j": (-0.06, 0.0), "sig_j": (0.003, 0.06)},
    # mu_j is SYMMETRIC for BTC. The first run bounded it at (-0.25, 0] --
    # an equity prior (crashes go down) imposed on a market that prices
    # upward jump risk too: the diffusive residuals show the CALL wing bid
    # (+1..+3 sigma bucket at -3.2 vp, model under market), and mu_j railed
    # against the zero bound trying to reach it.
    "BTC": {"lam": (0.1, 150.0), "mu_j": (-0.25, 0.25), "sig_j": (0.005, 0.25)},
}
D_BANDS = [(-3.0, -2.0), (-2.0, -1.0), (-1.0, -0.25), (-0.25, 0.25),
           (0.25, 1.0), (1.0, 3.0)]

OUT = ROOT.parent.joinpath  # unused; keep results in stdout + scratch JSON


def residuals(cal, snap, model):
    rows = []
    for q, p in zip(cal.quotes, model):
        iv_m = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, snap.rate)
        if iv_m is None:
            continue
        rows.append((math.log(q.strike / q.fwd_pv) / (q.iv * math.sqrt(q.tau)),
                     (iv_m - q.iv) * 100.0))
    d = np.array([r[0] for r in rows])
    e = np.array([r[1] for r in rows])
    return d, e


def bucket_table(tag, d, e):
    print(f"\n  {tag}: mean IV error by standardised moneyness "
          f"(+ = model above market)")
    for lo, hi in D_BANDS:
        m = (d >= lo) & (d < hi)
        if m.sum() < 4:
            continue
        rmse = math.sqrt(float(np.mean(e[m] ** 2)))
        print(f"    {lo:+.2f}..{hi:+.2f}  n={m.sum():>4}  "
              f"mean {e[m].mean():+7.3f}  rmse {rmse:6.3f}  "
              f"|mean|/rmse {abs(e[m].mean()) / rmse if rmse else 0:5.2f}")


def fetch_market(market: str):
    """(snapshot-ish meta, quotes, rate, jump bounds, diffusive bounds)."""
    if market == "SPY":
        snap = fetch_calibration_set("SPY", 17, min_tau_hours=MIN_TAU_HOURS)
        diff_bounds = [BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]
        return (f"{snap.pricing_time.isoformat()} spot {snap.spot:.2f} "
                f"stale={snap.stale}"), snap.quotes, snap.rate, diff_bounds
    from backend.quant.calibrate_deribit import (ETA_MAX_BTC,
                                                 quotes_from_surface)
    from backend.quant.deribit import DeribitClient
    from backend.quant.surface import build_surface
    snap = DeribitClient().snapshot("BTC")
    quotes, drops = quotes_from_surface(build_surface(snap))
    diff_bounds = [(BOUNDS["eta"][0], ETA_MAX_BTC), BOUNDS["rho"],
                   BOUNDS["H"], BOUNDS["xi"]]
    return (f"{snap.captured_at_iso} index {snap.index_price:,.0f} "
            f"(live 24/7)"), quotes, 0.0, diff_bounds


def fit_arm(cal, jumps: bool, diff_bounds, jump_bounds,
            warm_start: np.ndarray | None = None):
    bounds = list(diff_bounds)
    if jumps:
        bounds += list(jump_bounds.values())

    def objective(theta):
        cal.jumps = tuple(map(float, theta[4:7])) if jumps else None
        return cal.loss(np.asarray(theta[:4], dtype=float))

    t0 = time.perf_counter()
    cal.n_paths = SEARCH
    de = differential_evolution(objective, bounds, seed=SEED, maxiter=12,
                                popsize=6, tol=1e-3, mutation=(0.4, 0.9),
                                recombination=0.8, polish=False, init="sobol",
                                updating="deferred")
    starts = [de.x]
    if jumps and warm_start is not None:
        # The jump model CONTAINS the diffusive one (lam -> 0), so any jump
        # fit worse than the diffusive fit is a search failure, not a result.
        # The first BTC run proved this the hard way: the 7-param DE landed in
        # a basin at 6.4 vp while the 4-param arm sat at 2.8. Seeding a second
        # polish from (diffusive optimum, tiny jumps) makes the comparison
        # structurally fair: the jump arm can only ever lose by refusing to
        # use its extra parameters, never by failing to find the subspace.
        lam0 = jump_bounds["lam"][0]
        mid = lambda b: 0.5 * (b[0] + b[1])
        starts.append(np.array([*warm_start[:4], lam0,
                                mid(jump_bounds["mu_j"]),
                                mid(jump_bounds["sig_j"])]))
    best = None
    for x0 in starts:
        r2 = minimize(objective, x0, method="Powell", bounds=bounds,
                      options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 200})
        cal.n_paths = POLISH
        r3 = minimize(objective, r2.x, method="Powell", bounds=bounds,
                      options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 200})
        cal.n_paths = SEARCH
        if best is None or r3.fun < best.fun:
            best = r3
    secs = time.perf_counter() - t0
    theta = np.asarray(best.x, dtype=float)
    cal.jumps = tuple(theta[4:7]) if jumps else None
    return theta, secs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="SPY", choices=("SPY", "BTC"))
    market = ap.parse_args().market
    meta, all_quotes, rate, diff_bounds = fetch_market(market)
    jump_bounds = JUMP_BOUNDS[market]
    expiries = sorted({q.expiry for q in all_quotes})
    print(f"\nONE SNAPSHOT [{market}] {meta}  {len(all_quotes)} quotes  "
          f"{len(expiries)} expiries", flush=True)

    class _Snap:                       # the one field residuals() reads
        pass
    snap = _Snap()
    snap.rate = rate
    snap.quotes = all_quotes

    results = {}
    for jumps in (False, True):
        tag = "jumps" if jumps else "diffusive"
        cal = Calibrator(snap.rate, snap.quotes, n_paths=SEARCH)
        print(f"\n--- {tag} arm ({4 + 3 * jumps} parameters) ---", flush=True)
        theta, secs = fit_arm(cal, jumps, diff_bounds, jump_bounds)
        eta, rho, H, xi = theta[:4]
        print(f"  eta={eta:.4f} rho={rho:.4f} H={H:.4f} "
              f"sqrt(xi)={math.sqrt(xi):.2%}  ({secs:.0f}s)", flush=True)
        if jumps:
            lam, mu_j, sig_j = theta[4:7]
            print(f"  lam={lam:.2f}/yr  mu_j={mu_j:+.4f}  sig_j={sig_j:.4f}  "
                  f"(expected jump {math.exp(mu_j + sig_j**2 / 2) - 1:+.2%}, "
                  f"~{lam * 10 / 252:.2f} jumps per 10 trading days)",
                  flush=True)
            for name, v, (lo, hi) in (("lam", lam, jump_bounds["lam"]),
                                      ("mu_j", mu_j, jump_bounds["mu_j"]),
                                      ("sig_j", sig_j, jump_bounds["sig_j"])):
                frac = (v - lo) / (hi - lo)
                if not 0.02 < frac < 0.98:
                    print(f"  WARNING: {name} at {frac:.1%} of its bound "
                          f"[{lo}, {hi}] — treat this arm as unidentified",
                          flush=True)

        model = cal.model_prices(eta, rho, H, xi, n_paths=REPORT,
                                 seed=REPORT_SEED)
        rep = iv_fit_report(cal.quotes, model, snap.rate)
        d, e = residuals(cal, snap, model)
        left = e[(d >= -3) & (d < -2)]
        body = e[(d >= -2) & (d <= 2)]
        print(f"  RMSE {rep['rmse_volpts']:.3f} vp at {REPORT:,} paths "
              f"({rep['n_unpriceable']} unpriceable)", flush=True)
        bucket_table(tag, d, e)
        results[tag] = {
            "theta": [round(float(v), 6) for v in theta],
            "rmse": round(rep["rmse_volpts"], 3),
            "left_tail_bias": round(float(left.mean()), 3) if len(left) else None,
            "left_tail_n": int(len(left)),
            "body_bias": round(float(body.mean()), 3),
            "seconds": round(secs),
        }

    a, b = results["diffusive"], results["jumps"]
    print(f"\n{'=' * 62}")
    print(f"{'':>18} {'diffusive':>12} {'jumps':>12}")
    for k in ("rmse", "left_tail_bias", "body_bias", "seconds"):
        print(f"{k:>18} {str(a[k]):>12} {str(b[k]):>12}")
    closed = (a["left_tail_bias"] is not None and b["left_tail_bias"] is not None
              and abs(b["left_tail_bias"]) < 0.5 * abs(a["left_tail_bias"]))
    print(f"\nleft-tail bias {'CLOSED >50%' if closed else 'NOT closed'} "
          f"({a['left_tail_bias']} -> {b['left_tail_bias']} vp on "
          f"{a['left_tail_n']} quotes)")

    out = Path(__file__).with_name(f"_fit_jumps_last_{market.lower()}.json")
    out.write_text(json.dumps({"market": market, "as_of": meta,
                               "n_quotes": len(all_quotes), **results},
                              indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
