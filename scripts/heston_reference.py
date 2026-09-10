"""Heston as the classical reference: verification, calibration to the committed
SPY captures, and the short-dated ATM skew term structure against the market
and rough Bergomi.

What this script measures
-------------------------
1. Verification of backend/quant/heston.py: the Fang & Oosterlee (2008) Table 3
   reference value versus N and versus the truncation range L, put-call
   parity, the Black-Scholes limit, the closed-form cumulants against the
   characteristic function, a full-truncation Euler Monte Carlo cross-check
   (Lord, Koekkoek & van Dijk 2010) and the pricer's wall-clock.

2. Calibration of (v0, kappa, theta, sigma_v, rho) to committed SPY captures
   under data/surfaces/equity/ (backend.quant.calibrate_map.quotes_from_capture),
   on exactly the quote set the rough-Bergomi neural map was fitted on (the
   map's (k, tau) box, MapPricer.usable), priced on each expiry's forward with
   r = 0 as backend/quant/surface.py does. Three objectives per capture:
     LS       plain least squares on implied vols in vol points (the number
              minimised is the RMSE reported);
     Huber    Huber with delta = 2 vp, the objective the map calibration in
              artifacts/intraday_params.json used, for a like-for-like RMSE;
     LS-hs    least squares weighted by 1/half_spread_iv (the market's own
              resolution);
     LS-kappa10, LS-kappa3   least squares with the mean-reversion bound
              lowered to 10/yr and 3/yr (1/kappa >= 25 and >= 84 trading
              days): the textbook regime in which Heston's short-dated skew
              is flat in T, and what that regime costs on the smile.
   (The captures are the ones scripts/intraday_params.py enumerates; there
   are no spy_*.json.gz files under data/pricing_map*/, which hold the map's
   training shards.)
   Each capture's recorded rough-Bergomi map RMSE (rmse_volpts) is read from
   artifacts/intraday_params.json and ALSO recomputed on the identical quote
   set from the recorded parameters, so the comparison is on one footing.

3. The classical failure. With each capture's calibrated Heston parameters,
   psi(T) = d sigma_imp/dk at k = 0 on the maturity ladder of
   docs/atm_skew_term_structure.json (read, never rerun), fitted as a power
   law over T <= 45 trading days exactly as that document did
   (scripts.atm_skew_term_structure.fit_power_law, imported, not edited), and
   laid over the market points and the rough-Bergomi curve read from the JSON.
   Heston's skew is semi-analytic, so its psi carries no Monte Carlo error:
   the "fine" psi uses h = 1e-4 (numerically exact), and the document's own
   five-point stencil with h = max(0.25 atm_iv sqrt(T), 0.002) is computed
   alongside with its truncation estimate, so both conventions are on record.
   The power-law "SE" quoted for a Heston exponent is the straight-line
   regression SE of a noise-free curve: it measures how far the log-log
   curve is from a straight line, not sampling error.

4. The kappa profile. On the first capture, kappa is HELD at each of
   KAPPA_PROFILE (1 ... 100 per year) and the other four parameters are
   refitted, recording the smile RMSE and the skew exponent side by side.
   This locates, on one axis, the textbook regime (T << 1/kappa, flat skew)
   and the regime the free fit selects, and what each costs on the smile.

    python -m scripts.heston_reference            # ~15 min CPU
    python -m scripts.heston_reference --quick    # 2 captures, LS only

Outputs docs/heston_reference.{png,json}; docs/heston_reference.md is written
by hand from the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant import heston as H  # noqa: E402
from backend.quant.calibrate import Quote, bs_call  # noqa: E402
from backend.quant.calibrate_map import MapPricer, quotes_from_capture  # noqa: E402
from scripts import atm_skew_term_structure as ats  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DATA = ROOT / "data" / "surfaces" / "equity"
TRADING_DAYS = 252.0
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, sigma_v=0.5751, rho=-0.5711)
FO_REF = 5.785155450
#: rounded parameters of the kind a short-dated SPY calibration selects (below);
#: used only for the wide-range convergence study in verification()
SPY_LIKE = dict(v0=0.0016, kappa=100.0, theta=0.024, sigma_v=4.7, rho=-0.57)
#: trading-hour captures of 2026-08-20 that are BOTH in the skew document's
#: capture list (13:17, 14:06, 15:15 NY) and in artifacts/intraday_params.json,
#: plus two more from the same session that the intraday run also fitted.
DEFAULT_CAPTURES = ("spy_20260820T171756Z.json.gz", "spy_20260820T174820Z.json.gz",
                    "spy_20260820T180618Z.json.gz", "spy_20260820T191516Z.json.gz",
                    "spy_20260820T194521Z.json.gz")
#: mean-reversion speeds (per year) at which the OTHER four parameters are
#: refitted on the first capture: the smile RMSE and the skew exponent as a
#: function of kappa, so "flat Heston skew" and "good Heston smile" can be
#: located on one axis. 1/kappa in trading days: 252, 126, 84, 50, 25, 13, 5, 2.5.
KAPPA_PROFILE = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
BG, PANEL, BLUE, AMBER, MINT, WHITE, GRID, GREY = (
    "#000000", "#1c1c1e", "#0A84FF", "#FF9F0A", "#30D158", "#FFFFFF", "#3a3a3c",
    "#8E8E93")


def log(msg: str) -> None:
    print(msg, flush=True)


# ── 1. verification ────────────────────────────────────────────────────────
def verification() -> dict:
    out: dict = {}
    t0 = time.perf_counter()
    rows = []
    ref14 = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** 14, L=12.0,
                          range_tol=float("inf"), **FO)[0]
    for n in (5, 6, 7, 8, 9, 10, 11, 12):
        fixed = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** n, L=12.0,
                              range_tol=float("inf"), **FO)[0]
        adapt = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=2 ** n, **FO)[0]
        rep = H.cos_range_report(1.0, N=2 ** n, **FO)
        rows.append({"N": 2 ** n, "call_L12": fixed, "err_L12_vs_ref": fixed - FO_REF,
                     "err_L12_vs_N14": fixed - ref14, "call_default": adapt,
                     "err_default_vs_ref": adapt - FO_REF, "L_used": rep["L"],
                     "N_used": rep["N"], "defect": rep["defect"]})
    out["fo2008_vs_N"] = rows
    lrows = []
    for L in (8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0):
        c = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=4096, L=L,
                          range_tol=float("inf"), **FO)[0]
        p = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=4096, L=L,
                          range_tol=float("inf"), via_parity=True, **FO)[0]
        rep = H.cos_range_report(1.0, N=4096, L=L, range_tol=float("inf"), **FO)
        lrows.append({"L": L, "call_err": c - FO_REF, "put_parity_err": p - FO_REF,
                      "defect": rep["defect"], "a": rep["a"], "b": rep["b"]})
    out["fo2008_vs_L"] = lrows
    out["fo2008_reference"] = FO_REF
    out["fo2008_converged"] = H.heston_call(100.0, 100.0, 1.0, 0.0, 0.0, N=4096, **FO)[0]

    # convergence in N where the spec's 2^8 -> 2^10 -> 2^12 ladder IS resolvable:
    # a pinned WIDE range (L = 40) leaves few terms per unit of width, so the
    # error is above the floating floor at 2^8 and geometric beyond it
    wide = []
    for name, pp, T_ in (("FO2008 T=1", FO, 1.0), ("SPY-like T=1d", SPY_LIKE, 1.0 / TRADING_DAYS)):
        Ks = np.array([90.0, 100.0, 110.0]) if T_ > 0.05 else np.array([97.0, 100.0, 103.0])
        kw = dict(L=40.0, range_tol=float("inf"), **pp)
        ref = H.heston_call(100.0, Ks, T_, 0.0, 0.0, N=2 ** 15, **kw)
        errs = {str(2 ** n): float(np.max(np.abs(
            H.heston_call(100.0, Ks, T_, 0.0, 0.0, N=2 ** n, **kw) - ref))) for n in (6, 7, 8, 9, 10, 11, 12)}
        wide.append({"case": name, "params": pp, "T": T_, "L": 40.0, "K": Ks.tolist(),
                     "max_abs_err_vs_N15": errs})
    out["convergence_wide_range"] = wide

    S, T, r, q = 100.0, 0.75, 0.03, 0.01
    K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    c = H.heston_call(S, K, T, r, q, **FO)
    p = H.heston_put(S, K, T, r, q, **FO)
    out["parity_max_abs_err"] = float(np.max(np.abs(c - p - (S * math.exp(-q * T) - K * math.exp(-r * T)))))
    out["direct_vs_parity_call_max_abs_err"] = float(np.max(np.abs(
        c - H.heston_call(S, K, T, r, q, via_parity=True, **FO))))

    K = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    bs = np.array([bs_call(100.0, k, 1.0, 0.2, 0.05) for k in K])
    out["bs_limit"] = {str(sv): float(np.max(np.abs(
        H.heston_call(100.0, K, 1.0, 0.05, 0.0, v0=0.04, kappa=1.5, theta=0.04,
                      sigma_v=sv, rho=-0.7) - bs))) for sv in (0.0, 1e-9, 1e-6, 1e-4, 1e-2)}

    cum = []
    for T_ in (0.01, 0.5, 2.0):
        for pp in (FO, dict(v0=0.006, kappa=50.0, theta=0.02, sigma_v=5.0, rho=-0.6)):
            c1, c2 = H.heston_cumulants(T_, **pp)
            h = 1e-3

            def f(u):
                return np.log(H.heston_cf(np.array([u], dtype=complex), T_, **pp))[0]

            d1 = (4 * (f(h / 2) - f(-h / 2)) / h - (f(h) - f(-h)) / (2 * h)) / 3
            d2 = (4 * (f(h / 2) - 2 * f(0.0) + f(-h / 2)) / (h / 2) ** 2
                  - (f(h) - 2 * f(0.0) + f(-h)) / h ** 2) / 3
            cum.append({"T": T_, "sigma_v": pp["sigma_v"], "c1": c1, "c1_numeric": float(d1.imag),
                        "c2": c2, "c2_numeric": float(-d2.real),
                        "c2_rel_err": float(abs(c2 + d2.real) / c2)})
    out["cumulants"] = cum

    mc_rows = []
    for (T_, steps, seed) in ((1.0, 400, 1), (0.5, 400, 1)):
        Ks = np.array([90.0, 100.0, 110.0])
        t1 = time.perf_counter()
        mc, se = H.heston_mc_call(100.0, Ks, T_, 0.02, 0.0, n_paths=200_000,
                                  n_steps=steps, seed=seed, chunk=100_000, **FO)
        cos = H.heston_call(100.0, Ks, T_, 0.02, 0.0, **FO)
        mc_rows.append({"T": T_, "n_steps": steps, "n_paths": 200_000, "seed": seed,
                        "K": Ks.tolist(), "cos": cos.tolist(), "mc": mc.tolist(),
                        "se": se.tolist(), "z": ((mc - cos) / se).tolist(),
                        "seconds": time.perf_counter() - t1})
    out["monte_carlo"] = mc_rows

    Ks = np.linspace(60, 140, 100)
    for N in (256, 1024):
        t1 = time.perf_counter()
        for _ in range(100):
            H.heston_call(100.0, Ks, 0.1, 0.0, 0.0, N=N, **FO)
        out[f"ms_per_100_strike_smile_N{N}"] = (time.perf_counter() - t1) * 10.0
        t1 = time.perf_counter()
        for _ in range(100):
            H.heston_implied_vol(100.0, Ks, 0.1, 0.0, 0.0, N=N, **FO)
        out[f"ms_per_100_strike_iv_N{N}"] = (time.perf_counter() - t1) * 10.0
    out["seconds"] = time.perf_counter() - t0
    return out


# ── 2. calibration ─────────────────────────────────────────────────────────
def load_intraday() -> dict[str, dict]:
    d = json.loads((ARTIFACTS / "intraday_params.json").read_text(encoding="utf-8"))
    return {r["capture"]: r for r in d["rows"]}


def map_ivs_for(pricer: MapPricer, quotes: list[Quote], rec: dict) -> np.ndarray:
    return pricer.ivs(quotes, rec["eta"], rec["rho"], rec["H"], rec["sqrt_xi"] ** 2)


def calibrate_capture(path: Path, pricer: MapPricer, rec: dict | None,
                      variants: tuple[str, ...], n_random: int, N: int) -> dict:
    quotes_all, rate, meta = quotes_from_capture(path)
    quotes, n_out = pricer.usable(quotes_all)
    groups = H._groups(quotes, rate)
    info = {"capture": path.name, "pricing_time": meta.get("pricing_time"),
            "spot": meta.get("spot"), "rate": rate, "stale": meta.get("stale"),
            "n_quotes_capture": len(quotes_all), "n_quotes_in_map_box": len(quotes),
            "n_outside_box": n_out, "expiries": [g["expiry"] for g in groups],
            "expiry_taus": [g["tau"] for g in groups]}
    log(f"[{path.name}] {info['pricing_time']}  spot {info['spot']:.2f}  "
        f"{len(quotes)} quotes in the map box ({n_out} outside), "
        f"{len(info['expiries'])} expiries")
    fits = {}
    for name in variants:
        kw = {"LS": dict(loss="linear", weights="uniform", n_random_starts=n_random),
              "Huber": dict(loss="huber", weights="uniform", n_random_starts=0),
              "LS-hs": dict(loss="linear", weights="half_spread", n_random_starts=0),
              # textbook mean reversion: 1/kappa >= 25 trading days, the regime
              # in which Heston's short-dated skew is flat in T
              "LS-kappa10": dict(loss="linear", weights="uniform", n_random_starts=0,
                                 bounds={**H.HESTON_BOUNDS, "kappa": (1e-2, 10.0)}),
              # textbook mean reversion proper: 1/kappa >= 84 trading days, the
              # regime of the "flat short-dated Heston skew" folklore
              "LS-kappa3": dict(loss="linear", weights="uniform", n_random_starts=0,
                                bounds={**H.HESTON_BOUNDS, "kappa": (1e-2, 3.0)}),
              "LS-kappa1000": dict(loss="linear", weights="uniform", n_random_starts=0,
                                   bounds={**H.HESTON_BOUNDS, "kappa": (1e-2, 1000.0)}),
              }[name]
        fit = H.calibrate_heston(quotes, rate, N=N, **kw)
        d = fit.as_dict()
        d["range_at_optimum"] = [
            dict(expiry=g["expiry"], tau=g["tau"], **H.cos_range_report(
                g["tau"], N=N, **fit.params)) for g in H._groups(quotes, rate)]
        fits[name] = d
        p = fit.params
        log(f"  {name:<12} RMSE {fit.rmse_volpts:.4f} vp  (weighted {fit.rmse_weighted_volpts:.4f}; "
            f"unpriceable {fit.n_unpriceable})  v0={p['v0']:.5f} (sqrt {math.sqrt(p['v0']):.2%}) "
            f"kappa={p['kappa']:.2f} theta={p['theta']:.5f} (sqrt {math.sqrt(p['theta']):.2%}) "
            f"sigma_v={p['sigma_v']:.3f} rho={p['rho']:.4f}  Feller {fit.feller_ratio:.3f} "
            f"{'ok' if fit.feller else 'VIOLATED'}  {fit.seconds:.1f}s {fit.n_evals} evals  "
            f"pinned: {fit.pinned or 'none'}")
        best_costs = sorted(s["cost"] for s in fit.starts if np.isfinite(s.get("cost", np.nan)))
        log(f"    starts: {len(fit.starts)}, final costs {', '.join(f'{c:.3f}' for c in best_costs)}")
    # rough-Bergomi map on the same quotes. artifacts/intraday_params.json was
    # produced by "pricing_map v3" (2026-08-20); the committed
    # artifacts/pricing_map.pt is map v5 (committed 2026-08-21), so the
    # recorded RMSE is (i) reported as recorded, (ii) recomputed by evaluating
    # v5 at v3's optimum, and (iii) REFIT with v5 on the identical quote set -
    # (iii) is the like-for-like number.
    mkt = np.array([q.iv for q in quotes])

    def per_expiry(err):
        out = []
        for e in info["expiries"]:
            m = np.array([q.expiry == e for q in quotes])
            out.append({"expiry": e, "n": int(m.sum()),
                        "rmse_volpts": float(np.sqrt(np.mean(err[m] ** 2)))})
        return out

    map_block: dict = {"recorded": rec, "n_recorded": rec["n"] if rec else None}
    if rec is not None:
        err = (map_ivs_for(pricer, quotes, rec) - mkt) * 100.0
        map_block["rmse_recomputed_volpts"] = float(np.sqrt(np.mean(err ** 2)))
        map_block["per_expiry_recorded_params"] = per_expiry(err)
    from backend.quant.calibrate_map import MapCalibrator
    cal = MapCalibrator(quotes_all, market="SPY", pricer=pricer)
    theta, secs = cal.fit()
    eta, rho_m, H_m, xi = map(float, theta)
    refit = {"eta": eta, "rho": rho_m, "H": H_m, "sqrt_xi": math.sqrt(xi), "xi": xi,
             "rmse_volpts": cal.rmse_volpts(theta), "seconds": secs, "n_evals": cal.n_evals,
             "n": len(cal.quotes)}
    err = (map_ivs_for(pricer, quotes, refit) - mkt) * 100.0
    refit["per_expiry"] = per_expiry(err)
    map_block["refit"] = refit
    log(f"  map (rough Bergomi) recorded v3 RMSE "
        f"{(rec or {}).get('rmse_volpts', float('nan')):.3f} vp on n={(rec or {}).get('n')}; "
        f"v5 at those params {map_block.get('rmse_recomputed_volpts', float('nan')):.4f} vp; "
        f"v5 REFIT {refit['rmse_volpts']:.4f} vp on n={refit['n']} (eta={eta:.4f} rho={rho_m:.4f} "
        f"H={H_m:.4f} sqrt_xi={math.sqrt(xi):.4f}; {secs:.1f}s, {cal.n_evals} evals)")
    return {"info": info, "fits": fits, "map": map_block}


# ── 3. skew term structure ─────────────────────────────────────────────────
def heston_skew_ladder(params: dict, T_days) -> list[dict]:
    rows = []
    for d in T_days:
        T = float(d) / TRADING_DAYS
        fine = H.heston_atm_skew(T, h=1e-4, **params)
        h = max(0.25 * fine["atm_iv"] * math.sqrt(T), 0.002)
        st = H.heston_atm_skew(T, h=h, **params)
        rows.append({"T_days": float(d), "T": T, "atm_iv": fine["atm_iv"],
                     "psi_fine": fine["psi"], "h_fine": 1e-4,
                     "psi": st["psi"], "psi_2h": st["psi_2h"],
                     "psi_richardson": st["psi_richardson"],
                     "truncation": st["truncation"], "h": h})
    return rows


def skew_fits(rows: list[dict], params: dict) -> dict:
    T = [r["T"] for r in rows]
    short = [r["T_days"] <= ats.SHORT_FIT_MAX_DAYS for r in rows]
    fine = [r["psi_fine"] for r in rows]
    se_fine = [1e-6 * abs(v) for v in fine]           # uniform weights in log space
    fits = {
        "fine_short": ats.fit_power_law(T, fine, se_fine, short),
        "fine_all": ats.fit_power_law(T, fine, se_fine),
        "stencil_short": ats.fit_power_law(T, [r["psi"] for r in rows],
                                           [max(r["truncation"], 1e-6) for r in rows], short),
        "richardson_short": ats.fit_power_law(T, [r["psi_richardson"] for r in rows],
                                              [max(r["truncation"], 1e-6) for r in rows], short),
        "local_fine": ats.local_exponents(
            [dict(r, psi=r["psi_fine"], se=1e-6 * abs(r["psi_fine"])) for r in rows]),
        "short_limit_analytic": H.heston_short_skew_limit(params["v0"], params["sigma_v"],
                                                          params["rho"]),
        "one_over_kappa_days": TRADING_DAYS / params["kappa"],
    }
    return fits


def heston_at_capture_expiries(params: dict, expiries: list[str], taus: list[float],
                               market_rows: list[dict], capture: str) -> dict:
    """Heston psi at the capture's own listed expiries (the window the smile
    was fitted on), its power-law exponent there, and - where the skew
    document measured this capture - the market's psi at the same expiries."""
    mk = {r["expiry"]: r for r in market_rows if r["capture"] == capture}
    rows = []
    for e, tau in zip(expiries, taus):
        s = H.heston_atm_skew(tau, h=1e-4, **params)
        row = {"expiry": e, "tau": tau, "T_days_equiv": tau * TRADING_DAYS,
               "heston_psi": s["psi"], "heston_atm_iv": s["atm_iv"]}
        if e in mk:
            row.update({"market_psi": mk[e]["psi"], "market_se": mk[e]["se"],
                        "market_atm_iv": mk[e]["atm_iv"],
                        "diff_sigmas": (s["psi"] - mk[e]["psi"]) / mk[e]["se"]})
        rows.append(row)
    fit = ats.fit_power_law([r["tau"] for r in rows], [r["heston_psi"] for r in rows],
                            [1e-6 * abs(r["heston_psi"]) for r in rows])
    return {"rows": rows, "fit_capture_window": fit, "n_market_rows": len(mk)}


def kappa_profile(path: Path, pricer: MapPricer, N: int, T_days, market_rows: list[dict],
                  kappas=KAPPA_PROFILE) -> list[dict]:
    """On one capture, hold kappa at each value in `kappas`, refit
    (v0, theta, sigma_v, rho) by least squares, and record the smile RMSE next
    to the skew exponent on the document ladder and over the capture's own
    expiries. One axis, two quantities: where Heston's skew is flat in T and
    where its smile fits are different places on it."""
    quotes_all, rate, meta = quotes_from_capture(path)
    quotes, _ = pricer.usable(quotes_all)
    groups = H._groups(quotes, rate)
    expiries = [g["expiry"] for g in groups]
    taus = [g["tau"] for g in groups]
    rows = []
    for kap in kappas:
        fit = H.calibrate_heston(quotes, rate, N=N, fixed={"kappa": float(kap)},
                                 n_random_starts=0)
        prm = fit.params
        ladder = heston_skew_ladder(prm, T_days)
        fits = skew_fits(ladder, prm)
        win = heston_at_capture_expiries(prm, expiries, taus, market_rows, path.name)
        by_day = {r["T_days"]: r for r in ladder}
        row = {"kappa": float(kap), "one_over_kappa_days": TRADING_DAYS / kap,
               "params": prm, "se": fit.se, "rmse_volpts": fit.rmse_volpts,
               "n_unpriceable": fit.n_unpriceable, "pinned": fit.pinned,
               "feller_ratio": fit.feller_ratio, "seconds": fit.seconds,
               "n_evals": fit.n_evals,
               "per_expiry": fit.per_expiry,
               "b_ladder_short": fits["fine_short"]["b"],
               "se_b_ladder_short": fits["fine_short"]["se_b"],
               "b_capture_window": win["fit_capture_window"]["b"],
               "se_b_capture_window": win["fit_capture_window"]["se_b"],
               "local_fine": fits["local_fine"],
               "psi_1d": by_day[1.0]["psi_fine"], "psi_3d": by_day[3.0]["psi_fine"],
               "psi_12d": by_day[12.0]["psi_fine"], "psi_45d": by_day[45.0]["psi_fine"],
               "short_limit_analytic": fits["short_limit_analytic"],
               "ladder": ladder, "capture_window": win}
        rows.append(row)
        p = prm
        log(f"  kappa={kap:>6.1f} (1/kappa {row['one_over_kappa_days']:>5.1f} d)  "
            f"RMSE {fit.rmse_volpts:.4f} vp  sqrt(v0) {math.sqrt(p['v0']):.2%} "
            f"sqrt(theta) {math.sqrt(p['theta']):.2%} sigma_v {p['sigma_v']:.3f} "
            f"rho {p['rho']:.4f}  slope T<=45d {row['b_ladder_short']:+.4f}, over the "
            f"capture's expiries {row['b_capture_window']:+.4f}  psi(3d) {row['psi_3d']:+.3f} "
            f"psi(12d) {row['psi_12d']:+.3f}  {fit.seconds:.1f}s  pinned: {fit.pinned or 'none'}")
    return rows


# ── 4. figure ──────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=WHITE, labelsize=12, which="both")
    ax.grid(True, which="major", color=GRID, lw=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5)


def make_figure(payload: dict, out_png: Path) -> None:
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    LIGHT_BLUE = "#7CC4FF"
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.45], left=0.055, right=0.985,
                          top=0.795, bottom=0.075, wspace=0.16, hspace=0.42)
    sm = payload["smiles"]
    for i, blk in enumerate(sm["panels"]):
        ax = fig.add_subplot(gs[i, 0])
        _style(ax)
        k = np.array(blk["k"])
        ax.errorbar(k, 100 * np.array(blk["iv"]), yerr=np.array(blk["half_spread_iv"]),
                    fmt="D", ms=3.2, color=AMBER, ecolor=AMBER, elinewidth=0.7,
                    capsize=0, alpha=0.9, label="market mid IV (bars: half spread)")
        kg = np.array(blk["k_grid"])
        ax.plot(kg, 100 * np.array(blk["heston_iv"]), color=BLUE, lw=2.2,
                label=f"Heston LS fit, expiry RMSE {blk['heston_rmse']:.2f} vp")
        if blk.get("map_iv") is not None:
            ax.plot(kg, 100 * np.array(blk["map_iv"]), color=MINT, lw=2.0, ls="--",
                    label=f"rough Bergomi map (refit), expiry RMSE {blk['map_rmse']:.2f} vp")
        ax.set_xlim(k.min() - 0.005, k.max() + 0.005)
        lo = 100 * min(np.nanmin(blk["iv"]), np.nanmin(blk["heston_iv"]))
        hi = 100 * max(np.nanmax(blk["iv"]), np.nanmax(blk["heston_iv"]))
        ax.set_ylim(lo - 1.5, hi + 0.12 * (hi - lo) + 1.5)
        ax.set_ylabel("implied vol (%)", color=WHITE, fontsize=12)
        ax.set_title(f"{blk['expiry']}  (tau = {blk['tau']:.4f} y = {blk['tau'] * TRADING_DAYS:.1f} "
                     f"trading-day equiv.), n = {blk['n']}", color=WHITE, fontsize=12.5, pad=6)
        if i == 2:
            ax.set_xlabel("log-moneyness k = ln(K / F)", color=WHITE, fontsize=12)
        leg = ax.legend(loc="upper right", fontsize=9.5, facecolor=BG, edgecolor=GRID,
                        labelcolor=WHITE, framealpha=0.9)
        leg.get_frame().set_linewidth(0.8)

    ax = fig.add_subplot(gs[:, 1])
    _style(ax)
    sk = payload["skew"]
    rb = sk["rough_bergomi"]
    Trb = np.array([r["T"] for r in rb["model"]])
    prb = np.abs([r["psi"] for r in rb["model"]])
    srb = np.array([r["se"] for r in rb["model"]])
    ax.fill_between(Trb, prb - srb, prb + srb, color=MINT, alpha=0.25, lw=0)
    ax.plot(Trb, prb, color=MINT, lw=2.4, marker="o", ms=5,
            label=f"rough Bergomi MC (H = {rb['H']:.3f}), slope T<=45d "
                  f"{rb['fit_short']['b']:+.3f} +- {rb['fit_short']['se_b']:.3f}")
    mk = sk["market"]
    Tm = np.array([r["tau"] for r in mk["rows"]])
    pm = np.abs([r["psi"] for r in mk["rows"]])
    smk = np.array([r["se"] for r in mk["rows"]])
    rng = np.random.default_rng(0)
    jit = np.exp(rng.uniform(-0.03, 0.03, size=len(Tm)))
    ax.errorbar(Tm * jit, pm, yerr=smk, fmt="D", color=AMBER, ms=5, alpha=0.9,
                ecolor=AMBER, elinewidth=0.8, capsize=0,
                label=f"SPY market, {len(Tm)} expiry x capture rows, slope "
                      f"{mk['fit']['b']:+.3f} +- {mk['fit']['se_b']:.3f}")
    fk = mk["fit"]
    Tf = np.geomspace(fk["T_min"], fk["T_max"], 40)
    ax.plot(Tf, fk["C"] * Tf ** fk["b"], "--", color=AMBER, lw=1.6)
    hs = sk["heston"]
    vs = sk["summary"]["variants"]
    ls_blocks = [b for b in hs["per_capture"] if b["variant"] == "LS"]
    kaps = [b["params"]["kappa"] for b in ls_blocks]
    kap_txt = (f"kappa = {min(kaps):.3g}" if max(kaps) - min(kaps) < 0.05 * max(kaps)
               else f"kappa {min(kaps):.3g}-{max(kaps):.3g}")
    for j, blk in enumerate(ls_blocks):
        Th = np.array([r["T"] for r in blk["ladder"]])
        ph = np.abs([r["psi_fine"] for r in blk["ladder"]])
        lab = None
        if j == 0:
            lab = (f"Heston LS optimum, {len(ls_blocks)} captures ({kap_txt}/yr), "
                   f"slope T<=45d {vs['LS']['b_mean']:+.3f} (sd {vs['LS']['b_sd']:.3f})")
        ax.plot(Th, ph, color=BLUE, lw=2.6 if j == 0 else 1.2, alpha=1.0 if j == 0 else 0.55,
                marker="o" if j == 0 else None, ms=5, label=lab)
    f0 = ls_blocks[0]["fits"]["fine_short"]
    Tf0 = np.geomspace(f0["T_min"], f0["T_max"], 40)
    ax.plot(Tf0, f0["C"] * Tf0 ** f0["b"], "--", color=BLUE, lw=1.2, alpha=0.8)
    for variant, ls_, mk_ in (("LS-kappa10", "-.", "s"), ("LS-kappa3", ":", "^")):
        blocks = [b for b in hs["per_capture"] if b["variant"] == variant]
        if not blocks:
            continue
        blk = blocks[0]
        Th = np.array([r["T"] for r in blk["ladder"]])
        ph = np.abs([r["psi_fine"] for r in blk["ladder"]])
        kap = blk["params"]["kappa"]
        ax.plot(Th, ph, color=LIGHT_BLUE, lw=2.0, ls=ls_, marker=mk_, ms=4.5,
                label=f"Heston {variant[3:]} bound, first capture (kappa = {kap:.3g}, 1/kappa = "
                      f"{TRADING_DAYS / kap:.0f} d), slope T<=45d {vs[variant]['b_mean']:+.3f} "
                      f"(sd {vs[variant]['b_sd']:.3f})")
        if variant == "LS-kappa3":
            lim = abs(blk["fits"]["short_limit_analytic"])
            ax.axhline(lim, color=GREY, lw=1.2, ls=":",
                       label=f"its T->0 limit |rho sigma_v / (4 sqrt v0)| = {lim:.2f} (slope 0)")
    ref = rb["reference_slope"]
    Tc = math.sqrt(fk["T_min"] * fk["T_max"])
    yc = fk["C"] * Tc ** fk["b"]
    ax.plot(Tf, yc * (Tf / Tc) ** ref, ":", color=WHITE, lw=1.8,
            label=f"reference slope H - 1/2 = {ref:+.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ys = np.concatenate([prb, pm, np.abs([r["psi_fine"] for b in hs["per_capture"] for r in b["ladder"]])])
    ax.set_ylim(ys.min() / 4.5, ys.max() * 1.5)
    ax.set_xlabel("T (years, log scale)", color=WHITE, fontsize=13)
    ax.set_ylabel("|psi(T)| = |d sigma_imp / dk| at k = 0", color=WHITE, fontsize=13)
    ax.text(0.985, 0.975, "ATM skew term structure\nmarket vs rough Bergomi vs Heston",
            transform=ax.transAxes, color=WHITE, fontsize=12.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=GRID, lw=0.8))
    sec = ax.secondary_xaxis("top", functions=(lambda t: t * TRADING_DAYS,
                                               lambda d: d / TRADING_DAYS))
    sec.tick_params(colors=WHITE, labelsize=11, which="both")
    sec.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 126]))
    sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    sec.xaxis.set_minor_formatter(NullFormatter())
    sec.set_xlabel("trading days (T x 252)", color=WHITE, fontsize=11)
    sec.spines["top"].set_color(GRID)
    ylo, yhi = ax.get_ylim()
    cands = [c * 10.0 ** e for e in range(-3, 2) for c in (1, 2, 3, 5, 7)]
    ax.yaxis.set_major_locator(FixedLocator([c for c in cands if ylo <= c <= yhi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    leg = ax.legend(loc="lower left", fontsize=9.5, facecolor=BG, edgecolor=GRID,
                    labelcolor=WHITE, framealpha=0.92)
    leg.get_frame().set_linewidth(0.8)

    cal = payload["calibration"]
    n_caps = len(cal["captures"])
    rm = cal["summary"]
    fig.suptitle("Heston (COS) as the classical reference: calibrated smiles and the short-dated skew",
                 color=WHITE, fontsize=18, y=0.985)
    rmse_bits = [f"Heston LS {rm['heston_LS_mean']:.3f} vp"]
    if "heston_LS-kappa10_mean" in rm:
        rmse_bits.append(f"kappa<=10 {rm['heston_LS-kappa10_mean']:.3f} vp")
    if "heston_LS-kappa3_mean" in rm:
        rmse_bits.append(f"kappa<=3 {rm['heston_LS-kappa3_mean']:.3f} vp")
    rmse_bits.append(f"rough Bergomi map refit {rm['map_refit_mean']:.3f} vp "
                     f"(recorded in intraday_params.json {rm['map_recorded_mean']:.3f} vp)")
    slope_bits = [f"Heston LS {rm['heston_b_mean']:+.3f} (sd {rm['heston_b_sd']:.3f} across captures)"]
    if "heston_kappa10_b_mean" in rm:
        slope_bits.append(f"kappa<=10 {rm['heston_kappa10_b_mean']:+.3f}")
    if "heston_kappa3_b_mean" in rm:
        slope_bits.append(f"kappa<=3 {rm['heston_kappa3_b_mean']:+.3f}")
    slope_bits.append(f"market {mk['fit']['b']:+.3f} +- {mk['fit']['se_b']:.3f}")
    slope_bits.append(f"rough Bergomi {rb['fit_short']['b']:+.3f} +- {rb['fit_short']['se_b']:.3f}")
    sub = textwrap.fill(
        f"Left: SPY capture {sm['pricing_time'][:16]} NY, three of its {sm['n_expiries']} "
        f"expiries, market mid IVs vs the calibrated Heston smile. Right: Heston ATM skew from each of "
        f"{n_caps} calibrated captures on the maturity ladder of docs/atm_skew_term_structure.json, "
        f"against that document's market rows and rough Bergomi curve. "
        f"Smile RMSE over {n_caps} captures: {'; '.join(rmse_bits)}. "
        f"Power-law exponent, T <= 45 d: {'; '.join(slope_bits)}.", width=150)
    fig.text(0.5, 0.952, sub, color=GREY, fontsize=10, ha="center", va="top", linespacing=1.3)
    fig.savefig(out_png, dpi=100, facecolor=BG)
    plt.close(fig)


def smile_panels(path: Path, pricer: MapPricer, fit: dict, rec: dict | None,
                 expiries: list[str]) -> dict:
    quotes_all, rate, meta = quotes_from_capture(path)
    quotes, _ = pricer.usable(quotes_all)
    p = fit["params"]
    per_exp_h = {r["expiry"]: r["rmse_volpts"] for r in fit["per_expiry"]}
    panels = []
    for e in expiries:
        qs = [q for q in quotes if q.expiry == e]
        tau = float(np.median([q.tau for q in qs]))
        fwd_pv = float(np.median([q.fwd_pv for q in qs]))
        F = fwd_pv * math.exp(rate * tau)
        k = np.array([math.log(q.strike / F) for q in qs])
        iv = np.array([q.iv for q in qs])
        hs = np.array([q.half_spread_iv for q in qs])
        kg = np.linspace(k.min(), k.max(), 121)
        h_iv = H.heston_implied_vol(F, F * np.exp(kg), tau, 0.0, 0.0, N=512, **p)
        m_iv, m_rmse = None, None
        if rec is not None:
            grid_q = [Quote(tau=tau, strike=float(F * math.exp(kk)), mid_call=0.0, iv=0.0,
                            vega=1.0, kind="C", expiry=e, fwd_pv=fwd_pv) for kk in kg]
            grid_q, _ = pricer.usable(grid_q)
            kk_ok = np.array([math.log(q.strike / F) for q in grid_q])
            m_iv = np.interp(kg, kk_ok, map_ivs_for(pricer, grid_q, rec), left=np.nan, right=np.nan)
            m_rmse = float(np.sqrt(np.mean(((map_ivs_for(pricer, qs, rec) - iv) * 100) ** 2)))
        panels.append({"expiry": e, "tau": tau, "n": len(qs), "k": k.tolist(), "iv": iv.tolist(),
                       "half_spread_iv": hs.tolist(), "k_grid": kg.tolist(),
                       "heston_iv": h_iv.tolist(), "heston_rmse": per_exp_h[e],
                       "map_iv": None if m_iv is None else m_iv.tolist(), "map_rmse": m_rmse})
    return {"capture": path.name, "pricing_time": meta.get("pricing_time"),
            "n_expiries": len({q.expiry for q in quotes}), "panels": panels}


# ── driver ─────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", nargs="*", default=None)
    p.add_argument("--N", type=int, default=256,
                   help="COS terms for calibration (converged to <1e-5 vp at 256; "
                        "the range self-check raises it where the tails demand)")
    p.add_argument("--random-starts", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=DOCS)
    args = p.parse_args(argv)
    t_all = time.perf_counter()

    log("[verify] pricer checks")
    ver = verification()
    log(f"  FO2008: L=12 N=2^10 -> {[r for r in ver['fo2008_vs_N'] if r['N'] == 1024][0]['call_L12']:.10f} "
        f"(ref {FO_REF}); range-converged {ver['fo2008_converged']:.10f}; "
        f"parity {ver['parity_max_abs_err']:.1e}; BS limit {ver['bs_limit']}")
    for m in ver["monte_carlo"]:
        log(f"  MC T={m['T']} {m['n_steps']} steps: z = {np.round(m['z'], 2)} ({m['seconds']:.1f}s)")
    log(f"  {ver['ms_per_100_strike_smile_N256']:.2f} ms per 100-strike smile at N=256, "
        f"{ver['ms_per_100_strike_iv_N256']:.2f} ms with IVs; verification {ver['seconds']:.0f}s")

    caps = [DATA / c for c in (args.captures or DEFAULT_CAPTURES)]
    variants = ("LS",) if args.quick else ("LS", "Huber", "LS-hs", "LS-kappa10", "LS-kappa3")
    if args.quick:
        caps = caps[:2]
    intraday = load_intraday()
    pricer = MapPricer()
    log(f"[calibrate] {len(caps)} captures, variants {variants}, N={args.N}")
    captures = []
    for i, c in enumerate(caps):
        rec = intraday.get(c.name)
        var = variants + (("LS-kappa1000",) if (i == 0 and not args.quick) else ())
        captures.append(calibrate_capture(c, pricer, rec, var, args.random_starts, args.N))

    def collect(key, fn):
        return [fn(cp) for cp in captures if fn(cp) is not None]

    summary = {}
    for v in variants:
        vals = np.array([cp["fits"][v]["rmse_volpts"] for cp in captures])
        summary[f"heston_{v}_mean"] = float(vals.mean())
        summary[f"heston_{v}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    with_rec = [cp for cp in captures if cp["map"]["recorded"]]
    rec_vals = np.array([cp["map"]["recorded"]["rmse_volpts"] for cp in with_rec])
    rc_vals = np.array([cp["map"]["rmse_recomputed_volpts"] for cp in with_rec])
    rf_vals = np.array([cp["map"]["refit"]["rmse_volpts"] for cp in captures])
    summary["map_recorded_mean"] = float(rec_vals.mean()) if len(rec_vals) else float("nan")
    summary["map_recomputed_mean"] = float(rc_vals.mean()) if len(rc_vals) else float("nan")
    summary["map_refit_mean"] = float(rf_vals.mean())
    summary["map_refit_sd"] = float(rf_vals.std(ddof=1)) if len(rf_vals) > 1 else float("nan")
    ls = np.array([cp["fits"]["LS"]["rmse_volpts"] for cp in with_rec])
    ls_all = np.array([cp["fits"]["LS"]["rmse_volpts"] for cp in captures])
    summary["heston_LS_minus_map_recorded_mean"] = float((ls - rec_vals).mean()) if len(rec_vals) else float("nan")
    summary["heston_LS_minus_map_recorded_sd"] = float((ls - rec_vals).std(ddof=1)) if len(rec_vals) > 1 else float("nan")
    summary["heston_LS_minus_map_refit_mean"] = float((ls_all - rf_vals).mean())
    summary["heston_LS_minus_map_refit_sd"] = float((ls_all - rf_vals).std(ddof=1)) if len(rf_vals) > 1 else float("nan")
    for v in ("LS-kappa10", "LS-kappa3"):
        if v in variants:
            kv = np.array([cp["fits"][v]["rmse_volpts"] for cp in captures])
            summary[f"heston_{v[3:]}_minus_map_refit_mean"] = float((kv - rf_vals).mean())
            summary[f"heston_{v[3:]}_minus_map_refit_sd"] = float((kv - rf_vals).std(ddof=1)) if len(rf_vals) > 1 else float("nan")
    summary["captures_pinned"] = {cp["info"]["capture"]: cp["fits"]["LS"]["pinned"] for cp in captures}
    summary["seconds_per_LS_fit_mean"] = float(np.mean([cp["fits"]["LS"]["seconds"] for cp in captures]))

    # ── skew ──────────────────────────────────────────────────────────────
    log("[skew] Heston ATM skew on the ladder of docs/atm_skew_term_structure.json")
    ref = json.loads((DOCS / "atm_skew_term_structure.json").read_text(encoding="utf-8"))
    T_days = ref["protocol"]["T_days"]
    spy = ref["spy"]
    market_per_capture = {c["capture"]: c for c in spy["market_per_capture"]}
    per_capture = []
    for cp in captures:
        for v in ("LS", "Huber", "LS-kappa10", "LS-kappa3"):
            if v not in cp["fits"]:
                continue
            prm = cp["fits"][v]["params"]
            ladder = heston_skew_ladder(prm, T_days)
            fits = skew_fits(ladder, prm)
            win = heston_at_capture_expiries(prm, cp["info"]["expiries"],
                                             cp["info"]["expiry_taus"], spy["market_rows"],
                                             cp["info"]["capture"])
            blk = {"capture": cp["info"]["capture"], "variant": v, "params": prm,
                   "ladder": ladder, "fits": fits, "capture_window": win,
                   "market_fit_same_capture": market_per_capture.get(cp["info"]["capture"])}
            per_capture.append(blk)
            f = fits["fine_short"]
            fw = win["fit_capture_window"]
            mk_same = blk["market_fit_same_capture"]
            log(f"  {cp['info']['capture'][4:19]} {v:<10} psi(1d)={ladder[0]['psi_fine']:+.3f} "
                f"psi(45d)={[r for r in ladder if r['T_days'] == 45][0]['psi_fine']:+.3f}  "
                f"slope T<=45d {f['b']:+.4f} +- {f['se_b']:.4f}; over the capture's own "
                f"expiries {fw['b']:+.4f}"
                + (f" (market on the same capture {mk_same['b']:+.4f} +- {mk_same['se_b']:.4f})"
                   if mk_same else "")
                + "; local " + " ".join(f"[{w['T_lo_days']}-{w['T_hi_days']}d]{w['b']:+.3f}"
                                        for w in fits["local_fine"])
                + f"; T->0 limit {fits['short_limit_analytic']:+.3f}; 1/kappa = "
                  f"{fits['one_over_kappa_days']:.1f} trading days")

    def variant_summary(v: str) -> dict:
        blocks = [b for b in per_capture if b["variant"] == v]
        if not blocks:
            return {}
        bs_ = np.array([b["fits"]["fine_short"]["b"] for b in blocks])
        bw = np.array([b["capture_window"]["fit_capture_window"]["b"] for b in blocks])
        levels = [r for b in blocks for r in b["capture_window"]["rows"] if "market_psi" in r]
        return {
            "b_mean": float(bs_.mean()),
            "b_sd": float(bs_.std(ddof=1)) if len(bs_) > 1 else float("nan"),
            "b_per_capture": [{"capture": b["capture"], "b": b["fits"]["fine_short"]["b"],
                               "se_b": b["fits"]["fine_short"]["se_b"],
                               "b_capture_window": b["capture_window"]["fit_capture_window"]["b"],
                               "market_b_same_capture": (b["market_fit_same_capture"] or {}).get("b"),
                               "market_se_b_same_capture": (b["market_fit_same_capture"] or {}).get("se_b")}
                              for b in blocks],
            "b_capture_window_mean": float(bw.mean()),
            "b_capture_window_sd": float(bw.std(ddof=1)) if len(bw) > 1 else float("nan"),
            "psi_1d_mean": float(np.mean([b["ladder"][0]["psi_fine"] for b in blocks])),
            "psi_45d_mean": float(np.mean([[r for r in b["ladder"] if r["T_days"] == 45][0]["psi_fine"]
                                           for b in blocks])),
            "short_limit_mean": float(np.mean([b["fits"]["short_limit_analytic"] for b in blocks])),
            "level_vs_market_rms_sigmas": (float(np.sqrt(np.mean([r["diff_sigmas"] ** 2 for r in levels])))
                                           if levels else float("nan")),
            "level_vs_market_mean_ratio": (float(np.mean([r["heston_psi"] / r["market_psi"] for r in levels]))
                                           if levels else float("nan")),
            "n_level_rows": len(levels),
        }

    # ── kappa profile on the first capture ────────────────────────────────
    prof_kappas = (3.0, 100.0) if args.quick else KAPPA_PROFILE
    log(f"[kappa profile] {caps[0].name}: refit (v0, theta, sigma_v, rho) at fixed kappa")
    t0 = time.perf_counter()
    profile = kappa_profile(caps[0], pricer, args.N, T_days, spy["market_rows"], prof_kappas)
    profile_seconds = time.perf_counter() - t0

    mk_fit = spy["fits"]["market_pooled"]
    rb_fit = spy["fits"]["model_short"]
    variants_sum = {v: variant_summary(v) for v in ("LS", "Huber", "LS-kappa10", "LS-kappa3")}
    summary["heston_b_mean"] = variants_sum["LS"]["b_mean"]
    summary["heston_b_sd"] = variants_sum["LS"]["b_sd"]
    if variants_sum["LS-kappa10"]:
        summary["heston_kappa10_b_mean"] = variants_sum["LS-kappa10"]["b_mean"]
        summary["heston_kappa10_b_sd"] = variants_sum["LS-kappa10"]["b_sd"]
    if variants_sum["LS-kappa3"]:
        summary["heston_kappa3_b_mean"] = variants_sum["LS-kappa3"]["b_mean"]
        summary["heston_kappa3_b_sd"] = variants_sum["LS-kappa3"]["b_sd"]

    def sigmas(b_mean, b_sd, n, ref_b, ref_se):
        return (b_mean - ref_b) / math.hypot(ref_se, b_sd / math.sqrt(max(n, 1)))

    n_ls = len(variants_sum["LS"]["b_per_capture"])
    sk_sum = {
        "variants": variants_sum,
        "market_b": mk_fit["b"], "market_se_b": mk_fit["se_b"],
        "rough_bergomi_b": rb_fit["b"], "rough_bergomi_se_b": rb_fit["se_b"],
        "reference_slope_H_minus_half": spy["reference_slope"],
        "heston_LS_minus_market_sigmas": sigmas(
            variants_sum["LS"]["b_mean"], variants_sum["LS"]["b_sd"], n_ls, mk_fit["b"], mk_fit["se_b"]),
        "heston_LS_minus_rough_sigmas": sigmas(
            variants_sum["LS"]["b_mean"], variants_sum["LS"]["b_sd"], n_ls, rb_fit["b"], rb_fit["se_b"]),
        "heston_LS_window_minus_market_sigmas": sigmas(
            variants_sum["LS"]["b_capture_window_mean"], variants_sum["LS"]["b_capture_window_sd"],
            n_ls, mk_fit["b"], mk_fit["se_b"]),
    }
    for v in ("LS-kappa10", "LS-kappa3"):
        if variants_sum[v]:
            sk_sum[f"heston_{v[3:]}_minus_market_sigmas"] = sigmas(
                variants_sum[v]["b_mean"], variants_sum[v]["b_sd"],
                len(variants_sum[v]["b_per_capture"]), mk_fit["b"], mk_fit["se_b"])
    log(f"  Heston LS slope mean {variants_sum['LS']['b_mean']:+.4f} sd {variants_sum['LS']['b_sd']:.4f} "
        f"(over the captures' own expiries {variants_sum['LS']['b_capture_window_mean']:+.4f}); "
        f"market {mk_fit['b']:+.4f} +- {mk_fit['se_b']:.4f}; rough Bergomi {rb_fit['b']:+.4f} +- "
        f"{rb_fit['se_b']:.4f}; H - 1/2 = {spy['reference_slope']:+.4f}; Heston LS - market = "
        f"{sk_sum['heston_LS_minus_market_sigmas']:+.1f} sigma"
        + (f"; Heston kappa<=10 slope {variants_sum['LS-kappa10']['b_mean']:+.4f} sd "
           f"{variants_sum['LS-kappa10']['b_sd']:.4f}" if variants_sum["LS-kappa10"] else ""))

    # ── smiles for the figure ─────────────────────────────────────────────
    first = captures[0]
    exps = first["info"]["expiries"]
    chosen = [exps[0], exps[len(exps) // 2], exps[-1]] if len(exps) >= 3 else exps
    smiles = smile_panels(caps[0], pricer, first["fits"]["LS"], first["map"]["refit"], chosen)

    total = time.perf_counter() - t_all
    payload = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {"captures": [c.name for c in caps], "variants": list(variants),
                     "N_calibration": args.N, "L": H.DEFAULT_L, "range_tol": H.RANGE_TOL,
                     "bounds": H.HESTON_BOUNDS, "random_starts": args.random_starts,
                     "T_days": T_days, "short_fit_max_days": ats.SHORT_FIT_MAX_DAYS,
                     "skew_stencil": "fine h = 1e-4; document stencil h = max(0.25 atm_iv sqrt(T), 0.002)",
                     "quote_set": "MapPricer.usable (the map's (k, tau) box)",
                     "pricing": "on the per-expiry forward F = fwd_pv e^{r tau}, r = q = 0, undiscounted"},
        "verification": ver,
        "calibration": {"captures": captures, "summary": summary},
        "skew": {"heston": {"per_capture": per_capture}, "summary": sk_sum,
                 "kappa_profile": {"capture": caps[0].name, "kappas": list(prof_kappas),
                                   "rows": profile, "seconds": profile_seconds},
                 "market": {"rows": spy["market_rows"], "fit": mk_fit,
                            "captures": [c["capture"] for c in spy["captures"]]},
                 "rough_bergomi": {"model": [{k: r[k] for k in ("T_days", "T", "psi", "se", "psi_richardson", "atm_iv")}
                                             for r in spy["model"]],
                                   "fit_short": rb_fit, "H": spy["params"]["H"],
                                   "params": spy["params"], "reference_slope": spy["reference_slope"]}},
        "smiles": smiles,
        "seconds_total": total,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "heston_reference" + ("_quick" if args.quick else "")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=1, default=_json_default),
                                               encoding="utf-8")
    make_figure(payload, args.out_dir / f"{stem}.png")
    log(f"wrote {args.out_dir / (stem + '.json')} and {args.out_dir / (stem + '.png')} in {total:.0f}s")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        _out = DOCS / "heston_reference"
        make_figure(json.loads(_out.with_suffix(".json").read_text(encoding="utf-8")),
                    _out.with_suffix(".png"))
        print(f"redrew {_out.with_suffix('.png')}")
        sys.exit(0)
    sys.exit(main())
