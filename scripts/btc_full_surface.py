"""The GPU's finale: BTC's full surface, monthlies and quarterlies included.

Every BTC fit this project has run was starved: the quote filter capped
maturity at ~17 days (the old surrogate's box), leaving ~120 quotes across 5
near expiries -- too few to identify 4 parameters, let alone 7, which is how
the jump question stayed open through three instruments. Map v5 covers 120
days and strikes to ln(K/F) in [-0.7, 0.5], so for the first time the whole
liquid Deribit surface is in-box.

Fit diffusive and jump arms on the map (CPU, deterministic), then render the
verdict under 100-step 200k-path Monte Carlo on the GPU -- its last scheduled
job before the laptop goes back.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from backend.quant.calibrate import Quote
from backend.quant.calibrate_deribit import implied_vol as db_implied_vol
from backend.quant.calibrate_map import MapCalibrator, MapPricer
from backend.quant.deribit import DeribitClient
from backend.quant.rough_vol import rough_bergomi_mc
from backend.quant.surface import build_surface
from backend.quant.calibrate import implied_vol as bs_implied_vol

# Map-range boxes, superseding the old surrogate's 17-day cap.
WIDE_MAT = (1.0 / 252.0, 118.0 / 365.0)
WIDE_MONEY = (math.exp(-0.68), math.exp(0.48))
WIDE_SIGMA = (0.05, 2.0)
MC_PATHS, MC_STEPS, SEED = 200_000, 100, 20260821


def wide_quotes(surface):
    out, dropped = [], 0
    for q in surface.quotes:
        if not (q.is_clean and q.is_otm):
            dropped += 1
            continue
        m = q.strike / q.forward
        if not (WIDE_MONEY[0] <= m <= WIDE_MONEY[1]
                and WIDE_MAT[0] <= q.tenor <= WIDE_MAT[1]):
            dropped += 1
            continue
        mid = q.mid_usd
        if not math.isfinite(mid) or mid <= 0:
            dropped += 1
            continue
        mid_call = mid if q.right == "call" else mid + (q.forward - q.strike)
        if mid_call <= 0:
            dropped += 1
            continue
        iv = db_implied_vol(mid_call, q.forward, q.strike, q.tenor, "call")
        if not (math.isfinite(iv) and WIDE_SIGMA[0] <= iv <= WIDE_SIGMA[1]):
            dropped += 1
            continue
        out.append(Quote(tau=float(q.tenor), strike=float(q.strike),
                         mid_call=float(mid_call), iv=float(iv), vega=1.0,
                         kind="C" if q.right == "call" else "P",
                         expiry=q.expiry_iso, fwd_pv=float(q.forward)))
    return out, dropped


def mc_verdict(quotes, theta, jumps):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eta, rho, H, xi = map(float, theta[:4])
    jp = tuple(map(float, theta[4:7])) if jumps else None
    groups: dict[str, list] = {}
    for q in quotes:
        groups.setdefault(q.expiry, []).append(q)
    errs = {}
    for exp, qs in groups.items():
        b = len(qs)
        t = lambda v: torch.full((b,), float(v), device=dev)
        prices = rough_bergomi_mc(
            t(qs[0].fwd_pv),
            torch.tensor([q.strike for q in qs], dtype=torch.float32,
                         device=dev),
            t(qs[0].tau), t(xi), t(eta), t(rho), t(0.0),
            n_paths=MC_PATHS, n_steps=MC_STEPS, H=H, seed=SEED,
            jumps=jp).cpu().numpy()
        for q, p in zip(qs, prices):
            iv = bs_implied_vol(float(p), q.fwd_pv, q.strike, q.tau, 0.0)
            if iv is not None:
                errs[id(q)] = (iv - q.iv) * 100.0
    return errs


def buckets(quotes, errs, tag):
    print(f"  {tag}")
    for name, lo, hi in (("<= 17d", 0, 17.5 / 365), ("17-45d", 17.5 / 365, 45 / 365),
                         ("45-120d", 45 / 365, 0.34)):
        e = np.array([errs[id(q)] for q in quotes
                      if lo < q.tau <= hi and id(q) in errs])
        if len(e) >= 4:
            print(f"    {name:>8}  n={len(e):>4}  mean {e.mean():+7.3f}  "
                  f"rmse {math.sqrt(float(np.mean(e ** 2))):6.3f} vp")
    e = np.array(list(errs.values()))
    print(f"    {'ALL':>8}  n={len(e):>4}  rmse "
          f"{math.sqrt(float(np.mean(e ** 2))):6.3f} vp")


def main() -> None:
    snap = DeribitClient().snapshot("BTC")
    quotes, dropped = wide_quotes(build_surface(snap))
    exps = sorted({q.expiry for q in quotes})
    taus = sorted({round(q.tau * 365) for q in quotes})
    print(f"BTC {snap.captured_at_iso}  index {snap.index_price:,.0f}  "
          f"{len(quotes)} quotes in the WIDE box ({dropped} dropped)  "
          f"{len(exps)} expiries, {taus[0]}-{taus[-1]} days")

    pricer = MapPricer()
    cal_d = MapCalibrator(quotes, market="BTC", pricer=pricer)
    th_d, s_d = cal_d.fit()
    print(f"\ndiffusive (map, {s_d:.0f}s CPU): eta={th_d[0]:.3f} "
          f"rho={th_d[1]:.3f} H={th_d[2]:.3f} sqrt(xi)={math.sqrt(th_d[3]):.2%}"
          f"  map-RMSE {cal_d.rmse_volpts(th_d):.3f} vp")
    ed = mc_verdict(cal_d.quotes, th_d, False)
    buckets(cal_d.quotes, ed, "MC verdict:")

    cal_j = MapCalibrator(quotes, market="BTC", jumps=True, pricer=pricer)
    th_j, s_j = cal_j.fit(warm_start=th_d)
    print(f"\njumps (map, {s_j:.0f}s CPU): eta={th_j[0]:.3f} rho={th_j[1]:.3f} "
          f"H={th_j[2]:.3f} sqrt(xi)={math.sqrt(th_j[3]):.2%}  "
          f"lam={th_j[4]:.1f} mu_j={th_j[5]:+.4f} sig_j={th_j[6]:.4f}  "
          f"map-RMSE {cal_j.rmse_volpts(th_j):.3f} vp")
    for name, v, (lo, hi) in (("lam", th_j[4], cal_j.bounds[4]),
                              ("mu_j", th_j[5], cal_j.bounds[5]),
                              ("sig_j", th_j[6], cal_j.bounds[6])):
        f = (v - lo) / (hi - lo)
        if not 0.02 < f < 0.98:
            print(f"  WARNING: {name} at {f:.0%} of bound")
    ej = mc_verdict(cal_j.quotes, th_j, True)
    buckets(cal_j.quotes, ej, "MC verdict:")

    rd = math.sqrt(float(np.mean(np.array(list(ed.values())) ** 2)))
    rj = math.sqrt(float(np.mean(np.array(list(ej.values())) ** 2)))
    print(f"\njumps vs diffusive under MC on the FULL surface: {rd - rj:+.3f} vp "
          f"({'jumps HELP' if rd - rj > 0.15 else 'no help beyond noise'})")


if __name__ == "__main__":
    main()
