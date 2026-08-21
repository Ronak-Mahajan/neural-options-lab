"""Certify the map's maturity extension: live 60-day SPY, MC as referee.

Every certification so far ran on surfaces the recorder could see: 17 days or
less, the old box. v5 trained on 120k new parameter sets spanning 15-120 days
precisely so calibration could reach weeklies and monthlies after the GPU is
gone. This is the test that earns that claim, run while the GPU still exists
to referee: fit a live max-dte-60 SPY surface on the map (CPU), reprice the
map's parameters under 100-step Monte Carlo (the discretization the long-tau
labels used), and bucket the error by maturity so the new region is judged
separately from the already-certified one.

Also answers this morning's finding: the 10:00 scheduled recalibration railed
eta at the SPY box ceiling (3.935 on [0.5, 4.0]) and was rightly rejected.
The map trained on eta up to 8, so the same surface is fitted twice, ceiling
4 and ceiling 8, to see whether eta settles at an interior optimum when the
box stops binding.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from backend.quant.calibrate import (MIN_TAU_HOURS, fetch_calibration_set,
                                     implied_vol)
from backend.quant.calibrate_map import MapCalibrator, MapPricer
from backend.quant.rough_vol import rough_bergomi_mc

MC_PATHS, MC_STEPS, SEED = 200_000, 100, 20260821


def mc_reprice(quotes, rate, eta, rho, H, xi):
    """MC IVs for the quotes at the given parameters, grouped per expiry."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    groups: dict[str, list] = {}
    for q in quotes:
        groups.setdefault(q.expiry, []).append(q)
    out = {}
    for exp, qs in groups.items():
        b = len(qs)
        t = lambda v: torch.full((b,), float(v), device=dev)
        prices = rough_bergomi_mc(
            t(qs[0].fwd_pv),
            torch.tensor([q.strike for q in qs], dtype=torch.float32,
                         device=dev),
            t(qs[0].tau), t(xi), t(eta), t(rho), t(rate),
            n_paths=MC_PATHS, n_steps=MC_STEPS, H=H, seed=SEED,
        ).cpu().numpy()
        for q, p in zip(qs, prices):
            iv = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, rate)
            if iv is not None:
                out[id(q)] = (iv - q.iv) * 100.0
    return out


def report(tag, quotes, errs):
    print(f"\n  {tag}")
    buckets = [("<= 17d (already certified)", 0.0, 17.5 / 365),
               ("17-45d (NEW)", 17.5 / 365, 45.0 / 365),
               ("45-60d (NEW)", 45.0 / 365, 61.0 / 365)]
    for name, lo, hi in buckets:
        e = np.array([errs[id(q)] for q in quotes
                      if lo < q.tau <= hi and id(q) in errs])
        if len(e) < 4:
            continue
        print(f"    {name:>28}  n={len(e):>4}  mean {e.mean():+7.3f}  "
              f"rmse {math.sqrt(float(np.mean(e**2))):6.3f} vp")
    e = np.array(list(errs.values()))
    print(f"    {'ALL':>28}  n={len(e):>4}  rmse "
          f"{math.sqrt(float(np.mean(e**2))):6.3f} vp")


def main() -> None:
    snap = fetch_calibration_set("SPY", 60, min_tau_hours=MIN_TAU_HOURS)
    taus = sorted({round(q.tau * 365) for q in snap.quotes})
    print(f"live SPY {snap.pricing_time:%H:%M:%S}  {len(snap.quotes)} quotes  "
          f"{len(snap.expiries)} expiries  maturities {taus[0]}-{taus[-1]}d  "
          f"stale={snap.stale}")

    pricer = MapPricer()
    print(f"map tau range: {365*pricer.meta['tau_range'][0]:.1f}-"
          f"{365*pricer.meta['tau_range'][1]:.1f}d  "
          f"k range [{pricer.k_lo:.2f}, {pricer.k_hi:.2f}]")

    for eta_cap, tag in ((4.0, "eta ceiling 4.0 (production box)"),
                         (8.0, "eta ceiling 8.0 (map's trained range)")):
        cal = MapCalibrator(snap.quotes, market="SPY", pricer=pricer)
        cal.bounds[0] = (0.5, eta_cap)
        theta, secs = cal.fit(seed=7)
        eta, rho, H, xi = map(float, theta)
        pin = " <-- PINNED" if eta > eta_cap - 0.02 * (eta_cap - 0.5) else ""
        print(f"\n=== {tag} ===")
        print(f"  map fit ({secs:.0f}s CPU, {len(cal.quotes)} quotes, "
              f"{cal.n_out_of_box} outside box): eta={eta:.4f}{pin} "
              f"rho={rho:.4f} H={H:.4f} sqrt(xi)={math.sqrt(xi):.2%}  "
              f"map-RMSE {cal.rmse_volpts(theta):.3f} vp")
        errs = mc_reprice(cal.quotes, snap.rate, eta, rho, H, xi)
        report(f"MC verdict ({MC_PATHS:,} paths, {MC_STEPS} steps):",
               cal.quotes, errs)


if __name__ == "__main__":
    main()
