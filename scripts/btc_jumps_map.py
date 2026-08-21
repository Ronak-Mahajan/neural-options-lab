"""The BTC jump question, asked properly: deterministic search, MC verdict.

The MC-based experiment could not answer whether jumps help BTC: with a
frozen-path noise floor of ~0.5 on the objective and 120 quotes, the
7-parameter arm won in-search and lost under independent repricing --
overfitting to the draw, not evidence about jumps. The map removes the
mechanism: its objective is deterministic, so a parameter only lowers the
loss by fitting the SURFACE.

Both arms are searched on the map (CPU, generous budget), then the VERDICT is
rendered by the true model: both thetas repriced under Monte Carlo at 200k
paths on the GPU, bucketed by standardised moneyness. The map proposes; MC
disposes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.calibrate import Calibrator, implied_vol, iv_fit_report
from backend.quant.calibrate_map import MapCalibrator, fetch_live

D_BANDS = [(-3.0, -2.0), (-2.0, -1.0), (-1.0, -0.25), (-0.25, 0.25),
           (0.25, 1.0), (1.0, 3.0)]


def mc_verdict(cal_mc, quotes, rate, theta, jumps):
    cal_mc.jumps = tuple(map(float, theta[4:7])) if jumps else None
    model = cal_mc.model_prices(*map(float, theta[:4]), n_paths=200_000,
                                seed=20260821)
    rep = iv_fit_report(cal_mc.quotes, model, rate)
    rows = []
    for q, p in zip(cal_mc.quotes, model):
        iv_m = implied_vol(float(p), q.fwd_pv, q.strike, q.tau, rate)
        if iv_m is not None:
            rows.append((math.log(q.strike / q.fwd_pv)
                         / (q.iv * math.sqrt(q.tau)),
                         (iv_m - q.iv) * 100.0))
    return rep["rmse_volpts"], np.array(rows)


def main() -> None:
    quotes, rate, meta = fetch_live("BTC")
    print(f"live BTC {meta['as_of']}  index {meta.get('index_price'):,.0f}  "
          f"{len(quotes)} quotes", flush=True)

    cal_d = MapCalibrator(quotes, market="BTC", jumps=False)
    th_d, s_d = cal_d.fit()
    print(f"map diffusive ({s_d:.0f}s): eta={th_d[0]:.3f} rho={th_d[1]:.3f} "
          f"H={th_d[2]:.3f} sqrt(xi)={math.sqrt(th_d[3]):.2%}  "
          f"map-RMSE {cal_d.rmse_volpts(th_d):.3f} vp", flush=True)

    cal_j = MapCalibrator(quotes, market="BTC", jumps=True)
    th_j, s_j = cal_j.fit(warm_start=th_d)
    print(f"map jumps     ({s_j:.0f}s): eta={th_j[0]:.3f} rho={th_j[1]:.3f} "
          f"H={th_j[2]:.3f} sqrt(xi)={math.sqrt(th_j[3]):.2%}  "
          f"lam={th_j[4]:.1f} mu_j={th_j[5]:+.4f} sig_j={th_j[6]:.4f}  "
          f"map-RMSE {cal_j.rmse_volpts(th_j):.3f} vp", flush=True)

    # ---- the verdict: same quotes, true model, independent seed ---------- #
    cal_mc = Calibrator(rate, cal_d.quotes, n_paths=200_000)
    out = {}
    for tag, th, jumps in (("diffusive", th_d, False), ("jumps", th_j, True)):
        rmse, rows = mc_verdict(cal_mc, cal_d.quotes, rate, th, jumps)
        out[tag] = (rmse, rows)
        print(f"\nMC verdict, {tag}: RMSE {rmse:.3f} vp", flush=True)
        d, e = rows[:, 0], rows[:, 1]
        for lo, hi in D_BANDS:
            m = (d >= lo) & (d < hi)
            if m.sum() >= 4:
                print(f"    {lo:+.2f}..{hi:+.2f}  n={m.sum():>3}  "
                      f"mean {e[m].mean():+7.3f}  "
                      f"rmse {math.sqrt(np.mean(e[m]**2)):6.3f}")
    gain = out["diffusive"][0] - out["jumps"][0]
    print(f"\njumps vs diffusive under MC: {gain:+.3f} vp "
          f"({'jumps HELP' if gain > 0.15 else 'jumps do not help beyond noise'})")


if __name__ == "__main__":
    main()
