"""Fit every recorded surface of a session: the parameter time series.

The recorder banks a surface every 15 minutes; the map calibrates one in ~20s
of CPU. Composed, the archive becomes something the MC engine could never
afford: rough-Bergomi parameters as a TIME SERIES through the trading day.
Ten days apart, accepted fits moved eta 2.69 -> 3.66 and H 0.104 -> 0.255;
this measures how much of such drift happens WITHIN a day — is the surface's
parameterization stable for hours, or is a morning fit stale by lunch?

    python -m scripts.intraday_params --glob "data/surfaces/equity/spy_2026082*"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.calibrate_map import MapCalibrator, MapPricer, quotes_from_capture


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="data/surfaces/equity/spy_*.json.gz")
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts" / "intraday_params.json")
    args = p.parse_args()

    rows = []
    print(f"  {'capture (UTC)':>22} {'stale':>5} {'n':>4} {'eta':>7} "
          f"{'rho':>7} {'H':>7} {'sqrt_xi':>8} {'RMSE':>6} {'s':>5}")
    pricer = MapPricer()
    for path in sorted(glob.glob(str(ROOT / args.glob))):
        path = Path(path)
        try:
            quotes, rate, meta = quotes_from_capture(path)
            cal = MapCalibrator(quotes, market="SPY", pricer=pricer)
            theta, secs = cal.fit()
            eta, rho, H, xi = map(float, theta)
            rmse = cal.rmse_volpts(theta)
        except Exception as exc:
            print(f"  {path.name:>22}  FAILED: {exc}")
            continue
        stamp = path.stem.split("_")[1].replace(".json", "")
        rows.append(dict(capture=path.name, stamp=stamp,
                         stale=bool(meta.get("stale")),
                         n=len(cal.quotes), eta=round(eta, 4),
                         rho=round(rho, 4), H=round(H, 4),
                         sqrt_xi=round(math.sqrt(xi), 4),
                         rmse_volpts=round(rmse, 3), seconds=round(secs, 1)))
        r = rows[-1]
        print(f"  {stamp:>22} {str(r['stale'])[0]:>5} {r['n']:>4} "
              f"{r['eta']:>7.3f} {r['rho']:>7.3f} {r['H']:>7.3f} "
              f"{r['sqrt_xi']:>8.2%} {r['rmse_volpts']:>6.3f} "
              f"{r['seconds']:>5.1f}", flush=True)

    live = [r for r in rows if not r["stale"]]
    if len(live) >= 3:
        for k in ("eta", "rho", "H", "sqrt_xi"):
            v = np.array([r[k] for r in live])
            print(f"  live-session {k:>7}: mean {v.mean():+.4f}  "
                  f"range [{v.min():+.4f}, {v.max():+.4f}]  sd {v.std(ddof=1):.4f}")
    args.out.write_text(json.dumps(
        {"engine": "pricing_map v3 (CPU)", "rows": rows}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
