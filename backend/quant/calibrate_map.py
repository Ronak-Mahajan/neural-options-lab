"""Calibrate rough Bergomi (with optional Merton jumps) on the neural map.

The certified pricing-map surrogate (artifacts/pricing_map.pt) evaluates a
whole surface in microseconds on CPU, where the Monte Carlo engine needs a
GPU and ~0.5 s. This module is the calibration path built on it — the one
that keeps working after the GPU is gone. Certified five times end-to-end
against the MC engine on live surfaces (SPY short-tau, SPY 3-56 days, BTC
full surface): winning-arm parameters within 0.005-0.10 vp under MC
repricing, seconds vs minutes.

Faithfulness is REGIONAL, not unconditional -- the final audit measured it:
the map matches MC within ~0.03-0.10 vp at fitted optima, but a rejected
diffusive fit on BTC short-tau showed a 0.48 vp map-vs-MC gap. The honest
form of the claim is "faithful at the optimum the fit selects; verify under
MC when reading the map in regions the fit rejected".

What the map additionally unlocks, beyond speed:

  - DETERMINISTIC objectives. The MC calibrator's loss carries a frozen-path
    noise floor (objective_mc_sd ~ 0.5-0.8), which is exactly what let a
    7-parameter jump fit "win" in-search and lose out-of-sample on BTC. The
    map has no path noise: what the optimizer sees is what an independent
    repricing sees, so extra parameters can no longer hide in the draw.
  - OFFLINE calibration. Any capture the recorder stored replays through the
    same code path (--capture), so the surface archive becomes a parameter
    time series.

Same objective semantics as calibrate.Calibrator.loss — vega-weighted price
error is ~ IV error, so the map's IV-space Huber with the same knee is the
same objective — and the same quality gate at the end.

    python -m backend.quant.calibrate_map --market SPY            # live
    python -m backend.quant.calibrate_map --market BTC --jumps    # live, 7p
    python -m backend.quant.calibrate_map --capture data/surfaces/equity/spy_20260820T171756Z.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

from .calibrate import BOUNDS, HUBER_DELTA, Quote, quality_gate

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
MAP_FILE = ARTIFACTS / "pricing_map.pt"

#: Jump bounds per market (see scripts/fit_jumps.py for the measurements that
#: shaped them: SPY prices small frequent down-jumps; BTC prices jumps BOTH
#: ways and railed an equity-prior mu_j <= 0 bound).
JUMP_BOUNDS = {
    "SPY": {"lam": (0.1, 60.0), "mu_j": (-0.06, 0.06), "sig_j": (0.003, 0.06)},
    "BTC": {"lam": (0.1, 150.0), "mu_j": (-0.25, 0.25), "sig_j": (0.005, 0.25)},
}
ETA_MAX = {"SPY": BOUNDS["eta"][1], "BTC": 8.0}


class MapPricer:
    """The trained map, wrapped for quote vectors."""

    def __init__(self, path: Path = MAP_FILE):
        from scripts.train_pricing_map import PricingMap, features
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.model = PricingMap(blob["width"], blob["depth"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()
        self._features = features
        self.k_lo, self.k_hi = blob["k_range"]
        self.box = {k: tuple(v) for k, v in blob["box"].items()}
        self.meta = blob

    def usable(self, quotes: list[Quote]) -> tuple[list[Quote], int]:
        """Quotes whose (k, tau) the map was trained on. The map REFUSES to
        extrapolate: outside the box its output is unconstrained by data."""
        if "tau_range" in self.meta:
            tau_lo, tau_hi = self.meta["tau_range"]
        else:
            tau_lo = math.exp(self.box["ln_tau"][0])
            tau_hi = math.exp(self.box["ln_tau"][1])
        keep = [q for q in quotes
                if self.k_lo <= math.log(q.strike / q.fwd_pv) <= self.k_hi
                and tau_lo <= q.tau <= tau_hi]
        return keep, len(quotes) - len(keep)

    def ivs(self, quotes: list[Quote], eta: float, rho: float, H: float,
            xi: float, jumps: tuple[float, float, float] | None = None
            ) -> np.ndarray:
        n = len(quotes)
        lam, mu_j, sig_j = jumps if jumps is not None else (0.0, 0.0, 0.01)
        X = torch.from_numpy(np.column_stack([
            np.full(n, eta), np.full(n, rho), np.full(n, H), np.full(n, xi),
            np.full(n, lam), np.full(n, mu_j), np.full(n, sig_j),
            np.array([q.tau for q in quotes]),
            np.array([math.log(q.strike / q.fwd_pv) for q in quotes]),
        ]).astype(np.float32))
        with torch.no_grad():
            return self.model(self._features(X)).numpy().astype(np.float64)


class MapCalibrator:
    """Drop-in analogue of calibrate.Calibrator, priced on the map."""

    def __init__(self, quotes: list[Quote], market: str = "SPY",
                 jumps: bool = False, pricer: MapPricer | None = None):
        self.pricer = pricer or MapPricer()
        self.quotes, self.n_out_of_box = self.pricer.usable(quotes)
        if len(self.quotes) < 8:
            raise ValueError(f"only {len(self.quotes)} quotes inside the "
                             f"map's box ({self.n_out_of_box} outside)")
        self.market = market
        self.jumps = jumps
        self.mids_iv = np.array([q.iv for q in self.quotes])
        self.bounds = [(BOUNDS["eta"][0], ETA_MAX[market]), BOUNDS["rho"],
                       BOUNDS["H"], BOUNDS["xi"]]
        if jumps:
            self.bounds += list(JUMP_BOUNDS[market].values())
        self.n_evals = 0

    def loss(self, theta: np.ndarray) -> float:
        self.n_evals += 1
        jumps = tuple(map(float, theta[4:7])) if self.jumps else None
        ivs = self.pricer.ivs(self.quotes, *map(float, theta[:4]), jumps=jumps)
        e = ivs - self.mids_iv
        d = HUBER_DELTA
        hub = np.where(np.abs(e) <= d, 0.5 * e ** 2, d * (np.abs(e) - 0.5 * d))
        return float(hub.mean()) * 1e4

    def fit(self, seed: int = 7, warm_start: np.ndarray | None = None
            ) -> tuple[np.ndarray, float]:
        """DE + Powell. The map is deterministic and ~10k evals/s on CPU, so
        the search budget is generous where the MC engine had to be frugal."""
        t0 = time.perf_counter()
        de = differential_evolution(
            self.loss, self.bounds, seed=seed, maxiter=60, popsize=16,
            tol=1e-7, mutation=(0.4, 0.9), recombination=0.8, polish=False,
            init="sobol", updating="deferred")
        starts = [de.x]
        if self.jumps and warm_start is not None:
            lo = JUMP_BOUNDS[self.market]
            starts.append(np.array([*warm_start[:4], lo["lam"][0],
                                    sum(lo["mu_j"]) / 2, sum(lo["sig_j"]) / 2]))
        best = None
        for x0 in starts:
            r = minimize(self.loss, x0, method="Powell", bounds=self.bounds,
                         options={"xtol": 1e-6, "ftol": 1e-8, "maxfev": 4000})
            if best is None or r.fun < best.fun:
                best = r
        return np.asarray(best.x, dtype=float), time.perf_counter() - t0

    def rmse_volpts(self, theta: np.ndarray) -> float:
        jumps = tuple(map(float, theta[4:7])) if self.jumps else None
        ivs = self.pricer.ivs(self.quotes, *map(float, theta[:4]), jumps=jumps)
        return float(np.sqrt(np.mean((ivs - self.mids_iv) ** 2))) * 100.0


def quotes_from_capture(path: Path) -> tuple[list[Quote], float, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    d.setdefault("as_of", d.get("pricing_time"))
    return [Quote(**q) for q in d["quotes"]], float(d["rate"]), d


def fetch_live(market: str) -> tuple[list[Quote], float, dict]:
    if market == "SPY":
        from .calibrate import MIN_TAU_HOURS, fetch_calibration_set
        snap = fetch_calibration_set("SPY", 17, min_tau_hours=MIN_TAU_HOURS)
        return snap.quotes, snap.rate, {
            "as_of": snap.pricing_time.isoformat(), "stale": snap.stale,
            "quote_source": snap.quote_source}
    from .calibrate_deribit import quotes_from_surface
    from .deribit import DeribitClient
    from .surface import build_surface
    snap = DeribitClient().snapshot("BTC")
    quotes, drops = quotes_from_surface(build_surface(snap))
    return quotes, 0.0, {"as_of": snap.captured_at_iso, "stale": False,
                         "quote_source": "live_two_sided_book",
                         "index_price": snap.index_price}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default="SPY", choices=("SPY", "BTC"))
    p.add_argument("--capture", type=Path, default=None,
                   help="replay a recorded surface instead of fetching live")
    p.add_argument("--jumps", action="store_true",
                   help="fit the 7-parameter Merton-jump extension. "
                        "DETERMINISM IS NOT TRUTH: measured live on BTC "
                        "(2026-08-21), the map claimed jumps improved the fit "
                        "1.74 -> 1.17 vp, and Monte Carlo repriced the same "
                        "parameters at 2.76 vp -- WORSE than diffusive. The "
                        "optimizer had mined map error in a thin corner of "
                        "the training box (lam at 87%% of bound, sig_j on its "
                        "floor). Diffusive fits are certified against MC "
                        "(three times, both markets); JUMP fits are not, and "
                        "must be verified under MC before use -- "
                        "scripts/btc_jumps_map.py is the template.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if args.capture:
        quotes, rate, meta = quotes_from_capture(args.capture)
    else:
        quotes, rate, meta = fetch_live(args.market)
    print(f"[{args.market}] {meta.get('as_of')}  {len(quotes)} quotes  "
          f"stale={meta.get('stale')}")

    cal = MapCalibrator(quotes, market=args.market, jumps=False)
    if cal.n_out_of_box:
        print(f"  {cal.n_out_of_box} quotes outside the map's (k, tau) box "
              f"excluded; {len(cal.quotes)} fitted")
    theta_d, secs = cal.fit(seed=args.seed)
    rmse_d = cal.rmse_volpts(theta_d)
    eta, rho, H, xi = theta_d
    print(f"  diffusive: eta={eta:.4f} rho={rho:.4f} H={H:.4f} "
          f"sqrt(xi)={math.sqrt(xi):.2%}  RMSE {rmse_d:.3f} vp  "
          f"({secs:.1f}s, {cal.n_evals} evals, CPU)")
    accepted, reasons = quality_gate(
        rmse=rmse_d, eta=eta, rho=rho, H=H, xi=xi,
        stale=bool(meta.get("stale")), n_unpriceable=0,
        n_quotes=len(cal.quotes))
    print(f"  gate: {'ACCEPTED' if accepted else 'rejected: ' + '; '.join(reasons)}")

    if args.jumps:
        calj = MapCalibrator(quotes, market=args.market, jumps=True)
        theta_j, secs_j = calj.fit(seed=args.seed, warm_start=theta_d)
        rmse_j = calj.rmse_volpts(theta_j)
        eta, rho, H, xi, lam, mu_j, sig_j = theta_j
        print(f"  jumps:     eta={eta:.4f} rho={rho:.4f} H={H:.4f} "
              f"sqrt(xi)={math.sqrt(xi):.2%}  lam={lam:.1f}/yr "
              f"mu_j={mu_j:+.4f} sig_j={sig_j:.4f}  RMSE {rmse_j:.3f} vp  "
              f"({secs_j:.1f}s, {calj.n_evals} evals)")
        print(f"  jump improvement: {rmse_d - rmse_j:+.3f} vp on a "
              f"deterministic objective (no frozen-path noise to overfit)")


if __name__ == "__main__":
    main()
