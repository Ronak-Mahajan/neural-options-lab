"""Certify the pricing-map surrogate against the Monte Carlo it replaces.

A held-out RMSE says the network matches MC pointwise. That is necessary and
insufficient: calibration composes thousands of map evaluations inside an
optimizer, and small correlated errors can steer a fit even when pointwise
errors look fine. So the certification is END TO END, on a real recorded
surface (the recorder's own captures):

    1. calibrate the diffusive model against the surface with the MC engine
       (GPU, the reference), and
    2. calibrate against the NETWORK instead (CPU-fast), same optimizer, same
       bounds, same quotes;
    3. compare parameters and the MC-repriced RMSE OF THE NN-FOUND PARAMETERS
       - the honest question is not "does the NN think its fit is good" but
       "are the parameters the NN finds good under the true model".

Pass criterion: the NN-found parameters, repriced under MC, land within the
MC fit's own noise floor (objective_mc_sd) of the MC fit's RMSE.

    python -m scripts.validate_pricing_map                # newest SPY capture
    python -m scripts.validate_pricing_map --capture data/surfaces/equity/spy_....json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scipy.optimize import differential_evolution, minimize

from backend.quant.calibrate import (BOUNDS, Calibrator, Quote,
                                     iv_fit_report)
from scripts.train_pricing_map import PricingMap, features

SEARCH, REPORT = 8_000, 200_000
SEED = 7


def load_capture(path: Path) -> tuple[list[Quote], float, str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    quotes = [Quote(**q) for q in d["quotes"]]
    return quotes, float(d["rate"]), d["pricing_time"]


def load_map():
    blob = torch.load(ROOT / "artifacts" / "pricing_map.pt",
                      map_location="cpu", weights_only=False)
    model = PricingMap(blob["width"], blob["depth"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob


class MapObjective:
    """The calibrator's vega-weighted Huber loss, evaluated on the network.

    Mirrors Calibrator.loss exactly (same Huber knee, same vega floor) so the
    two calibrations differ ONLY in the pricing engine.
    """

    def __init__(self, quotes: list[Quote], model: PricingMap, box: dict):
        self.model = model
        self.mids_iv = np.array([q.iv for q in quotes])
        self.k = np.array([math.log(q.strike / q.fwd_pv) for q in quotes])
        self.tau = np.array([q.tau for q in quotes])
        k_lo, k_hi = box.get("_k_range", (-0.25, 0.20))
        self.in_range = (self.k >= k_lo) & (self.k <= k_hi)
        self.n_evals = 0

    def ivs(self, eta, rho, H, xi) -> np.ndarray:
        n = len(self.k)
        X = torch.from_numpy(np.column_stack([
            np.full(n, eta), np.full(n, rho), np.full(n, H), np.full(n, xi),
            np.zeros(n), np.zeros(n), np.full(n, 0.01),   # lam=0: diffusive
            self.tau, self.k]).astype(np.float32))
        with torch.no_grad():
            return self.model(features(X)).numpy().astype(np.float64)

    def __call__(self, theta) -> float:
        eta, rho, H, xi = map(float, theta)
        self.n_evals += 1
        # IV-space Huber: identical shape to Calibrator.loss, which divides
        # price error by vega (≈ IV error) before the same knee.
        e = self.ivs(eta, rho, H, xi) - self.mids_iv
        e = e[self.in_range]
        d = 0.02
        hub = np.where(np.abs(e) <= d, 0.5 * e ** 2, d * (np.abs(e) - 0.5 * d))
        return float(hub.mean()) * 1e4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", type=Path, default=None)
    args = p.parse_args()

    cap = args.capture
    if cap is None:
        caps = sorted((ROOT / "data" / "surfaces" / "equity").glob("spy_*.json.gz"))
        # newest LIVE capture
        for c in reversed(caps):
            with gzip.open(c, "rt", encoding="utf-8") as fh:
                if not json.load(fh)["stale"]:
                    cap = c
                    break
    quotes, rate, when = load_capture(cap)
    model, blob = load_map()
    print(f"capture {cap.name} ({when}, {len(quotes)} quotes)  "
          f"map val RMSE {blob['val_rmse_volpts']:.3f} vp")

    bounds = [BOUNDS["eta"], BOUNDS["rho"], BOUNDS["H"], BOUNDS["xi"]]

    # --- reference: MC calibration -------------------------------------- #
    cal = Calibrator(rate, quotes, n_paths=SEARCH)
    t0 = time.perf_counter()
    de = differential_evolution(cal.loss, bounds, seed=SEED, maxiter=12,
                                popsize=6, tol=1e-3, mutation=(0.4, 0.9),
                                recombination=0.8, polish=False, init="sobol",
                                updating="deferred")
    r_mc = minimize(cal.loss, de.x, method="Powell", bounds=bounds,
                    options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    t_mc = time.perf_counter() - t0
    mc_theta = np.asarray(r_mc.x, dtype=float)

    # --- candidate: NN calibration -------------------------------------- #
    box = {k: tuple(v) for k, v in blob["box"].items()}
    box["_k_range"] = tuple(blob["k_range"])
    obj = MapObjective(quotes, model, box)
    t0 = time.perf_counter()
    de = differential_evolution(obj, bounds, seed=SEED, maxiter=12,
                                popsize=6, tol=1e-3, mutation=(0.4, 0.9),
                                recombination=0.8, polish=False, init="sobol",
                                updating="deferred")
    r_nn = minimize(obj, de.x, method="Powell", bounds=bounds,
                    options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 120})
    t_nn = time.perf_counter() - t0
    nn_theta = np.asarray(r_nn.x, dtype=float)

    # --- the honest comparison: reprice BOTH under MC at report paths --- #
    out = {}
    for tag, theta in (("mc", mc_theta), ("nn", nn_theta)):
        model_px = cal.model_prices(*map(float, theta), n_paths=REPORT,
                                    seed=20260820)
        rep = iv_fit_report(cal.quotes, model_px, rate)
        out[tag] = rep["rmse_volpts"]
        eta, rho, H, xi = map(float, theta)
        print(f"  {tag}: eta={eta:.4f} rho={rho:.4f} H={H:.4f} "
              f"sqrt(xi)={math.sqrt(xi):.2%}  -> MC-repriced RMSE "
              f"{rep['rmse_volpts']:.3f} vp  "
              f"({t_mc if tag == 'mc' else t_nn:.0f}s, "
              f"{cal.n_evals if tag == 'mc' else obj.n_evals} evals)")

    gap = out["nn"] - out["mc"]
    noise = cal.objective_noise(mc_theta, n_reps=3)[1]
    print(f"\nNN-found parameters cost {gap:+.3f} vp under the true model "
          f"(MC fit noise floor ~{noise:.3f})")
    print("CERTIFIED: the map can calibrate without a GPU" if gap < 0.25
          else "NOT certified: retrain with more data/epochs before relying "
               "on it")


if __name__ == "__main__":
    main()
