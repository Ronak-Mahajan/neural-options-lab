"""Measure true surrogate error against high-precision Monte Carlo references.

Validation RMSE during training is computed against *noisy* MC labels, so it
overstates the model's real error. This script draws an independent LHS test
set and, for each point, runs a 200k-path Monte Carlo that produces three
references in one pass:

    price  - control-variate estimator (SE well under 1 bp over most of box)
    delta  - pathwise estimator of dPrice/dm
    vega   - pathwise estimator of dPrice/dsigma

It then reports signed errors for the first ensemble member alone ("single
model") and the full ensemble average, for all three quantities. Everything
is expressed in 1e-4 units of the quantity (price: bps of strike; delta and
vega: x10^-4), cached to artifacts/eval.json, and served to the dashboard's
error-distribution chart.

Usage (from the repo root, after training):
    python -m backend.quant.evaluate                # 600 points, ~4 min
    python -m backend.quant.evaluate --points 1000 --ref-paths 400000
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
from scipy.stats import qmc

from .dataset import PARAM_RANGES, _simulate_chunk
from .engine import ARTIFACTS, PricingEngine


def summarize(err: np.ndarray) -> dict:
    """Error stats in 1e-4 units of the underlying quantity."""
    e = err * 1e4
    return {
        "rmse_bps": float(np.sqrt(np.mean(e ** 2))),
        "mae_bps": float(np.mean(np.abs(e))),
        "p95_abs_bps": float(np.percentile(np.abs(e), 95)),
        "max_abs_bps": float(np.max(np.abs(e))),
        "mean_bps": float(np.mean(e)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=int, default=600)
    p.add_argument("--ref-paths", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    engine = PricingEngine()
    lows = np.array([lo for lo, _ in PARAM_RANGES.values()])
    highs = np.array([hi for _, hi in PARAM_RANGES.values()])
    sampler = qmc.LatinHypercube(d=4, seed=args.seed)
    X = lows + sampler.random(args.points) * (highs - lows)

    print(f"pricing {args.points} reference points with "
          f"{args.ref_paths:,}-path Monte Carlo (price + pathwise Greeks)...")
    ref = np.empty((args.points, 3))                       # price, delta, vega
    t0 = time.perf_counter()
    for i in range(args.points):
        rng = np.random.default_rng(10_000 + i)
        price, delta, vega = _simulate_chunk(X[i:i + 1], args.ref_paths,
                                             engine.n_steps, rng)
        ref[i] = price[0], delta[0], vega[0]
        if i % 100 == 0:
            print(f"  {i:>5}/{args.points}  "
                  f"({time.perf_counter() - t0:5.1f}s)", flush=True)

    # Surrogate predictions: price via batch; Greeks via autograd per point.
    strikes = np.ones(args.points)
    common = (X[:, 0], strikes, X[:, 1], X[:, 2], X[:, 3])
    pred: dict[str, dict[str, Any]] = {"single": {}, "ensemble": {}}
    pred["single"]["price"] = engine.price_batch(*common, option_type="call",
                                                 member=0)
    pred["ensemble"]["price"] = engine.price_batch(*common,
                                                   option_type="call")
    for name, member in (("single", 0), ("ensemble", None)):
        deltas, vegas = np.empty(args.points), np.empty(args.points)
        for i, (m, mat, sig, r) in enumerate(X):
            out = engine.price_with_greeks(float(m), 1.0, float(mat),
                                           float(sig), float(r), "call",
                                           member=member)
            deltas[i] = out["greeks"]["delta"]
            vegas[i] = out["greeks"]["vega"] * 100.0   # back to per unit vol
        pred[name]["delta"] = deltas
        pred[name]["vega"] = vegas

    metrics = ("price", "delta", "vega")
    errors = {met: {name: pred[name][met] - ref[:, j]
                    for name in ("single", "ensemble")}
              for j, met in enumerate(metrics)}

    report = {
        "n_points": args.points,
        "ref_paths": args.ref_paths,
        "n_members": engine.n_members,
        "differential_ml": bool(engine.meta.get("differential_ml", False)),
        "single": {met: summarize(errors[met]["single"]) for met in metrics},
        "ensemble": {met: summarize(errors[met]["ensemble"])
                     for met in metrics},
        "errors": {met: {name: np.round(errors[met][name] * 1e4, 3).tolist()
                         for name in ("single", "ensemble")}
                   for met in metrics},
        "params": {
            "moneyness": np.round(X[:, 0], 4).tolist(),
            "maturity": np.round(X[:, 1], 4).tolist(),
            "sigma": np.round(X[:, 2], 4).tolist(),
            "rate": np.round(X[:, 3], 4).tolist(),
        },
    }
    out_file = ARTIFACTS / "eval.json"
    out_file.write_text(json.dumps(report))

    for met in metrics:
        for name in ("single", "ensemble"):
            s = report[name][met]
            print(f"{met:>6} | {name:>8}:  RMSE {s['rmse_bps']:6.2f}   "
                  f"MAE {s['mae_bps']:6.2f}   P95 |e| {s['p95_abs_bps']:6.2f}"
                  f"   max |e| {s['max_abs_bps']:7.2f}   (x1e-4 units)")
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
