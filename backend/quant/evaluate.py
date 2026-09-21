"""Measure true surrogate error against high-precision Monte Carlo references.

Validation RMSE during training is computed against *noisy* MC labels, so it
overstates the model's real error. This script draws an independent LHS test
set and, for each point, runs a 200k-path Monte Carlo that produces four
references:

    price  - control-variate estimator (SE well under 1 bp over most of box)
    delta  - pathwise estimator of dPrice/dm
    vega   - pathwise estimator of dPrice/dsigma
    gamma  - conditional-density estimator of d2Price/dm2 (gamma_reference):
             the average is linear in the spot, so gamma is the discounted
             density of the scaled average at the strike, and conditioning
             on the first increment gives that density in closed form

It then reports signed errors for the first ensemble member alone ("single
model") and the full ensemble average, for all four quantities. Everything
is expressed in 1e-4 units of the quantity (price: bps of strike; delta,
vega and gamma: x10^-4 at unit strike), cached to artifacts/eval.json, and
served to the dashboard's error-distribution chart. The gamma reference is
itself a Monte Carlo estimate; its RMS standard error is recorded alongside
so the reported gamma error can be read against the noise floor.

The report records the SHA-256 of the checkpoint it measured, so a report and
the model it describes can never quietly come apart; the regression suite
checks the two against each other.

Usage (from the repo root, after training):
    python -m backend.quant.evaluate                # 600 points, ~25 min on 8 cores
    python -m backend.quant.evaluate --points 1000 --ref-paths 400000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from .dataset import PARAM_RANGES, _simulate_chunk
from .engine import ARTIFACTS, PricingEngine
from .gamma_reference import gamma_conditional

#: The checkpoint PricingEngine() loads by default, and the one every number in
#: this report describes: PARAM_RANGES starts at 0.05 years, above the 12/252
#: cutoff below which the engine routes to the 0DTE surrogate instead, so no
#: test point here is priced by model_0dte.pt.
SERVED_CHECKPOINT = ARTIFACTS / "model.pt"


def checkpoint_fingerprint(path: Path) -> dict:
    """Byte identity of the checkpoint a report was measured against.

    Without it a report cannot be told apart from a stale one. That is not
    hypothetical here: eval.json was committed once, model.pt was retrained
    and promoted four weeks later, and the dashboard went on quoting the
    retired head's error because nothing in either file could contradict the
    other. The hash ties them together, and
    tests/test_regression.py::test_eval_report_matches_the_served_checkpoint
    turns the drift into a failing test instead of a slide nobody can defend.

    Hashing the bytes rather than reading a git stamp is deliberate: the
    container that serves the site carries the artifacts but no .git, and a
    checkpoint retrained in place never moves the git stamp at all.
    """
    data = path.read_bytes()
    return {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


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
          f"{args.ref_paths:,}-path Monte Carlo (price + pathwise Greeks, "
          f"conditional-density gamma)...")
    ref = np.empty((args.points, 4))               # price, delta, vega, gamma
    gamma_ref_se = np.empty(args.points)
    t0 = time.perf_counter()
    for i in range(args.points):
        rng = np.random.default_rng(10_000 + i)
        price, delta, vega = _simulate_chunk(X[i:i + 1], args.ref_paths,
                                             engine.n_steps, rng)
        # Gamma has no pathwise estimator; it is the discounted density of
        # the scaled average at the strike, estimated by conditioning on the
        # first increment (see gamma_reference). Its own seed keeps it
        # reproducible independently of the pathwise draw above.
        m, mat, sig, r = (float(v) for v in X[i])
        g = gamma_conditional(m, mat, sig, r, engine.n_steps,
                              n_paths=args.ref_paths, seed=20_000 + i)
        ref[i] = price[0], delta[0], vega[0], g["gamma"]
        gamma_ref_se[i] = g["se"]
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
        gammas = np.empty(args.points)
        for i, (m, mat, sig, r) in enumerate(X):
            out = engine.price_with_greeks(float(m), 1.0, float(mat),
                                           float(sig), float(r), "call",
                                           member=member)
            deltas[i] = out["greeks"]["delta"]
            vegas[i] = out["greeks"]["vega"] * 100.0   # back to per unit vol
            # At unit strike the engine's gamma is d2(C/K)/dm2 exactly, the
            # quantity the conditional-density reference estimates.
            gammas[i] = out["greeks"]["gamma"]
        pred[name]["delta"] = deltas
        pred[name]["vega"] = vegas
        pred[name]["gamma"] = gammas

    metrics = ("price", "delta", "vega", "gamma")
    errors = {met: {name: pred[name][met] - ref[:, j]
                    for name in ("single", "ensemble")}
              for j, met in enumerate(metrics)}

    report = {
        "n_points": args.points,
        "ref_paths": args.ref_paths,
        "n_members": engine.n_members,
        "checkpoint": checkpoint_fingerprint(SERVED_CHECKPOINT),
        # What the checkpoint records about its own training, not a guess.
        # The served checkpoint was promoted out of scripts/fullscale_ablation.py,
        # whose meta block omits this key although both of its arms optimise
        # dml_loss (price MSE + pathwise delta and vega MSE, lam = 1.0). None
        # therefore means "the checkpoint does not say"; defaulting it to False
        # would publish a claim about the training recipe that the training
        # script contradicts.
        "differential_ml": (bool(engine.meta["differential_ml"])
                            if "differential_ml" in engine.meta else None),
        "training_arm": engine.meta.get("arm"),
        # The gamma reference is itself a Monte Carlo estimate; this is the
        # RMS of its per-point standard error, in the same 1e-4 units as the
        # gamma error statistics, so a reader can see how much of the
        # reported gamma error is the reference's own noise.
        "gamma_reference_se_rms_bps": float(
            np.sqrt(np.mean((gamma_ref_se * 1e4) ** 2))),
        "single": {met: summarize(errors[met]["single"]) for met in metrics},
        "ensemble": {met: summarize(errors[met]["ensemble"])
                     for met in metrics},
        "errors": {met: {name: np.round(errors[met][name] * 1e4, 3).tolist()
                         for name in ("single", "ensemble")}
                   for met in metrics},
        # Per-point gamma reference and its standard error, at unit strike, so
        # any relative-error or noise-floor statistic quoted for gamma can be
        # recomputed from this file.
        "gamma_ref": np.round(ref[:, 3], 6).tolist(),
        "gamma_ref_se": np.round(gamma_ref_se, 6).tolist(),
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
    fp = report["checkpoint"]
    print(f"\ncheckpoint {fp['file']}  sha256 {fp['sha256'][:16]}...")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
