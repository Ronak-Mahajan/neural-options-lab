"""Promote an ablation arm to the served checkpoint, with a gate.

`scripts/fullscale_ablation.py` trains its arms through a `Surrogate` wrapper
that owns the input normalisation and a fixed output scale, so its state dicts
are prefixed `net.` and carry `lows`, `highs` and `scale` buffers. The serving
engine loads a bare `AsianPricerNet` and does its own normalisation. This script
is the adapter, and it refuses to promote a checkpoint it has not first verified.

The gate is the point. Swapping the model a pricing service serves on the
strength of a training-time validation number is exactly how the previously
shipped model came to carry a +0.985 bp systematic bias while its README
described that bias as an irreducible label-noise floor. So promotion re-prices
a fresh Latin-hypercube test set against high-precision Monte Carlo references -
a seed used by neither training nor the ablation's own evaluation - and only
writes the checkpoint if the candidate genuinely beats the incumbent on BOTH
RMSE and |bias|.

Greeks are scored too, and this is worth knowing: the first version of this gate
tested price only and promoted a checkpoint whose delta RMSE was 5.9% WORSE than
the incumbent's (7.338e-4 -> 7.768e-4 on a paired 1,500-point set) without
noticing. Conditioning the output head helps the level and slightly hurts the
shape. Delta and vega are now reported on every run and flagged when they
regress, but they do not block by default, because whether that trade is
acceptable is a product decision - a pricing service and a hedging service
should answer differently. Pass --require-greeks to make it blocking.

Usage:
    python -m scripts.promote_model                    # dry run, report only
    python -m scripts.promote_model --write            # promote if it wins
    python -m scripts.promote_model --require-greeks   # also block on Greeks
    python -m scripts.promote_model --candidate artifacts/ablation_residual_geometric.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from backend.quant.dataset import PARAM_RANGES, N_MONITORING_STEPS
from backend.quant.engine import ARTIFACTS, PricingEngine
from backend.quant.gpu_labels import default_device, simulate_chunk_gpu


def adapt(candidate: Path) -> dict:
    """Convert a `Surrogate` ablation checkpoint into an engine checkpoint."""
    blob = torch.load(candidate, map_location="cpu", weights_only=False)
    meta = dict(blob["meta"])
    if meta.get("residual"):
        raise SystemExit(
            "refusing to promote the residual arm: the engine reconstructs "
            "prices as scale * softplus(net(x)) with no analytic baseline, so a "
            "residual checkpoint would be served without its C_geo term. "
            "Supporting it needs an engine change, not a file copy.")

    scales = {float(s["scale"]) for s in blob["members"]}
    if len(scales) != 1:
        raise SystemExit(f"members disagree on the output scale: {scales}")
    scale = scales.pop()

    members = []
    for state in blob["members"]:
        stripped = {k[len("net."):]: v for k, v in state.items()
                    if k.startswith("net.")}
        if not stripped:
            raise SystemExit(f"{candidate} has no net.* parameters")
        members.append(stripped)

    meta["output_scale"] = scale
    meta["promoted_from"] = candidate.name
    meta["n_parameters"] = sum(v.numel() for v in members[0].values())
    return {"members": members, "meta": meta}


def evaluate(engine: PricingEngine, X: np.ndarray, ref: np.ndarray,
             ref_delta: np.ndarray, ref_vega: np.ndarray) -> dict:
    """Price AND Greeks.

    The first version of this gate scored price only, and promoted a checkpoint
    whose delta was 5.9% WORSE than the incumbent's without noticing. The
    conditioned head helps the level and slightly hurts the shape - expected,
    since it moves where the magnitude lives in a network trained on a joint
    price-and-derivative loss. Greeks are therefore reported here but do NOT
    block: price accuracy is this model's primary claim, and a blocking Greeks
    criterion is a product decision, not a numerical one. A service that hedges
    off these Greeks should set `--require-greeks`.
    """
    pred = engine.price_batch(X[:, 0], np.ones(len(X)), X[:, 1], X[:, 2],
                              X[:, 3], option_type="call")
    err = (pred - ref) * 1e4
    d = np.empty(len(X))
    v = np.empty(len(X))
    for i, (m, t, s, r) in enumerate(X):
        o = engine.price_with_greeks(float(m), 1.0, float(t), float(s),
                                     float(r), "call")
        d[i] = o["greeks"]["delta"]
        v[i] = o["greeks"]["vega"] * 100.0
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()),
            "pct_positive": float((err > 0).mean() * 100),
            "p95": float(np.percentile(np.abs(err), 95)),
            "max": float(np.abs(err).max()),
            "bias2_over_mse": float(err.mean() ** 2 / (err ** 2).mean()),
            "delta_rmse": float(np.sqrt((((d - ref_delta) * 1e4) ** 2).mean())),
            "vega_rmse": float(np.sqrt((((v - ref_vega) * 1e4) ** 2).mean()))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate",
                   default=str(ARTIFACTS / "ablation_baseline_softplus.pt"))
    p.add_argument("--points", type=int, default=1500)
    p.add_argument("--ref-paths", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=606_060,
                   help="must differ from every training and ablation seed")
    p.add_argument("--write", action="store_true")
    p.add_argument("--require-greeks", action="store_true",
                   help="also require delta and vega not to regress")
    args = p.parse_args()

    candidate = Path(args.candidate)
    served = ARTIFACTS / "model.pt"
    dev = default_device()

    lo = np.array([l for l, _ in PARAM_RANGES.values()])
    hi = np.array([h for _, h in PARAM_RANGES.values()])
    X = lo + qmc.LatinHypercube(d=4, seed=args.seed).random(args.points) * (hi - lo)

    print(f"references: {args.points} points x {args.ref_paths:,} paths "
          f"(float64, {dev}) ...")
    X64 = torch.as_tensor(X, dtype=torch.float64, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 1)
    ref = torch.empty(args.points, dtype=torch.float64, device=dev)
    rd = torch.empty_like(ref)
    rv = torch.empty_like(ref)
    step = max(1, min(8, 2_000_000_000 // (args.ref_paths * N_MONITORING_STEPS * 8)))
    for s in range(0, args.points, step):
        e = min(s + step, args.points)
        pr, dl, vg = simulate_chunk_gpu(X64[s:e], args.ref_paths,
                                        N_MONITORING_STEPS, gen)
        ref[s:e], rd[s:e], rv[s:e] = pr, dl, vg
    ref_np = ref.cpu().numpy()
    rd_np, rv_np = rd.cpu().numpy(), rv.cpu().numpy()

    incumbent = evaluate(PricingEngine(served), X, ref_np, rd_np, rv_np)

    tmp = ARTIFACTS / "_candidate_engine.pt"
    torch.save(adapt(candidate), tmp)
    try:
        challenger = evaluate(PricingEngine(tmp), X, ref_np, rd_np, rv_np)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"\n{'':<12}{'RMSE':>8}{'bias':>9}{'%pos':>8}{'p95':>8}"
          f"{'max':>8}{'bias2/mse':>11}{'delta':>9}{'vega':>9}")
    for name, m in (("incumbent", incumbent), ("candidate", challenger)):
        print(f"{name:<12}{m['rmse']:8.3f}{m['bias']:+9.3f}"
              f"{m['pct_positive']:8.1f}{m['p95']:8.3f}{m['max']:8.2f}"
              f"{m['bias2_over_mse'] * 100:10.1f}%"
              f"{m['delta_rmse']:9.3f}{m['vega_rmse']:9.3f}")
    for greek in ("delta", "vega"):
        lo_, hi_ = incumbent[f"{greek}_rmse"], challenger[f"{greek}_rmse"]
        if hi_ > lo_:
            print(f"  NOTE: {greek} RMSE regresses {(hi_ / lo_ - 1) * 100:+.1f}% "
                  f"({lo_:.3f} -> {hi_:.3f}) - reported, not blocking; "
                  f"pass --require-greeks to make it blocking")

    wins = (challenger["rmse"] < incumbent["rmse"]
            and abs(challenger["bias"]) < abs(incumbent["bias"]))
    if args.require_greeks:
        wins = (wins
                and challenger["delta_rmse"] <= incumbent["delta_rmse"]
                and challenger["vega_rmse"] <= incumbent["vega_rmse"])
    print(f"\nRMSE {(incumbent['rmse'] - challenger['rmse']) / incumbent['rmse'] * 100:+.1f}%"
          f" | |bias| {(abs(incumbent['bias']) - abs(challenger['bias'])) / abs(incumbent['bias']) * 100:+.1f}%")

    if not wins:
        print("candidate does not beat the incumbent on both RMSE and |bias| "
              "- NOT promoting.")
        return 1
    if not args.write:
        print("candidate wins. Re-run with --write to promote.")
        return 0

    legacy = ARTIFACTS / "model_legacy_unconditioned_head.pt"
    if not legacy.exists():
        shutil.copy2(served, legacy)
        print(f"kept the incumbent as {legacy.name}")
    torch.save(adapt(candidate), served)
    print(f"promoted {candidate.name} -> {served.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
