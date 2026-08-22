"""Full-scale controlled ablation: Softplus-on-price vs residual-over-geometric.

The question
------------
The shipped surrogate carries a systematic +1.0 bp ADDITIVE price bias. Measured on
600 held-out points against 200,000-path references, 89.3% of price errors are
positive and the mean accounts for 47.6% of MSE. Bucketing by true price magnitude
shows the offset is flat (+1.05, +1.03, +1.06, +1.20, +0.89 bps across five decades
of price), and at points where the true price is below 0.01 bps the network still
predicts ~1.08 bps and never goes below 0.898.

The cause is the output layer. nn.Softplus() is log(1+e^x), which cannot emit zero:
reaching zero requires driving the pre-activation to -infinity. The optimizer settles
at a floor around 1 bp, and because the surface is smooth and globally parameterized
that floor lifts the entire function.

The fix under test
------------------
By AM-GM the arithmetic average dominates the geometric average pathwise, so
(A-K)+ >= (G-K)+ pathwise and therefore C_arith >= C_geo everywhere. The geometric
Asian has a closed form (Kemna & Vorst 1990), already implemented. So parameterize

    C_arith(x) = C_geo(x)  +  softplus(net(x))          [residual arm]

instead of

    C_arith(x) = softplus(net(x))                        [baseline arm]

This enforces C_arith >= C_geo by construction - a genuine structural guarantee, not
a learned approximation - and leaves the Softplus floor governing only the residual
Delta = C_arith - C_geo, which is orders of magnitude smaller than C itself.

Experimental control
--------------------
Both arms share: the same labels, the same loss, the same optimizer and schedule,
the same seeds, the same train/val split, the same architecture and parameter count,
and the same number of epochs and ensemble members. The ONLY difference is whether
the analytic baseline is added to the output. Gradients flow through the baseline in
the residual arm, so the differential (Greek-matching) loss targets dC_arith/dx in
both arms and remains directly comparable.

Both arms are evaluated against independent high-precision Monte Carlo references
(float64 on GPU) drawn from a Latin hypercube seed used by neither training arm.

Usage
-----
    python -m scripts.fullscale_ablation --probe        # time a few epochs, project
    python -m scripts.fullscale_ablation                # full run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

from backend.quant.dataset import PARAM_RANGES, N_MONITORING_STEPS
from backend.quant.gpu_labels import (default_device, generate_dataset_gpu,
                                      geometric_asian_call_torch,
                                      simulate_chunk_gpu)
from backend.quant.model import AsianPricerNet

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


# --------------------------------------------------------------------------- #
#  Parameterized surrogate: the ablation switch lives here and nowhere else.
# --------------------------------------------------------------------------- #
class Surrogate(nn.Module):
    """Wraps AsianPricerNet with an optional analytic geometric-Asian baseline.

    Takes PHYSICAL inputs (m, T, sigma, r) and normalizes internally, so autograd
    with respect to the input tensor yields dPrice/dm and dPrice/dsigma directly -
    no manual chain rule through the normalization, and identical handling in both
    arms.
    """

    def __init__(self, lows: torch.Tensor, highs: torch.Tensor, *,
                 residual: bool, n_steps: int, width: int = 128,
                 n_blocks: int = 4):
        super().__init__()
        self.net = AsianPricerNet(width=width, n_blocks=n_blocks)
        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.register_buffer("scale", torch.tensor(1.0))
        self.residual = residual
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = 2.0 * (x - self.lows) / (self.highs - self.lows) - 1.0
        out = self.scale * self.net(xn)
        if not self.residual:
            return out
        base = geometric_asian_call_torch(x[:, 0:1], x[:, 1:2], x[:, 2:3],
                                          x[:, 3:4], self.n_steps).squeeze(-1)
        return base + out

    @torch.no_grad()
    def init_output_level(self, target_mean: float) -> None:
        """Condition the Softplus head to operate near unity, and carry the
        magnitude in a fixed output scale instead.

        The first attempt at this ablation put the magnitude INSIDE the Softplus
        by setting the head bias to softplus^-1(target). That kills the residual
        arm. Its mean target is 175 bps = 0.0175, which needs a pre-activation of
        -4.04, where the Softplus derivative is sigmoid(-4.04) = 0.0174 - gradients
        attenuated 57x. Three of five members drove the pre-activation further
        negative, saturated, and died: validation RMSE pinned at exactly 287.89 bps
        for 300+ consecutive epochs, which is precisely the RMSE of predicting
        C_geo alone, i.e. Softplus output identically zero.

        Fix: keep the network's own output near 1 (Softplus derivative
        sigmoid(0.54) = 0.63, a healthy regime) and multiply by a FIXED,
        non-trainable scale carrying the arm's mean magnitude:

            price = base + scale * softplus(net(x))

        Both arms get this treatment at their own mean target, so the parameter
        count, the conditioning and the optimization regime stay matched and the
        only remaining difference is the analytic baseline - which is the thing
        the ablation is supposed to measure.
        """
        self.scale.fill_(max(target_mean, 1e-8))
        self.net.head.weight.mul_(0.1)
        self.net.head.bias.fill_(math.log(math.expm1(1.0)))   # softplus -> 1.0


def dml_loss(model, xb, yb, db, var_y, var_d, var_v, lam):
    """Variance-normalized price MSE + Greek-matching MSE (Huge & Savine sec 3.3)."""
    xb = xb.clone().requires_grad_(True)
    pred = model(xb)
    price_mse = nn.functional.mse_loss(pred, yb)
    (g,) = torch.autograd.grad(pred.sum(), xb, create_graph=True)
    delta_mse = nn.functional.mse_loss(g[:, 0], db[:, 0])
    vega_mse = nn.functional.mse_loss(g[:, 2], db[:, 1])
    combined = price_mse / var_y + lam * (delta_mse / var_d + vega_mse / var_v)
    return combined, price_mse


def train_arm(residual, X, y, D, lows, highs, args, dev):
    """Train one arm's ensemble. Returns (state_dicts, per-member val rmse)."""
    n_val = max(int(0.1 * len(X)), 1)
    torch.manual_seed(args.seed)                      # split shared by both arms
    perm = torch.randperm(len(X), device=dev)
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, ytr, Dtr = X[ti], y[ti], D[ti]
    Xva, yva, Dva = X[vi], y[vi], D[vi]

    var_y = float(ytr.var())
    var_d = float(Dtr[:, 0].var())
    var_v = float(Dtr[:, 1].var())
    steps = math.ceil(len(Xtr) / args.batch)

    # Each arm's Softplus head starts at that arm's own mean target: the whole
    # price for the baseline, the arithmetic-minus-geometric residual for the
    # residual arm. Computed on the TRAINING split only.
    with torch.no_grad():
        if residual:
            base_tr = geometric_asian_call_torch(
                Xtr[:, 0:1], Xtr[:, 1:2], Xtr[:, 2:3], Xtr[:, 3:4],
                N_MONITORING_STEPS).squeeze(-1)
            target_mean = float((ytr - base_tr).clamp(min=0).mean())
        else:
            target_mean = float(ytr.mean())
    print(f"  [{'residual' if residual else 'baseline'}] head initialized at "
          f"{target_mean*1e4:.2f} bps", flush=True)

    states, val_rmses = [], []
    for member in range(args.ensemble):
        torch.manual_seed(args.seed + 1000 * (member + 1))
        model = Surrogate(lows, highs, residual=residual,
                          n_steps=N_MONITORING_STEPS, width=args.width,
                          n_blocks=args.blocks).to(dev)
        model.init_output_level(target_mean)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_score, best_val, best_state = math.inf, math.inf, None
        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            ep = torch.randperm(len(Xtr), device=dev)
            for i in range(steps):
                idx = ep[i * args.batch:(i + 1) * args.batch]
                opt.zero_grad(set_to_none=True)
                loss, _ = dml_loss(model, Xtr[idx], ytr[idx], Dtr[idx],
                                   var_y, var_d, var_v, args.lam)
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            vs, vp = dml_loss(model, Xva, yva, Dva, var_y, var_d, var_v, args.lam)
            vs, vp = vs.item(), vp.item()
            if vs < best_score:
                best_score, best_val = vs, math.sqrt(vp)
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if epoch % 25 == 0 or epoch == 1:
                print(f"  [{'residual' if residual else 'baseline'}] "
                      f"member {member+1}/{args.ensemble} epoch {epoch:>4}/"
                      f"{args.epochs}  val price rmse {math.sqrt(vp)*1e4:7.2f} bps"
                      f"  ({time.perf_counter()-t0:6.1f}s)", flush=True)
        states.append(best_state)
        val_rmses.append(best_val)
    return states, val_rmses


@torch.no_grad()
def _predict(states, residual, Xe, lows, highs, args, dev):
    preds = []
    for st in states:
        m = Surrogate(lows, highs, residual=residual, n_steps=N_MONITORING_STEPS,
                      width=args.width, n_blocks=args.blocks).to(dev)
        m.load_state_dict(st)
        m.eval()
        preds.append(m(Xe))
    return torch.stack(preds).mean(0)


def bucket_report(err_bps, true_bps):
    out = []
    for lo, hi in [(-1, 1), (1, 10), (10, 100), (100, 1000), (1000, 1e18)]:
        s = (true_bps >= lo) & (true_bps < hi)
        if s.sum() == 0:
            continue
        e = err_bps[s]
        out.append({"bucket": f"[{lo:g},{hi:g})" if hi < 1e17 else "[1000,inf)",
                    "n": int(s.sum()), "mean_bps": float(e.mean()),
                    "rmse_bps": float(np.sqrt((e ** 2).mean())),
                    "pct_positive": float((e > 0).mean() * 100)})
    return out


def summarize(err_bps):
    return {"rmse_bps": float(np.sqrt((err_bps ** 2).mean())),
            "mae_bps": float(np.abs(err_bps).mean()),
            "bias_bps": float(err_bps.mean()),
            "pct_positive": float((err_bps > 0).mean() * 100),
            "p95_abs_bps": float(np.percentile(np.abs(err_bps), 95)),
            "max_abs_bps": float(np.abs(err_bps).max()),
            "bias2_over_mse": float(err_bps.mean() ** 2 /
                                    (err_bps ** 2).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=500_000)
    p.add_argument("--paths", type=int, default=5_000)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-points", type=int, default=2000)
    p.add_argument("--eval-paths", type=int, default=200_000)
    p.add_argument("--eval-seed", type=int, default=20261)
    p.add_argument("--probe", action="store_true",
                   help="tiny run to time an epoch and project the full cost")
    args = p.parse_args()
    if args.probe:
        args.samples, args.epochs, args.ensemble = 50_000, 3, 1
        args.eval_points = 200

    dev = default_device()
    print(f"device: {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")
    ARTIFACTS.mkdir(exist_ok=True)

    # ---- labels (float64 on GPU) -------------------------------------------
    cache = ARTIFACTS / f"dataset_gpu_{args.samples}_{args.paths}_{args.seed}.npz"
    if cache.exists():
        b = np.load(cache)
        Xn, yn, Dn = b["X"], b["y"], b["dydx"]
        print(f"loaded cached dataset {Xn.shape[0]:,}")
    else:
        print(f"generating {args.samples:,} labels x {args.paths:,} paths "
              f"(float64, GPU)...")
        Xn, yn, Dn = generate_dataset_gpu(args.samples, args.paths,
                                          seed=args.seed, chunk_size=256,
                                          device=dev)
        np.savez_compressed(cache, X=Xn, y=yn, dydx=Dn)

    lows = torch.tensor([lo for lo, _ in PARAM_RANGES.values()],
                        dtype=torch.float32, device=dev)
    highs = torch.tensor([hi for _, hi in PARAM_RANGES.values()],
                         dtype=torch.float32, device=dev)
    X = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
    y = torch.as_tensor(yn, dtype=torch.float32, device=dev)
    D = torch.as_tensor(Dn, dtype=torch.float32, device=dev)

    # ---- train both arms ----------------------------------------------------
    results = {}
    t_all = time.perf_counter()
    for name, residual in (("baseline_softplus", False),
                           ("residual_geometric", True)):
        t0 = time.perf_counter()
        states, vr = train_arm(residual, X, y, D, lows, highs, args, dev)
        secs = time.perf_counter() - t0
        torch.save({"members": states, "meta": {
            "arm": name, "residual": residual, "width": args.width,
            "blocks": args.blocks, "n_members": args.ensemble,
            "n_samples": args.samples, "mc_paths_per_label": args.paths,
            "epochs": args.epochs, "seed": args.seed,
            "n_monitoring_steps": N_MONITORING_STEPS,
            "param_ranges": PARAM_RANGES, "label_dtype": "float64",
            "label_device": str(dev), "train_seconds": round(secs, 1),
        }}, ARTIFACTS / f"ablation_{name}.pt")
        results[name] = {"states": states, "val_rmse": vr, "seconds": secs}
        print(f"[{name}] trained in {secs/60:.1f} min | "
              f"val rmse {np.mean(vr)*1e4:.2f} bps", flush=True)

    # ---- independent high-precision references ------------------------------
    lo_np = np.array([l for l, _ in PARAM_RANGES.values()])
    hi_np = np.array([h for _, h in PARAM_RANGES.values()])
    Xe = lo_np + qmc.LatinHypercube(d=4, seed=args.eval_seed).random(
        args.eval_points) * (hi_np - lo_np)
    print(f"\nreferences: {args.eval_points} points x {args.eval_paths:,} "
          f"paths (float64 GPU)...")
    Xe64 = torch.as_tensor(Xe, dtype=torch.float64, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.eval_seed + 1)
    ref = torch.empty(args.eval_points, dtype=torch.float64, device=dev)
    # Reference chunks hold ~4 live (CH, eval_paths, steps) float64 tensors, so
    # CH must stay small: at 200k paths one such tensor is CH * 80 MB.
    CH = max(1, min(8, 2_000_000_000 // (args.eval_paths * N_MONITORING_STEPS * 8)))
    t0 = time.perf_counter()
    for s in range(0, args.eval_points, CH):
        e = min(s + CH, args.eval_points)
        pr, _, _ = simulate_chunk_gpu(Xe64[s:e], args.eval_paths,
                                      N_MONITORING_STEPS, gen)
        ref[s:e] = pr
    print(f"  done in {time.perf_counter()-t0:.0f}s")
    ref_np = ref.cpu().numpy()
    Xe32 = torch.as_tensor(Xe, dtype=torch.float32, device=dev)

    report = {"config": vars(args), "device": str(dev),
              "reference": {"points": args.eval_points,
                            "paths": args.eval_paths, "seed": args.eval_seed,
                            "dtype": "float64"},
              "arms": {}}
    print("\n" + "=" * 92)
    print("FULL-SCALE ABLATION RESULT (bps of strike, vs float64 GPU references)")
    print("=" * 92)
    for name, residual in (("baseline_softplus", False),
                           ("residual_geometric", True)):
        pred = _predict(results[name]["states"], residual, Xe32, lows, highs,
                        args, dev).double().cpu().numpy()
        err = (pred - ref_np) * 1e4
        st = summarize(err)
        report["arms"][name] = {
            "residual": residual, "train_seconds": results[name]["seconds"],
            "val_rmse_bps": [v * 1e4 for v in results[name]["val_rmse"]],
            "ensemble": st, "by_true_price": bucket_report(err, ref_np * 1e4),
            "errors_bps": np.round(err, 4).tolist()}
        print(f"{name:>20} | rmse {st['rmse_bps']:7.3f} | bias "
              f"{st['bias_bps']:+7.3f} | %pos {st['pct_positive']:5.1f} | "
              f"p95 {st['p95_abs_bps']:6.3f} | bias^2/mse "
              f"{st['bias2_over_mse']*100:5.1f}%")

    a = report["arms"]["baseline_softplus"]["ensemble"]
    b = report["arms"]["residual_geometric"]["ensemble"]
    report["delta"] = {
        "rmse_improvement_pct": (a["rmse_bps"] - b["rmse_bps"]) / a["rmse_bps"] * 100,
        "bias_reduction_pct": (abs(a["bias_bps"]) - abs(b["bias_bps"]))
        / max(abs(a["bias_bps"]), 1e-12) * 100}
    print("-" * 92)
    print(f"residual vs baseline: RMSE {report['delta']['rmse_improvement_pct']:+.1f}%"
          f" | |bias| {report['delta']['bias_reduction_pct']:+.1f}%")
    print(f"total wall time: {(time.perf_counter()-t_all)/60:.1f} min")

    out = ARTIFACTS / ("ablation_probe.json" if args.probe else "ablation.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
