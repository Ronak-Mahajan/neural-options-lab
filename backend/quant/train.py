"""Train the neural Asian-option pricer with Differential Machine Learning.

The loss follows Huge & Savine (2020): alongside the price MSE, the
network's own input gradients (obtained by differentiating the forward pass)
are regressed onto the *pathwise* Monte Carlo differentials computed during
simulation:

    L = MSE(f, y) / Var(y)
        + lambda * [ MSE(df/dm, delta_pw) / Var(delta_pw)
                   + MSE(df/dsigma, vega_pw) / Var(vega_pw) ]

Each term is variance-normalized so lambda = 1 balances them regardless of
units. Matching differentials teaches the network the shape of the pricing
function between sample points - better Greeks *and* better prices per label.

Usage (from the repo root):
    python -m backend.quant.train --ensemble 5    # full DML ensemble
    python -m backend.quant.train --quick         # smoke test (~1 min)
    python -m backend.quant.train --no-differential   # price-only ablation

Artifacts land in ./artifacts:
    dataset.npz   cached MC training set incl. pathwise differentials
    model.pt      best-validation checkpoints + normalization + metadata
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

from .dataset import PARAM_RANGES, N_MONITORING_STEPS, generate_dataset
from .model import AsianPricerNet

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def normalize(X: np.ndarray) -> np.ndarray:
    lows = np.array([lo for lo, _ in PARAM_RANGES.values()])
    highs = np.array([hi for _, hi in PARAM_RANGES.values()])
    return (2.0 * (X - lows) / (highs - lows) - 1.0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=40_000)
    p.add_argument("--paths", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ensemble", type=int, default=1,
                   help="number of independently initialized members to train")
    p.add_argument("--lambda-diff", type=float, default=1.0,
                   help="weight of the differential (Greek-matching) loss")
    p.add_argument("--no-differential", action="store_true",
                   help="ablation: train on prices only")
    p.add_argument("--quick", action="store_true",
                   help="small run for smoke-testing the pipeline")
    args = p.parse_args()
    if args.quick:
        args.samples, args.paths, args.epochs = 8_000, 1_000, 60
    use_diff = not args.no_differential

    ARTIFACTS.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    cache = ARTIFACTS / "dataset.npz"
    key = {"samples": args.samples, "paths": args.paths, "seed": args.seed,
           "fmt": 2}  # fmt 2 = includes pathwise differentials
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        if json.loads(str(blob["key"])) == key:
            X, y, dydx = blob["X"], blob["y"], blob["dydx"]
            print(f"loaded cached dataset ({X.shape[0]:,} samples)")
        else:
            X = None
    else:
        X = None
    if X is None:
        print(f"generating {args.samples:,} CV Monte Carlo labels + pathwise "
              f"differentials ({args.paths:,} paths x {N_MONITORING_STEPS} "
              f"steps each)...")
        X, y, dydx = generate_dataset(args.samples, args.paths,
                                      seed=args.seed)
        np.savez_compressed(cache, X=X, y=y, dydx=dydx, key=json.dumps(key))

    Xn = torch.from_numpy(normalize(X))
    yt = torch.from_numpy(y.astype(np.float32))
    dt_lbl = torch.from_numpy(dydx.astype(np.float32))     # (N, 2)

    torch.manual_seed(args.seed)  # split is shared by all members
    n_val = max(int(0.1 * len(Xn)), 1)
    perm = torch.randperm(len(Xn))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = Xn[tr_idx], yt[tr_idx], Xn[val_idx], yt[val_idx]
    Dtr, Dva = dt_lbl[tr_idx], dt_lbl[val_idx]

    # Chain rule from normalized inputs back to physical (m, sigma):
    # xn = 2(x - lo)/(hi - lo) - 1  =>  df/dx = df/dxn * 2/(hi - lo).
    lows = [lo for lo, _ in PARAM_RANGES.values()]
    highs = [hi for _, hi in PARAM_RANGES.values()]
    scale_m = 2.0 / (highs[0] - lows[0])
    scale_sig = 2.0 / (highs[2] - lows[2])

    # Variance normalization makes lambda unit-free (Huge & Savine sec. 3.3).
    var_y = float(ytr.var())
    var_delta = float(Dtr[:, 0].var())
    var_vega = float(Dtr[:, 1].var())

    loss_fn = nn.MSELoss()

    def dml_loss(model: AsianPricerNet, xb: torch.Tensor, yb: torch.Tensor,
                 db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(normalized combined loss, raw price mse). Differentiates the
        network wrt its inputs with create_graph=True so the optimizer can
        backprop through the Greek-matching terms (double backprop)."""
        xb = xb.clone().requires_grad_(True)
        pred = model(xb)
        price_mse = loss_fn(pred, yb)
        if not use_diff:
            return price_mse / var_y, price_mse
        (g,) = torch.autograd.grad(pred.sum(), xb, create_graph=True)
        delta_mse = loss_fn(g[:, 0] * scale_m, db[:, 0])
        vega_mse = loss_fn(g[:, 2] * scale_sig, db[:, 1])
        combined = price_mse / var_y + args.lambda_diff * (
            delta_mse / var_delta + vega_mse / var_vega)
        return combined, price_mse

    steps_per_epoch = math.ceil(len(Xtr) / args.batch)
    member_states: list[dict[str, torch.Tensor]] = []
    member_val_rmse: list[float] = []
    t_all = time.perf_counter()

    # ---------------------------------------------------------- deep ensemble
    # Members share the data but differ in initialization and batch order -
    # the standard deep-ensembles recipe. Averaging N members shrinks the
    # (decorrelated part of the) approximation error roughly like 1/sqrt(N).
    for member in range(args.ensemble):
        torch.manual_seed(args.seed + 1000 * (member + 1))
        model = AsianPricerNet(width=args.width, n_blocks=args.blocks)
        if member == 0:
            print(f"model: {args.blocks} residual blocks x {args.width} wide "
                  f"({model.count_parameters():,} parameters) "
                  f"x {args.ensemble} ensemble member(s)")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs)

        best_score = math.inf     # combined normalized val loss
        best_val = math.inf       # price val RMSE of the selected state
        best_state: dict[str, torch.Tensor] | None = None
        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_perm = torch.randperm(len(Xtr))
            total = 0.0
            for i in range(steps_per_epoch):
                idx = ep_perm[i * args.batch:(i + 1) * args.batch]
                opt.zero_grad(set_to_none=True)
                loss, price_mse = dml_loss(model, Xtr[idx], ytr[idx],
                                           Dtr[idx])
                loss.backward()
                opt.step()
                total += price_mse.item() * len(idx)
            sched.step()

            model.eval()
            # Validation needs input grads for the Greek terms, so no
            # torch.no_grad() here; parameters get no .grad (we never
            # call backward).
            val_score_t, val_pmse_t = dml_loss(model, Xva, yva, Dva)
            val_score = val_score_t.item()
            val_rmse = math.sqrt(val_pmse_t.item())
            if val_score < best_score:
                best_score = val_score
                best_val = val_rmse
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if epoch % 25 == 0 or epoch == 1:
                print(f"member {member + 1}/{args.ensemble}  "
                      f"epoch {epoch:>4}/{args.epochs}  "
                      f"train price mse {total / len(Xtr):.3e}  "
                      f"val price rmse {val_rmse:.3e}  "
                      f"val score {val_score:.3e}  "
                      f"({time.perf_counter() - t0:5.1f}s)", flush=True)

        if best_state is not None:
            member_states.append(best_state)
        member_val_rmse.append(best_val)
        print(f"member {member + 1} done: val price rmse {best_val:.3e} "
              f"({best_val * 1e4:.1f} bps of strike)", flush=True)

    # ------------------------------------------------- ensemble validation
    with torch.no_grad():
        preds = []
        probe = AsianPricerNet(width=args.width, n_blocks=args.blocks)
        for state in member_states:
            probe.load_state_dict(state)
            probe.eval()
            preds.append(probe(Xva))
        ens_rmse = math.sqrt(loss_fn(torch.stack(preds).mean(0), yva).item())

    # ------------------------------------------------------------ checkpoint
    # Validation RMSE is on price/K; multiply by 1e4 to read it in
    # basis points of strike. Val labels are themselves MC-noisy, so this is
    # an upper bound - run backend.quant.evaluate for error vs high-precision
    # references.
    meta = {
        "width": args.width,
        "blocks": args.blocks,
        "n_members": args.ensemble,
        "differential_ml": use_diff,
        "lambda_diff": args.lambda_diff if use_diff else 0.0,
        "param_ranges": PARAM_RANGES,
        "n_monitoring_steps": N_MONITORING_STEPS,
        "n_samples": args.samples,
        "mc_paths_per_label": args.paths,
        "epochs": args.epochs,
        "member_val_rmse": member_val_rmse,
        "val_rmse_price_over_k": ens_rmse,
        "val_rmse_bps_of_strike": ens_rmse * 1e4,
        "n_parameters": AsianPricerNet(
            width=args.width, n_blocks=args.blocks
        ).count_parameters(),
        "train_seconds": round(time.perf_counter() - t_all, 1),
    }
    torch.save({"members": member_states, "meta": meta},
               ARTIFACTS / "model.pt")
    print(f"\nsaved {ARTIFACTS / 'model.pt'}")
    print(f"single-member val RMSE: {member_val_rmse[0] * 1e4:.1f} bps | "
          f"ensemble val RMSE: {ens_rmse * 1e4:.1f} bps of strike "
          f"(vs noisy MC labels)")


if __name__ == "__main__":
    main()
