"""Train the pricing-map surrogate on the banked GPU labels.

Companion to gen_pricing_map.py. Input (9): eta, rho, H, ln xi, lam/150,
mu_j, sig_j, ln tau, k. Output: Black IV. The network is small on purpose —
the map is smooth, and a calibration objective evaluates it tens of thousands
of times, so inference cost matters more than the last basis point of fit.

The success criterion is written down before training: held-out RMSE must be
comfortably below the Monte Carlo noise already accepted in calibration
labels (~0.3-0.5 vol points at 64k paths), i.e. the surrogate must not be the
dominant error source. Validation splits BY PARAMETER SET, never by row: 31
rows share a simulation, and splitting rows would leak every smile across
the boundary.

Saves artifacts/pricing_map.pt with the normalization constants and the
training box, and refuses at load time to extrapolate outside it.

    python -m scripts.train_pricing_map --epochs 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

DATA_DIR = ROOT / "data" / "pricing_map"
ARTIFACTS = ROOT / "artifacts"


def load_dataset() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """(X, y, k_grid): X rows are (theta 8, k 1) -> y IV. NaN rows dropped."""
    shards = sorted(DATA_DIR.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {DATA_DIR}; run gen_pricing_map first")
    xs, ys, set_ids = [], [], []
    base = 0
    k_grid = None
    for sh in shards:
        d = np.load(sh)
        theta, iv = d["theta"], d["iv"]
        k_grid = d["k_grid"]
        n, m = iv.shape
        X = np.concatenate([np.repeat(theta, m, axis=0),
                            np.tile(k_grid, n)[:, None]], axis=1)
        y = iv.reshape(-1)
        sid = np.repeat(np.arange(base, base + n), m)
        keep = ~np.isnan(y)
        xs.append(X[keep]); ys.append(y[keep]); set_ids.append(sid[keep])
        base += n
    X = np.concatenate(xs); y = np.concatenate(ys)
    sid = np.concatenate(set_ids)
    return (torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            sid), k_grid, base


def features(X: torch.Tensor) -> torch.Tensor:
    """Raw theta+k -> network features. ln for the decade-spanning inputs."""
    eta, rho, H, xi, lam, mu, sg, tau, k = X.unbind(dim=1)
    return torch.stack([eta / 8.0, rho, H, torch.log(xi),
                        lam / 150.0, mu, sg, torch.log(tau), k], dim=1)


class PricingMap(nn.Module):
    def __init__(self, width: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(9, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Softplus keeps IV positive without a hard floor gradient.
        return nn.functional.softplus(self.net(feats)).squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=65_536)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (X, y, sid), k_grid, n_sets = load_dataset()
    print(f"{len(y):,} rows from {n_sets:,} parameter sets on {device}")

    # Split by PARAMETER SET: rows of one smile share a simulation and must
    # never straddle the train/val boundary.
    rng = np.random.default_rng(args.seed)
    val_sets = set(rng.choice(n_sets, size=max(n_sets // 10, 1),
                              replace=False).tolist())
    val_mask = torch.from_numpy(np.isin(sid, list(val_sets)))
    Xtr, ytr = X[~val_mask].to(device), y[~val_mask].to(device)
    Xva, yva = X[val_mask].to(device), y[val_mask].to(device)
    Ftr, Fva = features(Xtr), features(Xva)
    print(f"train {len(ytr):,} / val {len(yva):,} rows "
          f"({len(val_sets):,} held-out sets)")

    torch.manual_seed(args.seed)
    model = PricingMap(args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best, best_state = math.inf, None
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(ytr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Ftr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = math.sqrt(float(loss_fn(model(Fva), yva)))
        if val < best:
            best = val
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:>3}/{args.epochs}  val RMSE "
                  f"{val * 100:.3f} vol points  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval().cpu()
    from scripts.gen_pricing_map import BOX
    torch.save({"state": model.state_dict(),
                "width": args.width, "depth": args.depth,
                "val_rmse_volpts": best * 100,
                "n_sets": n_sets, "n_rows": int(len(y)),
                "k_range": [float(k_grid.min()), float(k_grid.max())],
                "box": {k: list(v) for k, v in BOX.items()},
                "inputs": "eta rho H xi lam mu_j sig_j tau k (raw; "
                          "features() handles transforms)",
                "labels": "Black IV, forward units, r=0",
                "note": "Deep-learning-volatility surrogate of "
                        "rough_bergomi_mc(+Merton jumps). lam=0 is exactly "
                        "the diffusive model. Trained on GPU-banked labels "
                        "so calibration survives without CUDA."},
               ARTIFACTS / "pricing_map.pt")
    print(f"\nbest val RMSE {best * 100:.3f} vol points -> "
          f"{ARTIFACTS / 'pricing_map.pt'}")
    print("acceptance bar: comfortably below the ~0.3-0.5 vp MC label noise "
          "already accepted at 64k calibration paths"
          + (" — MET" if best * 100 < 0.3 else " — NOT met; consider more "
             "data, width, or epochs"))


if __name__ == "__main__":
    main()
