"""Train the pricing-map surrogate on the banked GPU labels.

Companion to gen_pricing_map.py. Input (9): eta, rho, H, ln xi, lam/150,
mu_j, sig_j, ln tau, k. Output: Black IV. The network is small on purpose —
the map is smooth, and a calibration objective evaluates it tens of thousands
of times, so inference cost matters more than the last basis point of fit.

The acceptance metric, revised once and the revision recorded: the original
pre-registered bar (held-out RMSE < ~0.3 vp) was WRONG, because it compared
the network against single noisy MC labels — measured, the labels themselves
carry ~2.2 vp of Monte Carlo noise in the production region at 131k paths, so
no network can score below the noise of its own validation targets. The
binding criterion is validate_pricing_map.py's END-TO-END test: calibrate a
real surface with MC and with the map, reprice the map's parameters under MC.
Certified 2026-08-20 on a live 618-quote SPY capture: the map's parameters
cost +0.046 vp under the true model (MC fit noise floor ~0.97) at 3 s on CPU
vs 68 s on the RTX 5080. Held-out RMSE remains printed as a training
diagnostic. Validation splits BY PARAMETER SET, never by row: 31 rows share
a simulation, and splitting rows would leak every smile across the boundary.

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


HQ_DIR = ROOT / "data" / "pricing_map_hq"


def load_dataset() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """(X, y, k_grid): X rows are (theta 8, k 1) -> y IV. NaN rows dropped.

    When the high-precision relabel exists (same seed, same thetas, 4x the
    paths into data/pricing_map_hq), each label becomes the PRECISION-WEIGHTED
    average of the two independent simulations: weights proportional to path
    count (1/variance), so 131k + 524k paths behave like one 655k-path label —
    a ~2.2x noise reduction over the original labels. Where only one run
    inverted (deep wings drift in and out of the no-arb region between path
    counts), the defined one is used. Thetas are verified identical per shard.
    """
    shards = sorted(DATA_DIR.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {DATA_DIR}; run gen_pricing_map first")
    xs, ys, set_ids = [], [], []
    base = 0
    k_grid = None
    n_merged = 0
    for sh in shards:
        d = np.load(sh)
        theta, iv = d["theta"], d["iv"]
        k_grid = d["k_grid"]
        hq = HQ_DIR / sh.name
        # Shards 0000-0008 predate the mu_j box widening, so their thetas
        # differ from the HQ run's (same LHS draw, different mu_j scaling).
        # Those stay as-is -- their labels are valid for the thetas they
        # store -- and only theta-identical shards merge.
        if hq.exists() and np.allclose(theta, np.load(hq)["theta"], atol=1e-6):
            d2 = np.load(hq)
            w1, w2 = float(d["paths"]), float(d2["paths"])
            iv2 = d2["iv"]
            both = ~np.isnan(iv) & ~np.isnan(iv2)
            merged = np.where(np.isnan(iv), iv2, iv)
            merged[both] = (w1 * iv[both] + w2 * iv2[both]) / (w1 + w2)
            iv = merged
            n_merged += 1
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
    print(f"loaded {len(shards)} shards ({n_merged} precision-merged with HQ "
          f"relabels)")
    return (torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            sid), k_grid, base


N_FEATURES = 11


def features(X: torch.Tensor) -> torch.Tensor:
    """Raw theta+k -> network features.

    Two engineered inputs carry the geometry the first version forced the
    network to rediscover (and it didn't, at 5.15 vp):

      d = k / sqrt(xi * tau)   standardised moneyness — smiles are functions
                               of d far more than of raw k, so this one
                               feature linearises most of the surface;
      ln(xi * tau)             total variance, the natural time-scale.
    """
    eta, rho, H, xi, lam, mu, sg, tau, k = X.unbind(dim=1)
    d = (k / torch.sqrt(xi * tau)).clamp(-20.0, 20.0)
    return torch.stack([eta / 8.0, rho, H, torch.log(xi),
                        lam / 150.0, mu, sg, torch.log(tau), k,
                        d / 20.0, torch.log(xi * tau)], dim=1)


class PricingMap(nn.Module):
    def __init__(self, width: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(N_FEATURES, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # The network predicts ln(IV); exp keeps IV positive and — the real
        # point — makes the training loss RELATIVE. Labels span 9% to 440%
        # vol (the box is log-uniform in xi), and an absolute-IV MSE spends
        # its capacity on the 300%-vol corners while calibration lives at
        # 10-30%. In log space a 1% relative error costs the same everywhere.
        return torch.exp(self.net(feats).squeeze(-1).clamp(-4.0, 2.2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=65_536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
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
    ztr, zva = torch.log(ytr), torch.log(yva)

    best, best_state = math.inf, None
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(ytr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(torch.log(model(Ftr[idx]).clamp_min(1e-4)),
                           ztr[idx])
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
    print("NOTE: this number includes the validation labels' own MC noise "
          "(~2.2 vp in the production region) and is a diagnostic, not the "
          "gate. The gate is scripts/validate_pricing_map.py — end-to-end "
          "parameter recovery on a real surface, repriced under MC.")


if __name__ == "__main__":
    main()
