"""Bank the GPU: generate the dataset for a rough-Bergomi pricing-map surrogate.

The RTX 5080 leaves in two days, and with it goes live calibration: one
objective evaluation at 64k paths is ~0.5 s on this GPU and ~90 s on CPU, so a
three-stage fit goes from 100 seconds to half a day. The standard escape
(Horvath, Muguruza & Tomas 2021, "Deep learning volatility") is to spend the
GPU ONCE on a dataset mapping model parameters to implied vols, train a small
network on it, and calibrate forever after against the network — milliseconds
per objective evaluation, on any machine.

One map covers BOTH drivers this project simulates: the input is

    (eta, rho, H, xi, lam, mu_j, sig_j, tau, k)  ->  Black IV

with (lam, mu_j, sig_j) the compensated Merton jump extension and lam = 0
EXACTLY the diffusive model. 35% of samples are drawn with lam = 0 so the
diffusive submanifold — the one production calibration lives on — is densely
covered rather than approached in the limit. Ranges cover the union of the
SPY box and the widened BTC box (eta to 8, jump sizes to -25%), because the
BTC surface is exactly where the diffusive model railed and the jump model is
needed.

Everything is normalized: the simulation starts at F = 1 with r = 0, so
prices are undiscounted forward-unit prices and k = ln(K/F). A calibration
against real quotes feeds the map its own (k, tau) per quote and compares in
IV space, where quotes from any underlying and any rate regime are directly
comparable.

Per parameter set, ONE simulation prices the whole 31-strike grid (the same
grouped-MC trick the calibrator uses), at 131,072 paths. Labels are stored as
IV where the Black inversion exists and NaN where it does not (deep wings of
short maturities); training masks NaNs. Shards of 2,000 sets land in
data/pricing_map/ (gitignored) so an interruption loses minutes, and
generation is resumable: existing shards are counted and skipped.

    python -m scripts.gen_pricing_map --n-sets 200000          # ~5-6 h GPU
    python -m scripts.gen_pricing_map --n-sets 400 --paths 8192  # smoke test
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scipy.stats import qmc

from backend.quant.calibrate import implied_vol
from backend.quant.rough_vol import rough_bergomi_mc

OUT_DIR = ROOT / "data" / "pricing_map"
SHARD = 2_000
K_GRID = np.linspace(-0.25, 0.20, 31)         # ln(K/F); covers BTC-wide smiles
DIFFUSIVE_FRAC = 0.35                          # exact-lam=0 share

#: Sampling box: the UNION of the SPY calibration box and the widened BTC one.
#: xi and tau are log-uniform (both span decades and matter at the low end);
#: everything else uniform.
BOX = {
    "eta":   (0.5, 8.0),
    "rho":   (-0.99, 0.0),
    "H":     (0.01, 0.5),
    "ln_xi": (math.log(0.05 ** 2), math.log(1.2 ** 2)),
    "lam":   (0.0, 150.0),
    # SYMMETRIC: the live BTC experiment found the market pricing POSITIVE
    # jumps (call wing bid, fitted mu_j +1.4%) after a mu_j <= 0 bound railed.
    # The 7 shards banked before this change carry mu_j <= 0 thetas; each
    # shard stores its own thetas, so mixing boxes only thins coverage of
    # mu_j > 0 by 7/100 -- harmless. Regeneration is impossible post-GPU, so
    # the box errs wide.
    "mu_j":  (-0.25, 0.25),
    "sig_j": (0.003, 0.25),
    "ln_tau": (math.log(0.8 / 365.0), math.log(17.0 / 365.0)),
}


def sample_sets(n: int, seed: int) -> np.ndarray:
    """(n, 8) parameter sets: eta, rho, H, xi, lam, mu_j, sig_j, tau."""
    keys = list(BOX)
    lhs = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    lo = np.array([BOX[k][0] for k in keys])
    hi = np.array([BOX[k][1] for k in keys])
    raw = lo + lhs * (hi - lo)
    out = np.empty_like(raw)
    for j, k in enumerate(keys):
        out[:, j] = np.exp(raw[:, j]) if k.startswith("ln_") else raw[:, j]
    # Exact-diffusive share: the production model IS lam=0, so cover it
    # densely instead of relying on samples near zero.
    rng = np.random.default_rng(seed)
    out[rng.random(n) < DIFFUSIVE_FRAC, 4] = 0.0
    return out


def price_one(theta: np.ndarray, n_paths: int, seed: int,
              device: str) -> np.ndarray:
    """IVs for the whole K_GRID under one parameter set. NaN = no inversion."""
    eta, rho, H, xi, lam, mu_j, sig_j, tau = (float(v) for v in theta)
    strikes = np.exp(K_GRID)
    b = len(strikes)
    t = lambda v: torch.full((b,), float(v), device=device)
    prices = rough_bergomi_mc(
        t(1.0), torch.tensor(strikes, dtype=torch.float32, device=device),
        t(tau), t(xi), t(eta), t(rho), t(0.0),
        n_paths=n_paths, n_steps=50, H=H, seed=seed,
        jumps=(lam, mu_j, sig_j) if lam > 0.0 else None,
    ).cpu().numpy()
    ivs = np.full(b, np.nan)
    for i, (K, p) in enumerate(zip(strikes, prices)):
        iv = implied_vol(float(p), 1.0, float(K), tau, 0.0)
        if iv is not None:
            ivs[i] = iv
    return ivs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sets", type=int, default=200_000)
    p.add_argument("--paths", type=int, default=131_072)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--stride", type=int, default=1,
                   help="process every stride-th shard (parallel workers)")
    p.add_argument("--offset", type=int, default=0,
                   help="this worker's shard offset in [0, stride)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device — this generator exists precisely "
              "because CPU is ~200x slower. Proceeding anyway.", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    thetas = sample_sets(args.n_sets, args.seed)
    n_shards = math.ceil(args.n_sets / SHARD)
    done = {int(f.stem.split("_")[1]) for f in OUT_DIR.glob("shard_*.npz")}
    print(f"{args.n_sets:,} parameter sets in {n_shards} shards of {SHARD}; "
          f"{len(done)} shards already on disk; device {device}; "
          f"{args.paths:,} paths x 50 steps x {len(K_GRID)} strikes each",
          flush=True)

    t0 = time.perf_counter()
    # Measured single-process: GPU 3% utilized, CPU 15% -- the bottleneck is
    # the sequential Python implied-vol inversion between GPU batches, so the
    # dataset parallelizes across PROCESSES almost linearly. Workers stripe
    # the shard index (s % stride == offset): disjoint files, no locks, and
    # the same resume semantics as a single worker.
    for s in range(n_shards):
        if s % args.stride != args.offset or s in done:
            continue
        lo, hi = s * SHARD, min((s + 1) * SHARD, args.n_sets)
        block = thetas[lo:hi]
        ivs = np.empty((len(block), len(K_GRID)), dtype=np.float32)
        for i, th in enumerate(block):
            ivs[i] = price_one(th, args.paths, args.seed + lo + i, device)
        np.savez_compressed(OUT_DIR / f"shard_{s:04d}.npz",
                            theta=block.astype(np.float32), iv=ivs,
                            k_grid=K_GRID.astype(np.float32),
                            paths=args.paths)
        done.add(s)
        rate = (len(done) * SHARD) / max(time.perf_counter() - t0, 1e-9)
        eta_s = (n_shards - len(done)) * SHARD / max(rate, 1e-9)
        nan_frac = float(np.isnan(ivs).mean())
        print(f"  shard {s:04d}  [{len(done)}/{n_shards}]  "
              f"{rate:,.0f} sets/s-equivalent  ETA {eta_s / 3600:.1f} h  "
              f"NaN {nan_frac:.1%}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
