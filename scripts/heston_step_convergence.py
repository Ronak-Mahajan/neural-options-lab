"""Time-step convergence of the Heston full-truncation Euler Monte Carlo.

The scheme is `backend.quant.heston.heston_mc_call`'s full-truncation Euler
(Lord, Koekkoek & van Dijk 2010), with the same update order, run at 50, 100,
200, 400 and 800 steps on common random numbers: one draw of normals on the
800-step grid, summed in blocks of m and divided by sqrt(m) for each coarser
level, so every level sees the same Brownian path. The COS price from
`backend.quant.heston.heston_call` is the exact answer. Parameters are the
Fang & Oosterlee (2008) set used in docs/heston_reference.md, which violates
the Feller condition, at S = 100, r = 2%, q = 0 and K = 90, 100, 110.

Standard errors come from 40 independent batches. For every pair of levels
the difference is computed batch by batch, so its standard error reflects the
common random numbers. Observed orders and Richardson values use each
consecutive triplet of levels.

The defaults are the settings behind docs/heston_reference.md, section (e):

    python scripts/heston_step_convergence.py      # writes docs/heston_step_convergence.json

It takes 5 to 11 minutes on 6 to 8 CPU threads.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.heston import heston_call  # noqa: E402

FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, sigma_v=0.5751, rho=-0.5711)
S, R, Q = 100.0, 0.02, 0.0
STRIKES = np.array([90.0, 100.0, 110.0])
NF = 800
LEVELS = [50, 100, 200, 400, 800]
NB = 40
CHUNK = 100_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths-millions", type=float, default=2.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "heston_step_convergence.json")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    T = a.T
    n_paths = int(a.paths_millions * 1e6)
    per = n_paths // NB
    cos = np.asarray(heston_call(S, STRIKES, T, R, Q, **FO))
    gen = torch.Generator().manual_seed(a.seed)
    Kt = torch.tensor(STRIKES, dtype=torch.float64)
    rp = math.sqrt(1 - FO["rho"] ** 2)
    sums = np.zeros((NB, len(LEVELS), len(STRIKES)))
    t0 = time.time()
    for b in range(NB):
        left = per
        while left > 0:
            n = min(CHUNK, left)
            left -= n
            x = {L: torch.full((n,), math.log(S), dtype=torch.float64) for L in LEVELS}
            v = {L: torch.full((n,), FO["v0"], dtype=torch.float64) for L in LEVELS}
            acc_v = {L: torch.zeros(n, dtype=torch.float64) for L in LEVELS}
            acc_i = {L: torch.zeros(n, dtype=torch.float64) for L in LEVELS}
            for i in range(NF):
                zv = torch.randn(n, dtype=torch.float64, generator=gen)
                zi = torch.randn(n, dtype=torch.float64, generator=gen)
                for L in LEVELS:
                    m = NF // L
                    acc_v[L] += zv
                    acc_i[L] += zi
                    if (i + 1) % m == 0:
                        dt = T / L
                        sq = math.sqrt(dt)
                        Zv = acc_v[L] / math.sqrt(m)
                        Zi = acc_i[L] / math.sqrt(m)
                        Zs = FO["rho"] * Zv + rp * Zi
                        vp = torch.clamp(v[L], min=0.0)
                        x[L] += (R - Q - 0.5 * vp) * dt + torch.sqrt(vp) * sq * Zs
                        v[L] += FO["kappa"] * (FO["theta"] - vp) * dt + FO["sigma_v"] * torch.sqrt(vp) * sq * Zv
                        acc_v[L].zero_()
                        acc_i[L].zero_()
            for li, L in enumerate(LEVELS):
                ST = torch.exp(x[L])
                sums[b, li] += torch.clamp(ST[:, None] - Kt[None, :], min=0).sum(0).numpy()
        if b % 10 == 9:
            print(f"batch {b + 1}/{NB} {time.time() - t0:.0f}s", flush=True)

    pr = sums / per * math.exp(-R * T)                     # (NB, levels, strikes)
    mean = pr.mean(0)
    se = pr.std(0, ddof=1) / math.sqrt(NB)
    err = mean - cos
    res = {"T": T, "S": S, "rate": R, "div": Q, "params": FO, "n_paths": per * NB,
           "n_batches": NB, "seed": a.seed, "levels": LEVELS, "K": STRIKES.tolist(),
           "cos": cos.tolist(), "mc": mean.tolist(), "se": se.tolist(),
           "err_vs_cos": err.tolist(), "z_vs_cos": (err / se).tolist(),
           "seconds": round(time.time() - t0, 1),
           "command": "python scripts/heston_step_convergence.py"}
    # every pair of levels, fine minus coarse, with a batch-wise (common random numbers) SE
    diffs = {}
    for i in range(len(LEVELS)):
        for j in range(i + 1, len(LEVELS)):
            dd = pr[:, j] - pr[:, i]
            diffs[f"{LEVELS[j]}-{LEVELS[i]}"] = {"diff": dd.mean(0).tolist(),
                                                 "se": (dd.std(0, ddof=1) / math.sqrt(NB)).tolist()}
    res["diffs"] = diffs
    trip = []
    for li in range(2, len(LEVELS)):
        f3, f2, f1 = mean[li - 2], mean[li - 1], mean[li]
        e32, e21 = f2 - f3, f1 - f2
        ratio = e32 / e21
        with np.errstate(invalid="ignore", divide="ignore"):
            order = np.where(ratio > 0, np.log(np.abs(ratio)) / math.log(2), np.nan)
            ext = f1 + e21 / (2 ** order - 1)
            gci = 1.25 * np.abs(e21 / f1) / (2 ** order - 1)
        trip.append({"levels": LEVELS[li - 2:li + 1], "p": order.tolist(), "richardson": ext.tolist(),
                     "richardson_minus_cos": (ext - cos).tolist(), "gci_fine": gci.tolist()})
    res["triplets"] = trip
    res["order_vs_exact"] = [np.log2(np.abs(err[i] / err[i + 1])).tolist() for i in range(len(LEVELS) - 1)]
    # NaN (an undefined observed order) is written as null so the file is strict JSON
    text = json.dumps(res, indent=1).replace("NaN", "null")
    a.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
