"""Time-step bias of the rough Bergomi ATM skew, with common random numbers.

One run draws the exact joint law of (dW, W~) on a fine grid with the engine's
own `backend.quant.rough_vol._joint_factor_unit` (float64) and builds every
coarser level from that same draw: coarse dW is the sum of the fine dW, W~ at
the coarse times is the fine W~ sampled there (exact, since W~_t does not
depend on the grid), and the independent normal is the block sum over
sqrt(m). Each level then applies the engine's left-point scheme. The
quantities are the ATM implied vol (k = 0 against the forward) and the
central-difference skew psi = (iv(+h) - iv(-h)) / 2h with the stencil h of
docs/atm_skew_term_structure.md. Standard errors come from 40 batches, and
level-to-level differences are taken batch by batch, so their standard errors
reflect the common random numbers. H, rho, xi and the rate are read from
artifacts/rough_calibration.json; eta is the calibrated value unless given.

One run per maturity, then a summary:

    python scripts/rb_step_convergence.py run --T-days 1   --paths-millions 4
    python scripts/rb_step_convergence.py run --T-days 5   --paths-millions 4
    python scripts/rb_step_convergence.py run --T-days 12  --paths-millions 4
    python scripts/rb_step_convergence.py run --T-days 45  --paths-millions 4
    python scripts/rb_step_convergence.py run --T-days 126 --paths-millions 8
    python scripts/rb_step_convergence.py run --T-days 126 --paths-millions 4 --eta 0.5
    python scripts/rb_step_convergence.py summarize

Each run writes docs/rb_step_convergence/rb_T{days}_nf{n_fine}_eta{eta}.json,
and `summarize` writes docs/rb_step_convergence/summary.json, the source of
the table in docs/atm_skew_term_structure.md, Section 5, caveat 4. A 4M-path
run takes 4 to 6 minutes on 6 to 8 CPU threads and the 8M-path run about 9.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.rough_vol import _joint_factor_unit  # noqa: E402

OUT_DIR = ROOT / "docs" / "rb_step_convergence"
N_BATCHES = 40
CHUNK = 25_000


def run(a: argparse.Namespace) -> None:
    cal = json.loads((ROOT / "artifacts" / "rough_calibration.json").read_text(encoding="utf-8"))
    T_days, NF, NP = float(a.T_days), int(a.n_fine), int(a.paths_millions * 1e6)
    torch.set_num_threads(a.threads)
    eta = float(a.eta) if a.eta is not None else float(cal["eta"])
    rho, H, xi, r = float(cal["rho"]), float(cal["H"]), float(cal["xi"]), float(cal["rate"])
    T = T_days / 252.0
    F = math.exp(r * T)
    h = max(0.25 * math.sqrt(xi) * math.sqrt(T), 0.002)
    ks = np.array([-h, 0.0, h])
    K = F * np.exp(ks)
    levels = [n for n in (25, 50, 100, 200, 400, 800) if n <= NF and NF % n == 0]
    L = torch.tensor(np.asarray(_joint_factor_unit(NF, H)), dtype=torch.float64)
    per_batch = NP // N_BATCHES
    gen = torch.Generator().manual_seed(a.seed)
    Kt = torch.tensor(K, dtype=torch.float64)

    def terminal(Zv_f, Wt_f_unit, Zi_f, n):
        m = NF // n
        dt = T / n
        nc = Zv_f.shape[0]
        Zv = Zv_f.view(nc, n, m).sum(-1) / math.sqrt(m)       # dW / sqrt(dt), coarse
        Zi = Zi_f.view(nc, n, m).sum(-1) / math.sqrt(m)
        Wt = Wt_f_unit[:, m - 1::m] * (T / NF) ** H             # W~ at the coarse times
        t = torch.arange(1, n + 1, dtype=torch.float64) * dt
        V = xi * torch.exp(eta * Wt - 0.5 * eta ** 2 * t ** (2 * H))
        V = torch.cat([torch.full((nc, 1), xi, dtype=torch.float64), V[:, :-1]], 1)
        Zs = rho * Zv + math.sqrt(1 - rho ** 2) * Zi
        return torch.exp(torch.sum((r - 0.5 * V) * dt, 1) + torch.sum(torch.sqrt(V) * Zs * math.sqrt(dt), 1))

    sums = np.zeros((N_BATCHES, len(levels), 3))
    t0 = time.time()
    for b in range(N_BATCHES):
        left = per_batch
        while left > 0:
            nc = min(CHUNK, left)
            left -= nc
            noise = torch.randn(nc, 2 * NF, dtype=torch.float64, generator=gen)
            joint = noise @ L.T
            Zv_f, Wt_f = joint[:, :NF], joint[:, NF:]
            Zi_f = torch.randn(nc, NF, dtype=torch.float64, generator=gen)
            for li, n in enumerate(levels):
                ST = terminal(Zv_f, Wt_f, Zi_f, n)
                sums[b, li] += torch.clamp(ST[:, None] - Kt[None, :], min=0).sum(0).numpy()
        if b % 10 == 9:
            print(f"batch {b + 1}/{N_BATCHES} {time.time() - t0:.0f}s", flush=True)
    prices = sums / per_batch * math.exp(-r * T)                  # (batches, levels, 3)

    def bs(sig, k):
        d1 = (-k + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
        return math.exp(-r * T) * (F * norm.cdf(d1) - F * math.exp(k) * norm.cdf(d1 - sig * math.sqrt(T)))

    def iv(p, k):
        return brentq(lambda s: bs(s, k) - p, 1e-4, 5.0, xtol=1e-12)

    def qoi(pr):
        out = []
        for li in range(len(levels)):
            v = [iv(pr[li, j], ks[j]) for j in range(3)]
            out.append((v[1], (v[2] - v[0]) / (2 * h), pr[li, 1] * 1e4))
        return np.array(out)                                      # (levels, [atm_iv, psi, atm_price_bp])

    full = qoi(prices.mean(0))
    per = np.array([qoi(prices[b]) for b in range(N_BATCHES)])
    se = per.std(0, ddof=1) / math.sqrt(N_BATCHES)
    res = {"T_days": T_days, "eta": eta, "rho": rho, "H": H, "xi": xi, "rate": r, "h": h,
           "n_paths": per_batch * N_BATCHES, "n_batches": N_BATCHES, "seed": a.seed, "levels": levels,
           "seconds": round(time.time() - t0, 1),
           "qoi_names": ["atm_iv", "psi", "atm_price_bp"], "value": full.tolist(), "se": se.tolist()}
    diffs = {}
    for li in range(1, len(levels)):
        d = per[:, li] - per[:, li - 1]
        diffs[f"{levels[li]}-{levels[li - 1]}"] = {"diff": (full[li] - full[li - 1]).tolist(),
                                                   "se": (d.std(0, ddof=1) / math.sqrt(N_BATCHES)).tolist()}
    res["diffs"] = diffs
    trip = []
    for li in range(2, len(levels)):
        f3, f2, f1 = full[li - 2], full[li - 1], full[li]            # coarse, medium, fine
        e32, e21 = f2 - f3, f1 - f2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = e32 / e21
            order = np.where(ratio > 0, np.log(np.abs(ratio)) / math.log(2), np.nan)
            fext = f1 + e21 / (2 ** order - 1)
            gci = 1.25 * np.abs(e21 / f1) / (2 ** order - 1)
        trip.append({"levels": [levels[li - 2], levels[li - 1], levels[li]], "p": order.tolist(),
                     "monotone": (ratio > 0).tolist(), "richardson": fext.tolist(), "gci_fine": gci.tolist()})
    res["triplets"] = trip
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rb_T{int(T_days)}_nf{NF}_eta{eta:.3g}.json"
    out.write_text(json.dumps(res, indent=1).replace("NaN", "null") + "\n", encoding="utf-8")
    print(f"wrote {out} ({res['seconds']}s)")


def summarize(a: argparse.Namespace) -> None:
    cal = json.loads((ROOT / "artifacts" / "rough_calibration.json").read_text(encoding="utf-8"))
    rows = []
    for f in sorted(glob.glob(str(OUT_DIR / f"rb_T*_nf{a.n_fine}_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        lv, v, s = r["levels"], r["value"], r["se"]
        i50 = lv.index(50)
        t = r["triplets"][-1]
        psi_ext, iv_ext = t["richardson"][1], t["richardson"][0]
        rows.append(dict(file=Path(f).name, T=r["T_days"], eta=r["eta"], paths=r["n_paths"],
                         psi_by_level={str(n): v[i][1] for i, n in enumerate(lv)},
                         psi50=v[i50][1], psi50_se=s[i50][1], psi400=v[-1][1], psi_ext=psi_ext,
                         psi_bias50=v[i50][1] - psi_ext, psi_bias50_rel=(v[i50][1] - psi_ext) / abs(psi_ext),
                         iv50=v[i50][0], iv_ext=iv_ext, iv_bias50_volpts=(v[i50][0] - iv_ext) * 100,
                         p_psi=[x["p"][1] for x in r["triplets"]], p_iv=[x["p"][0] for x in r["triplets"]],
                         gci_psi_fine=t["gci_fine"][1], gci_iv_fine=t["gci_fine"][0],
                         min_consecutive_psi_diff_z=min(abs(d["diff"][1]) / d["se"][1] for d in r["diffs"].values())))
    calibrated = sorted([x for x in rows if abs(x["eta"] - float(cal["eta"])) < 1e-9], key=lambda x: x["T"])
    slopes = []
    for lo, hi in ((1, 45), (1, 126), (45, 126), (5, 45)):
        pa = [x for x in calibrated if x["T"] == lo]
        pb = [x for x in calibrated if x["T"] == hi]
        if pa and pb:
            pa, pb = pa[0], pb[0]
            span = math.log(hi) - math.log(lo)
            s50 = (math.log(abs(pb["psi50"])) - math.log(abs(pa["psi50"]))) / span
            sex = (math.log(abs(pb["psi_ext"])) - math.log(abs(pa["psi_ext"]))) / span
            slopes.append(dict(window_days=[lo, hi], slope_n50=s50, slope_extrapolated=sex, shift=sex - s50))
            print(f"two-point slope {lo}-{hi} d: n=50 {s50:.4f}  extrapolated {sex:.4f}  shift {sex - s50:+.4f}")
    for x in rows:
        print(f"T={x['T']:g} eta={x['eta']:.3g} paths={x['paths']:,}: psi50 {x['psi50']:.4f} "
              f"ext {x['psi_ext']:.4f} bias {x['psi_bias50_rel']:.1%} iv bias {x['iv_bias50_volpts']:.3f} vp")
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps({"rows": rows, "two_point_slopes": slopes},
                              indent=1).replace("NaN", "null") + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="one maturity")
    pr.add_argument("--T-days", type=float, required=True)
    pr.add_argument("--n-fine", type=int, default=400)
    pr.add_argument("--paths-millions", type=float, default=4.0)
    pr.add_argument("--threads", type=int, default=8)
    pr.add_argument("--eta", type=float, default=None, help="default: the calibrated eta")
    pr.add_argument("--seed", type=int, default=20260924)
    ps = sub.add_parser("summarize", help="collect the runs into summary.json")
    ps.add_argument("--n-fine", type=int, default=400)
    a = p.parse_args()
    run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    main()
