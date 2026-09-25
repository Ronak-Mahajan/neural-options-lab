"""How much the geometric-Asian control variate cuts the Monte Carlo error.

For each contract, `backend.quant.monte_carlo.price_asian_mc` prices the same
arithmetic Asian call `--reps` times on seeds `base, base + 1, ...`, once with
the control variate and once without. Antithetic sampling is on in both arms
(the engine always uses it), so the ratio isolates the control variate. The
reported factor is the ratio of the empirical standard deviations of the two
sets of prices across replications, not a ratio of reported standard errors,
which would treat antithetic pairs as independent. A 2,000-draw bootstrap over
replications gives the factor's own standard error.

The defaults are the settings behind the figures quoted in README.md and on
the methodology page: 300 replications, 50 time steps, seed bases 500000 and
700000, the at-the-money one-year call at 20% vol and a 5% rate at 5,000 and
20,000 paths, a three-month at-the-money call at 20% vol, and three further
points of the trained box.

    python scripts/variance_reduction.py            # writes docs/variance_reduction.json

It runs in about a minute on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quant.monte_carlo import price_asian_mc  # noqa: E402

# (spot, strike, maturity in years, sigma, rate, n_paths)
CASES = [
    (100, 100, 1.0, 0.2, 0.05, 5000),
    (100, 100, 1.0, 0.2, 0.05, 20000),
    (100, 100, 0.25, 0.2, 0.05, 5000),
    (100, 100, 1.0, 0.6, 0.05, 5000),
    (80, 100, 0.25, 0.3, 0.03, 5000),
    (130, 100, 2.0, 0.7, 0.03, 5000),
]


def factor(spot, strike, maturity, sigma, rate, n_paths, n_steps, reps, base):
    kw = dict(n_paths=n_paths, n_steps=n_steps)
    plain = np.array([price_asian_mc(spot, strike, maturity, sigma, rate, seed=base + i,
                                     control_variate=False, **kw).price for i in range(reps)])
    cv = np.array([price_asian_mc(spot, strike, maturity, sigma, rate, seed=base + i,
                                  control_variate=True, **kw).price for i in range(reps)])
    ratio = float(plain.std(ddof=1) / cv.std(ddof=1))
    idx = np.random.default_rng(1).integers(0, reps, (2000, reps))
    boot = plain[idx].std(1, ddof=1) / cv[idx].std(1, ddof=1)
    return ratio, float(boot.std(ddof=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seed-bases", default="500000,700000",
                   help="comma-separated first seeds, one replication set each")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "variance_reduction.json")
    a = p.parse_args()

    rows = []
    t0 = time.time()
    for base in (int(s) for s in a.seed_bases.split(",")):
        for (S, K, T, sig, r, n) in CASES:
            ratio, se = factor(S, K, T, sig, r, n, a.n_steps, a.reps, base)
            rows.append(dict(seed_base=base, S=S, K=K, T=T, sigma=sig, rate=r, n_paths=n,
                             n_steps=a.n_steps, reps=a.reps, sd_ratio=ratio, sd_ratio_boot_se=se))
            print(f"base {base}  S={S} K={K} T={T} sigma={sig} r={r} n={n}: "
                  f"{ratio:.2f}x (bootstrap SE {se:.2f})", flush=True)
    out = {"protocol": ("ratio of empirical SDs of price_asian_mc across replications, "
                        "control_variate False vs True, antithetic in both arms"),
           "command": "python scripts/variance_reduction.py",
           "rows": rows}
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
