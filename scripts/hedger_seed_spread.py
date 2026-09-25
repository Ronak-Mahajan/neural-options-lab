"""Training-seed spread of the rough-measure hedging result.

The served rough-measure policy, artifacts/hedger_rbergomi_jumps.pt, is one
training run (torch seed 21). This script retrains the same recipe on other
seeds (`backend.quant.hedging.train`: 4,000 iterations of 2,048 paths under
rbergomi_jumps, the measure's own training box) and scores every policy with
the served-run protocol, `HedgingEngine.compare` on rbergomi_jumps with its
default 5 x 3,000 paths, on two evaluation seed sets: 17-21 (the served run's)
and 1017-1021. Costs are 10, 50, 100 and 200 bp. For each (training seed,
evaluation set, cost) it records every hedger's CVaR95 with its bootstrap
standard error and the paired deep-minus-delta and deep-minus-Whalley-Wilmott
differences.

    python scripts/hedger_seed_spread.py train --seed 22    # about 18 minutes on 4 threads
    python scripts/hedger_seed_spread.py train --seed 23
    python scripts/hedger_seed_spread.py train --seed 24
    python scripts/hedger_seed_spread.py eval               # writes docs/deep_hedging_seed_spread.json

Retrained checkpoints go to --ckpt-dir (default: a directory under the system
temp folder), never to artifacts/. The table in docs/deep_hedging_regimes.md
("Training-seed spread") is read from the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.quant.hedging as H  # noqa: E402

MEASURE = "rbergomi_jumps"
EVAL_SETS = ((17, 18, 19, 20, 21), (1017, 1018, 1019, 1020, 1021))
COSTS = (0.001, 0.005, 0.01, 0.02)
DEFAULT_CKPT_DIR = Path(tempfile.gettempdir()) / "neural-options-lab-hedger-seeds"


def ckpt_name(seed: int) -> str:
    return f"hedger_{MEASURE}_s{seed}.pt"


def train(a: argparse.Namespace) -> None:
    torch.set_num_threads(a.threads)
    a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    served = H.ARTIFACTS
    H.ARTIFACTS = a.ckpt_dir          # train() writes to ARTIFACTS / out_name
    try:
        meta = H.train(iters=a.iters, batch=a.batch, seed=a.seed, measure=MEASURE,
                       out_name=ckpt_name(a.seed), log_every=500)
    finally:
        H.ARTIFACTS = served
    print(json.dumps({k: meta[k] for k in ("seed", "iters", "train_seconds")}))


def evaluate(a: argparse.Namespace) -> None:
    torch.set_num_threads(a.threads)
    ckpts = {21: H.ARTIFACTS / f"hedger_{MEASURE}.pt"}
    for s in a.seeds:
        path = a.ckpt_dir / ckpt_name(s)
        if not path.exists():
            raise SystemExit(f"missing {path}; run: python scripts/hedger_seed_spread.py train --seed {s}")
        ckpts[s] = path
    p = H.rough_measure_params(MEASURE)
    sig, rate = p["xi"] ** 0.5, p["rate"]
    rows = []
    for s, path in ckpts.items():
        eng = H.HedgingEngine(path)
        assert int(eng.meta["seed"]) == s and eng.meta["train_measure"] == MEASURE, eng.meta
        for seeds in EVAL_SETS:
            for cost in COSTS:
                o = eng.compare(sig, rate, cost, primary=MEASURE, measures=(MEASURE,), seeds=seeds)
                pr = o["paired_bootstrap"]["pairs"]
                row = {"train_seed": s, "eval_seeds": list(seeds), "cost": cost}
                for k in ("deep", "delta", "whalley_wilmott", "linear"):
                    row[k] = {"cvar_bp": o[k]["cvar95"] * 1e4, "se_bp": o[k]["cvar95_se"] * 1e4}
                row["deep_minus_delta_bp"] = pr["deep|delta"]["diff"] * 1e4
                row["deep_minus_delta_se_bp"] = pr["deep|delta"]["se"] * 1e4
                row["deep_minus_ww_bp"] = pr["deep|whalley_wilmott"]["diff"] * 1e4
                row["deep_minus_ww_se_bp"] = pr["deep|whalley_wilmott"]["se"] * 1e4
                rows.append(row)
                print(f"train {s} eval {seeds[0]}.. cost {cost}: deep {row['deep']['cvar_bp']:.1f} "
                      f"delta {row['delta']['cvar_bp']:.1f} ww {row['whalley_wilmott']['cvar_bp']:.1f} "
                      f"deep-delta {row['deep_minus_delta_bp']:+.1f} +- {row['deep_minus_delta_se_bp']:.1f}",
                      flush=True)
    out = {"measure": MEASURE, "sigma": sig, "rate": rate,
           "train_recipe": {"iters": 4000, "batch": 2048, "measure": MEASURE},
           "command": "python scripts/hedger_seed_spread.py eval", "rows": rows}
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train", help="retrain the rough-measure policy on one seed")
    pt.add_argument("--seed", type=int, required=True)
    pt.add_argument("--iters", type=int, default=4000)
    pt.add_argument("--batch", type=int, default=2048)
    pt.add_argument("--threads", type=int, default=4)
    pt.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    pe = sub.add_parser("eval", help="score the served policy and the retrained ones")
    pe.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",") if x],
                    default=[22, 23, 24], help="retrained seeds to include besides the served 21")
    pe.add_argument("--threads", type=int, default=8)
    pe.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    pe.add_argument("--out", type=Path, default=ROOT / "docs" / "deep_hedging_seed_spread.json")
    a = p.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
