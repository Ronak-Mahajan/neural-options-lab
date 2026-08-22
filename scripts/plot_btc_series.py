"""Timeline of the Saturday BTC jump-stability series.

Parses every point_*.log and replay_*.log the series produced, extracts
the two identified quantities per snapshot (the MC jumps-vs-diffusive
delta and the jump-variance cumulant lam*(mu_j^2+sig_j^2)), and draws
both against UTC time. Points whose jump fit railed at lam~0 are drawn
hollow: those runs predate the multi-start fix in MapCalibrator.fit and
were retracted (the replay of the archived surface with the fixed fit is
plotted in their place).

    python -m scripts.plot_btc_series
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "data" / "btc_series"

HEAD = re.compile(r"BTC (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}):\d{2}Z")
JUMP = re.compile(r"lam=([\d.]+) mu_j=([+-][\d.]+) sig_j=([\d.]+)")
DELTA = re.compile(r"FULL surface: ([+-][\d.]+) vp")
RAIL = re.compile(r"WARNING: lam at 0% of bound")


def parse(path: Path):
    text = path.read_text(encoding="utf-8")
    head, jump, delta = HEAD.search(text), JUMP.search(text), DELTA.search(text)
    if not (head and jump and delta):
        return None
    lam, mu, sig = map(float, jump.groups())
    return {
        "utc": head.group(1).split("T")[1],
        "delta": float(delta.group(1)),
        "cumulant": lam * (mu * mu + sig * sig),
        "railed": bool(RAIL.search(text)),
        "name": path.stem,
    }


def main() -> None:
    rows = []
    for p in sorted(SERIES.glob("*.log")):
        r = parse(p)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["utc"])
    for r in rows:
        print(f"{r['utc']}Z  delta {r['delta']:+.3f} vp  "
              f"cumulant {r['cumulant']:.4f}/yr"
              f"{'  [retracted: lam railed at 0]' if r['railed'] else ''}"
              f"  ({r['name']})")

    def mins(utc: str) -> int:
        h, m = utc.split(":")
        return int(h) * 60 + int(m)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    good = [r for r in rows if not r["railed"]]
    bad = [r for r in rows if r["railed"]]
    for ax, key, label in ((ax1, "delta", "jumps vs diffusive,\nMC (vp)"),
                           (ax2, "cumulant",
                            "jump variance\nlam*(mu^2+sig^2) (1/yr)")):
        ax.plot([mins(r["utc"]) for r in good], [r[key] for r in good],
                "o-", color="#1252b3", zorder=3)
        if bad:
            ax.plot([mins(r["utc"]) for r in bad], [r[key] for r in bad],
                    "o", mfc="none", mec="#b3321b", zorder=3)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    ax1.axhline(0.15, ls="--", lw=0.8, color="#777",
                label="the script's noise threshold for its verdict")
    ax1.legend(fontsize=8, loc="center right")
    ticks = []
    for t in sorted({mins(r["utc"]) for r in rows}):
        if not ticks or t - ticks[-1] >= 25:    # thin clustered labels
            ticks.append(t)
    labels = [f"{t // 60:02d}:{t % 60:02d}" for t in ticks]
    ax2.set_xticks(ticks, labels, fontsize=8)
    ax2.set_xlabel("UTC, Saturday 2026-08-22")
    fig.suptitle("BTC full-surface jump premium through the day\n"
                 "hollow: retracted pre-fix runs whose jump fit missed the "
                 "basin (see SERIES.md)", fontsize=10)
    fig.tight_layout()
    out = SERIES / "series_timeline.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
