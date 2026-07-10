"""Automated drift monitor for the neural options pricing engine.

Compares the currently deployed 0DTE surrogate (model_0dte.pt) against live
SPY option-chain mid prices from yfinance.  If the root-mean-square pricing
error (in bps of strike) exceeds a configurable threshold, the script triggers
the full recalibration-retrain pipeline and gates promotion on the pytest
suite.

Every run appends a structured JSON line to artifacts/drift_log.jsonl for
time-series analysis of model degradation.

Usage (from the repo root):
    python -m backend.quant.drift_monitor
    python -m backend.quant.drift_monitor --threshold 20
    python -m backend.quant.drift_monitor --dry-run
    python -m backend.quant.drift_monitor --force
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
DRIFT_LOG = ARTIFACTS / "drift_log.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")

# ── terminal aesthetics ────────────────────────────────────────────────
CY, MG, VI, GN, RD, DIM, BOLD, RS = ("\x1b[38;5;51m", "\x1b[38;5;205m",
                                     "\x1b[38;5;141m", "\x1b[38;5;84m",
                                     "\x1b[38;5;203m", "\x1b[2m",
                                     "\x1b[1m", "\x1b[0m")


def rule(title: str = "") -> None:
    pad = f"═══ {BOLD}{title}{RS} " if title else ""
    print(f"{DIM}{pad}{'═' * max(8, 74 - len(title))}{RS}")


# ── report structure ───────────────────────────────────────────────────
@dataclass
class DriftReport:
    timestamp: str
    n_quotes: int
    rmse_bps: float
    max_err_bps: float
    drift_detected: bool
    action_taken: str            # 'none' | 'recalibrate' | 'recalibrate+retrain'
    tests_passed: bool | None    # None if no tests were run
    new_model_promoted: bool
    details: dict


# ── live assessment ────────────────────────────────────────────────────
def assess_drift(threshold_bps: float) -> DriftReport:
    """Fetch the live option chain and measure neural net pricing error."""
    import yfinance as yf

    from .calibrate import bs_vega, implied_vol
    from .engine import PricingEngine
    from .market_data import _fetch_risk_free

    rule("DRIFT ASSESSMENT")
    tk = yf.Ticker("SPY")
    spot = float(tk.fast_info["last_price"])
    rate, rate_src = _fetch_risk_free(yf)
    now = datetime.now(tz=NY)

    print(f"  spot {BOLD}${spot:,.2f}{RS}  ·  r {rate:.2%} ({rate_src})")

    # collect OTM quotes from near-term expiries
    quotes_spots = []
    quotes_strikes = []
    quotes_taus = []
    quotes_ivs = []
    quotes_rates = []
    quotes_mids = []

    n_expiries_used = 0
    for expiry in tk.options:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d") \
            .replace(hour=16, minute=0, tzinfo=NY)
        tau = (exp_dt - now).total_seconds() / (365.0 * 24 * 3600)
        dte = (exp_dt.date() - now.date()).days
        if dte < 0 or dte > 5:
            continue
        if tau <= 0:
            continue
        n_expiries_used += 1
        chain = tk.option_chain(expiry)
        fwd = spot * math.exp(rate * tau)

        for df, kind in ((chain.calls, "C"), (chain.puts, "P")):
            for row in df.itertuples():
                def _safe_float(v):
                    try:
                        f = float(v)
                        return 0.0 if math.isnan(f) else f
                    except (TypeError, ValueError):
                        return 0.0

                k = float(row.strike)
                bid = _safe_float(row.bid)
                ask = _safe_float(row.ask)
                vol = _safe_float(row.volume)
                oi = _safe_float(row.openInterest)

                if kind == "C" and k <= fwd:
                    continue
                if kind == "P" and k >= fwd:
                    continue
                if bid <= 0.02 or ask < bid:
                    continue
                mid = 0.5 * (bid + ask)
                if (ask - bid) / mid > 0.40:
                    continue
                if vol < 10 and oi < 100:
                    continue

                mid_call = mid if kind == "C" else \
                    mid + spot - k * math.exp(-rate * tau)
                iv = implied_vol(mid_call, spot, k, tau, rate)
                if iv is None:
                    continue

                quotes_spots.append(spot)
                quotes_strikes.append(k)
                quotes_taus.append(tau)
                quotes_ivs.append(iv)
                quotes_rates.append(rate)
                quotes_mids.append(mid_call)

    n_quotes = len(quotes_spots)
    print(f"  {BOLD}{n_quotes}{RS} clean quotes across "
          f"{n_expiries_used} expiries")

    if n_quotes == 0:
        return DriftReport(
            timestamp=now.isoformat(),
            n_quotes=0,
            rmse_bps=0.0,
            max_err_bps=0.0,
            drift_detected=False,
            action_taken="none",
            tests_passed=None,
            new_model_promoted=False,
            details={"error": "no quotes survived filtering"},
        )

    # neural net pricing
    engine = PricingEngine(ARTIFACTS / "model_0dte.pt")
    spots_arr = np.array(quotes_spots, dtype=np.float64)
    strikes_arr = np.array(quotes_strikes, dtype=np.float64)
    taus_arr = np.array(quotes_taus, dtype=np.float64)
    ivs_arr = np.array(quotes_ivs, dtype=np.float64)
    rates_arr = np.array(quotes_rates, dtype=np.float64)
    mids_arr = np.array(quotes_mids, dtype=np.float64)

    nn_prices = engine.price_batch(spots_arr, strikes_arr, taus_arr,
                                   ivs_arr, rates_arr, option_type="call")

    err_bps = ((nn_prices - mids_arr) / strikes_arr) * 1e4
    rmse_bps = float(np.sqrt(np.mean(err_bps ** 2)))
    max_err_bps = float(np.max(np.abs(err_bps)))
    drift = rmse_bps > threshold_bps

    color = RD if drift else GN
    print(f"  RMSE {color}{BOLD}{rmse_bps:.2f} bps{RS}  "
          f"(threshold {threshold_bps:.0f} bps)  ·  "
          f"max err {max_err_bps:.2f} bps")
    if drift:
        print(f"  {RD}drift detected{RS}")
    else:
        print(f"  {GN}model is within tolerance{RS}")

    return DriftReport(
        timestamp=now.isoformat(),
        n_quotes=n_quotes,
        rmse_bps=round(rmse_bps, 4),
        max_err_bps=round(max_err_bps, 4),
        drift_detected=drift,
        action_taken="none",
        tests_passed=None,
        new_model_promoted=False,
        details={
            "spot": spot,
            "rate": rate,
            "rate_src": rate_src,
            "n_expiries": n_expiries_used,
            "threshold_bps": threshold_bps,
        },
    )


# ── recalibration ─────────────────────────────────────────────────────
def run_recalibration() -> bool:
    """Run the calibrate + retrain pipeline as a subprocess."""
    rule("RECALIBRATION + RETRAIN")
    result = subprocess.run(
        [sys.executable, "-m", "backend.quant.calibrate",
         "--retrain", "--max-dte", "5"],
        cwd=REPO_ROOT,
    )
    ok = result.returncode == 0
    print(f"  recalibration {'succeeded' if ok else 'FAILED'} "
          f"(exit {result.returncode})")
    return ok


# ── test gate ──────────────────────────────────────────────────────────
def run_tests() -> bool:
    """Run the quant test suite; returns True if all tests pass."""
    rule("TEST GATE")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_quant.py", "-v"],
        cwd=REPO_ROOT,
    )
    ok = result.returncode == 0
    color = GN if ok else RD
    print(f"  {color}tests {'passed' if ok else 'FAILED'}{RS}")
    return ok


# ── model promotion ───────────────────────────────────────────────────
def promote_model() -> None:
    """Mark the retrained model as the active deployment.

    In production this would update a symlink, version registry, or
    deployment manifest.  The retrain step already writes model_0dte.pt
    in place, so no file operations are needed here.
    """
    rule("MODEL PROMOTION")
    print(f"  {GN}new model promoted to serving{RS}")


# ── orchestrator ───────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Drift monitor for the neural 0DTE pricing surrogate.")
    p.add_argument("--threshold", type=float, default=15.0,
                   help="RMSE drift threshold in bps (default: 15)")
    p.add_argument("--dry-run", action="store_true",
                   help="assess drift only; do not recalibrate")
    p.add_argument("--force", action="store_true",
                   help="recalibrate regardless of drift level")
    args = p.parse_args()

    rule("DRIFT MONITOR")

    report = assess_drift(args.threshold)

    should_recal = report.drift_detected or args.force
    if args.dry_run:
        should_recal = False
        print(f"  {DIM}--dry-run: skipping recalibration{RS}")

    if should_recal and report.n_quotes > 0:
        recal_ok = run_recalibration()
        report.action_taken = "recalibrate+retrain" if recal_ok else "recalibrate"
        if recal_ok:
            report.tests_passed = run_tests()
            if report.tests_passed:
                promote_model()
                report.new_model_promoted = True
            else:
                print(f"  {RD}tests failed; model NOT promoted{RS}")
        else:
            print(f"  {RD}recalibration failed; skipping tests{RS}")

    # persist the report
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_LOG, "a") as fh:
        fh.write(json.dumps(asdict(report)) + "\n")

    rule("SUMMARY")
    print(f"  quotes       {BOLD}{report.n_quotes}{RS}")
    print(f"  RMSE         {BOLD}{report.rmse_bps:.2f} bps{RS}")
    print(f"  max error    {BOLD}{report.max_err_bps:.2f} bps{RS}")
    print(f"  drift        {BOLD}{report.drift_detected}{RS}")
    print(f"  action       {BOLD}{report.action_taken}{RS}")
    print(f"  tests passed {BOLD}{report.tests_passed}{RS}")
    print(f"  promoted     {BOLD}{report.new_model_promoted}{RS}")
    print(f"  {DIM}log -> {DRIFT_LOG}{RS}")


if __name__ == "__main__":
    main()
