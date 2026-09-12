"""Fast checks for scripts/joint_skew_refit.py (whole file < 60 s).

The full profile / joint fit / Monte Carlo run takes ~20 minutes; these tests
exercise the building blocks at a small budget:

  (a) the map-based ATM skew at the served parameters is negative at every
      listed expiry and steepens toward short tau (|psi| decreasing in tau,
      power-law exponent < 0);
  (b) the central difference agrees with a step half as wide to within a
      documented tolerance (measured 0.006-0.015 at the served point, i.e.
      under 1.5% of |psi|; the Richardson value agrees with the finer step
      to ~0.002) - the discretisation is not what drives the results;
  (c) MapBatch reproduces MapCalibrator.loss bit-for-bit, and lambda = 0
      reproduces MapCalibrator.fit's optimum within tolerance on a small
      quote subset (same DE settings and seed, population evaluated batched);
  (d) the Pareto helper's monotonicity flags are correct on exact synthetic
      fronts, and a small real warm-started lambda chain is monotone
      (smile loss non-decreasing, skew chi2 non-increasing) within tolerance;
  (e) the analysis artifact loads, keeps rough_calibration.json's schema, has
      accepted = null, and its eta_interior flag agrees with the gate's
      bound-pinning rule applied to the stored eta.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant.calibrate import BOUNDS, PIN_FRAC  # noqa: E402
from backend.quant.calibrate_map import MapCalibrator, quotes_from_capture  # noqa: E402
from scripts import joint_skew_refit as jr  # noqa: E402

CAPTURES = sorted((ROOT / "data" / "surfaces" / "equity").glob("spy_*.json.gz"))
CAPTURE = ROOT / "data" / "surfaces" / "equity" / "spy_20260821T150017Z.json.gz"
if not CAPTURE.exists() and CAPTURES:
    CAPTURE = CAPTURES[0]
ARTIFACT = ROOT / "artifacts" / "rough_calibration_skewjoint.json"
DOCS_JSON = ROOT / "docs" / "joint_skew_refit.json"

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@pytest.fixture(scope="module")
def capture():
    if not CAPTURE.exists():
        pytest.skip("no committed SPY capture")
    return jr.CaptureSet(CAPTURE)


@pytest.fixture(scope="module")
def subset(capture):
    """Three shortest expiries, every 4th quote (~43 quotes): small enough
    for DE + Powell twice inside the test budget."""
    quotes, rate, meta = quotes_from_capture(CAPTURE)
    exps = sorted(set(q.expiry for q in quotes))[:3]
    sub = [q for q in quotes if q.expiry in exps][::4]
    return jr.CaptureSet.from_quotes(sub, rate, meta, capture.pricer, name="subset")


def test_map_skew_negative_and_steepening_at_served_params(capture):
    theta = jr.served_theta()
    sk = capture.stencil.skew(capture.pricer, theta)
    psi = sk["psi"]
    assert len(psi) >= 5
    assert np.all(np.isfinite(psi))
    assert np.all(psi < 0.0), psi
    # |psi| shrinks with tau: strictly, expiry by expiry
    assert np.all(np.diff(np.abs(psi)) < 0.0), psi
    f = jr.model_exponent(capture.taus, psi)
    assert f["b"] < 0.0, f
    assert -0.5 < f["b"] < -0.1, f
    # the stencil is centred on the forward (k0 = r tau, the market's and the
    # MC stencil's ATM), inside the map's k box, and the ATM vol is sane
    assert np.allclose(capture.stencil.k0, capture.rate * capture.taus)
    assert np.all(capture.stencil.k0 > 0.0)
    assert np.all(np.abs(capture.stencil.k0) + np.abs(capture.stencil.h * 2.0) < 0.1)
    assert np.all((sk["atm_iv"] > 0.05) & (sk["atm_iv"] < 0.5))


def test_central_difference_agrees_with_finer_step(capture):
    theta = jr.served_theta()
    coarse = capture.stencil.skew(capture.pricer, theta)
    fine = jr.SkewStencil(capture.taus, capture.stencil.h / 2.0,
                          capture.stencil.k0).skew(capture.pricer, theta)
    d = np.abs(coarse["psi"] - fine["psi"])
    # documented tolerance: 0.03 absolute and 2.5% relative (measured 0.006-0.015)
    assert np.all(d < 0.03), d
    assert np.all(d / np.abs(fine["psi"]) < 0.025), d / np.abs(fine["psi"])
    # Richardson extrapolation removes most of the O(h^2) term
    dr = np.abs(coarse["psi_richardson"] - fine["psi"])
    assert np.all(dr < 0.01), dr
    # and the map-based truncation estimate is of the same size as the change
    assert np.all(coarse["truncation"] < 0.03)


def test_batch_matches_map_calibrator_loss(capture):
    theta = jr.served_theta()
    assert capture.batch.smile_loss(theta) == capture.cal.loss(theta)
    assert abs(capture.batch.rmse_volpts(theta) - capture.cal.rmse_volpts(theta)) < 1e-9
    # A batched population evaluates member by member to within float32
    # precision. The pricing map runs in float32, and a batched matmul may pick
    # a different kernel from a single-row one: identical on the Windows/MKL
    # build this was written on, 5.8e-9 on a loss of 1.3 on the Linux CI
    # runner. Both are the same answer, so the bound is relative to the loss
    # at float32 resolution rather than an absolute 1e-9.
    thetas = np.array([theta, [2.0, -0.7, 0.15, 0.02], [1.0, -0.3, 0.4, 0.01]])
    smile, psi = capture.batch.evaluate(thetas)
    for th, s_ in zip(thetas, smile):
        ref = capture.cal.loss(th)
        assert abs(s_ - ref) <= 1e-7 * max(1.0, abs(ref)), (s_, ref)
    assert psi.shape == (3, capture.stencil.n)


def test_lambda_zero_reproduces_map_calibrator_optimum(subset):
    ref, _ = subset.cal.fit(seed=7)
    obj = jr.JointObjective(subset, 0.0)
    r = jr.joint_fit(obj, seed=7, extra_starts=[("served", jr.served_theta())])
    loss_ref = subset.cal.loss(ref)
    # same objective, same DE seed: the optima coincide up to Powell's tolerance
    assert abs(r["J"] - loss_ref) < 1e-4 * max(1.0, loss_ref), (r["J"], loss_ref)
    tol = np.array([0.02, 0.02, 0.02, 1e-3])
    assert np.all(np.abs(r["theta"] - ref) < tol), (r["theta"], ref)
    assert r["nfev_de"] == 64 * 61          # popsize 16 x 4 params, 60 generations + init


def test_pareto_helper_flags_monotonicity():
    exact = [{"lambda": l, "rmse_volpts": 1.0 + 0.1 * i, "smile_loss": 1.0 + 0.2 * i,
              "chi2": 100.0 / (1 + i), "b_expiries": -0.32 + 0.02 * i, "theta": [4, -0.5, 0.2, 0.01]}
             for i, l in enumerate((0, 0.1, 1, 10))]
    pf = jr.pareto_table(exact, b_market=-0.249, se_b_market=0.033)
    assert pf["smile_monotone"] and pf["chi2_monotone"]
    assert [r["lambda"] for r in pf["rows"]] == [0, 0.1, 1, 10]
    assert pf["knee_index"] is not None
    assert pf["first_within_1se_lambda"] == 1        # b = -0.28 is 0.031 from -0.249
    broken = [dict(r) for r in exact]
    broken[2]["smile_loss"] = 0.5                     # smile improving with lambda: impossible
    broken[1]["chi2"] = 500.0                         # chi2 worsening with lambda: impossible
    pf2 = jr.pareto_table(broken, b_market=-0.249, se_b_market=0.033)
    assert not pf2["smile_monotone"] and not pf2["chi2_monotone"]


def test_warm_started_lambda_chain_is_monotone(subset):
    from scipy.optimize import minimize
    theta = jr.served_theta()
    rows = []
    for lam in (0.0, 0.3, 30.0):
        obj = jr.JointObjective(subset, lam)
        r = minimize(obj, theta, method="Powell", bounds=jr.JOINT_BOUNDS,
                     options={"xtol": 1e-5, "ftol": 1e-8, "maxfev": 1500})
        theta = np.asarray(r.x, dtype=float)
        rows.append({"lambda": lam, **subset.evaluate(theta)})
    # Powell on a small quote subset converges to within ~0.02 in chi2 of the
    # optimum; the chain is monotone up to that optimizer noise.
    pf = jr.pareto_table(rows, b_market=-0.249, se_b_market=0.033, tol=0.1)
    assert pf["smile_monotone"], [r["smile_loss"] for r in rows]
    assert pf["chi2_monotone"], [r["chi2"] for r in rows]
    assert rows[-1]["chi2"] < rows[0]["chi2"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="analysis artifact not generated")
def test_artifact_schema_and_eta_interior_flag():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "artifacts" / "rough_calibration.json").read_text(encoding="utf-8"))
    missing = set(base) - set(art)
    assert not missing, missing
    assert art["accepted"] is None
    assert "analysis" in art["note"].lower()
    sts = art["skew_term_structure"]
    lo, hi = BOUNDS["eta"]
    w = PIN_FRAC * (hi - lo)
    interior = lo + w < art["eta"] < hi - w
    assert sts["eta_interior"] == interior == jr.eta_interior(art["eta"])
    assert lo <= art["eta"] <= hi and -1.0 <= art["rho"] <= 0.0 and 0.01 <= art["H"] <= 0.5
    assert abs(math.sqrt(art["xi"]) - art["sqrt_xi"]) < 1e-3
    for k in ("lambda", "exponent_ladder_mc", "chi2", "market_exponent"):
        assert k in sts
    if DOCS_JSON.exists():
        doc = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
        th = doc["recommended"]["theta"]
        assert abs(th[0] - art["eta"]) < 1e-3 and abs(th[2] - art["H"]) < 1e-3
        assert doc["recommended"]["lambda"] == sts["lambda"]


def test_polish_never_worse_than_start(subset):
    """The guarded polish returns the best of {start, bounded Powell, logit
    Powell}; in particular it can never end above its starting value, which
    scipy's bounded Powell alone can (measured at lambda = 10, see the
    module docstring)."""
    obj = jr.JointObjective(subset, 10.0)
    x0 = jr.served_theta()
    r = jr.polish(obj, x0, jr.JOINT_BOUNDS, {"maxfev": 600})
    f0 = obj(x0)
    assert r["fun"] <= f0 * (1 + 1e-12), (r["fun"], f0)
    assert r["fun"] <= min(r["bounded_fun"], r["logit_fun"]) * (1 + 1e-12)
    assert r["method"] in ("start", "bounded", "logit")
    assert abs(obj(r["x"]) - r["fun"]) < 1e-9
    lo, hi = zip(*jr.JOINT_BOUNDS)
    assert np.all(r["x"] >= np.array(lo)) and np.all(r["x"] <= np.array(hi))
    # the logit map round-trips inside the box
    z = jr.to_logit(x0)
    assert np.max(np.abs(jr.from_logit(z) - x0)) < 1e-9


def test_interior_candidate_and_contour_helpers():
    fits = {"0": {"lambda": 0, "eta_interior": False, "theta": [4, -.5, .2, .01], "rmse_volpts": 1.0,
                  "b_expiries": -0.38, "chi2": 1000.0, "smile_loss": 0.5},
            "0.1": {"lambda": 0.1, "eta_interior": False, "theta": [4, -.5, .28, .01], "rmse_volpts": 1.4,
                    "b_expiries": -0.27, "chi2": 5.0, "smile_loss": 0.9},
            "1": {"lambda": 1, "eta_interior": True, "theta": [3, -.6, .28, .01], "rmse_volpts": 1.8,
                  "b_expiries": -0.26, "chi2": 4.0, "smile_loss": 1.2}}
    ic = jr.interior_candidate(fits, lambdas=(0, 0.1, 1))
    assert ic["lambda"] == 1 and ic["theta"][0] == 3
    assert jr.interior_candidate({"0": fits["0"]}, lambdas=(0,)) is None
    prof = {"cells": {}}
    for i, (H, eta, b, loss) in enumerate([(0.2, 4.0, -0.38, 0.5), (0.27, 4.0, -0.26, 0.9),
                                            (0.27, 3.0, -0.25, 1.3), (0.27, 1.5, -0.27, 2.5)]):
        prof["cells"][str(i)] = {"H": H, "eta": eta, "rho": -0.5, "xi": 0.01, "rmse_volpts": 1 + loss,
                                 "b_expiries": b, "chi2": 10.0, "smile_loss": loss}
    cb = jr.contour_bests(prof, b_mkt=-0.249, se_mkt=0.033, eta_caps=(4.01, 3.5, 2.0))
    assert cb["best_smile_cell"]["H"] == 0.2 and cb["n_cells_within_1se"] == 3
    assert cb["by_eta_cap"]["4.01"]["eta"] == 4.0 and cb["by_eta_cap"]["3.5"]["eta"] == 3.0
    assert cb["by_eta_cap"]["2"]["eta"] == 1.5
    assert abs(cb["by_eta_cap"]["3.5"]["rmse_cost_volpts"] - 0.8) < 1e-12


def test_residual_bands_partition_the_quotes(capture):
    theta = jr.served_theta()
    r = jr.residual_bands(capture, theta)
    assert sum(b["n"] for b in r["bands"]) == len(capture.cal.quotes)
    assert sum(v["n"] for v in r["per_expiry"].values()) == len(capture.cal.quotes)
    assert abs(r["rmse_volpts"] - capture.cal.rmse_volpts(theta)) < 1e-9
    assert r["bands"][0]["z_lo"] == 0.0 and r["bands"][-1]["z_hi"] is None
