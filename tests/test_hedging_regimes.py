"""Tests for the rough-measure hedging experiment (scripts/deep_hedging_regimes.py).

Covers the new pieces that would silently poison every number in
docs/deep_hedging_regimes.md if they were wrong:

  - rough_bergomi_log_returns: deterministic per seed, discounted spot a
    martingale with and without per-step jumps, and the right terminal
    variance scale (eta = 0 collapses to GBM at sigma^2 = xi);
  - the Ruf-Wang linear hedge: OLS on cost-free GBM paths must return the
    Black-Scholes delta (c1 ~ 1, c0 and c2 ~ 0), because in a complete market
    the min-variance holding IS the delta;
  - the measure wiring in hedging.py: rough measures are read from disk, the
    training box pins sigma and rate at the calibrated level, and the old
    'gbm' path is bit-identical to what it was before.

The whole file runs in well under a minute on CPU.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.quant import hedging as H
from backend.quant.rough_vol import rough_bergomi_log_returns

DT = 1.0 / 252.0
N = 30
T = N * DT
SPY = dict(xi=0.012773, eta=3.9352, rho=-0.5363, H=0.2613, rate=0.03713)
JUMPS = (0.847938, -0.024123, 0.011667)


# --------------------------------------------------------------------------- #
#  rough_bergomi_log_returns
# --------------------------------------------------------------------------- #

def test_log_returns_deterministic_per_seed_and_shape():
    a = rough_bergomi_log_returns(1500, N, DT, seed=3, **SPY)
    b = rough_bergomi_log_returns(1500, N, DT, seed=3, **SPY)
    c = rough_bergomi_log_returns(1500, N, DT, seed=4, **SPY)
    assert a.shape == (1500, N) and a.dtype == torch.float32
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    # jumps=None must not consume generator state (common random numbers
    # across the jump axis, the same property rough_bergomi_mc documents).
    assert torch.equal(a, rough_bergomi_log_returns(1500, N, DT, seed=3,
                                                    jumps=None, **SPY))


def test_log_returns_chunking_is_exact():
    """Paths beyond _CHUNK_PATHS are drawn block by block; the first block
    of a large call must equal a small call with the same seed."""
    from backend.quant.rough_vol import _CHUNK_PATHS
    big = rough_bergomi_log_returns(_CHUNK_PATHS + 100, N, DT, seed=9, **SPY)
    small = rough_bergomi_log_returns(_CHUNK_PATHS, N, DT, seed=9, **SPY)
    assert torch.equal(big[:_CHUNK_PATHS], small)


@pytest.mark.parametrize("jumps", [None, JUMPS, (25.0, -0.02, 0.03)])
def test_discounted_spot_is_a_martingale(jumps):
    """E[S_T] = e^{rT} within 3 SE, with and without compensated jumps. The
    left-point variance and the per-step compensator are what make this
    hold; either error shows up here as a drift of many SE."""
    x = rough_bergomi_log_returns(150_000, N, DT, seed=11, jumps=jumps, **SPY)
    s_t = torch.exp(x.double().sum(dim=1))
    se = float(s_t.std()) / math.sqrt(s_t.numel())
    drift = float(s_t.mean()) - math.exp(SPY["rate"] * T)
    assert abs(drift) < 3.0 * se, f"drift {drift:.2e} vs 3 SE {3 * se:.2e}"


def test_jumps_are_compensated_step_by_step():
    """A hedger rebalances daily, so the compensator has to hold at EVERY
    step, not only at T: E[exp(r_i)] = e^{r dt} for each i."""
    x = rough_bergomi_log_returns(150_000, N, DT, seed=12,
                                  jumps=(25.0, -0.02, 0.03), **SPY).double()
    growth = torch.exp(x)
    se = growth.std(dim=0) / math.sqrt(x.shape[0])
    dev = (growth.mean(dim=0) - math.exp(SPY["rate"] * DT)).abs()
    assert bool((dev < 4.0 * se).all()), (dev / se).max()


def test_eta_zero_reduces_to_gbm_variance_scale():
    """With eta = 0 the variance is flat at xi, so every step is
    N((r - xi/2) dt, xi dt) exactly: the terminal-variance scale is xi*T."""
    x = rough_bergomi_log_returns(100_000, N, DT, seed=5, xi=SPY["xi"],
                                  eta=0.0, rho=SPY["rho"], H=SPY["H"],
                                  rate=SPY["rate"]).double()
    per_step_sd = float(x.std(dim=0).mean())
    assert per_step_sd == pytest.approx(math.sqrt(SPY["xi"] * DT), rel=0.01)
    term_var = float(x.sum(dim=1).var())
    assert term_var == pytest.approx(SPY["xi"] * T, rel=0.02)
    mean_step = float(x.mean())
    assert mean_step == pytest.approx((SPY["rate"] - 0.5 * SPY["xi"]) * DT,
                                      abs=3.0 * per_step_sd / math.sqrt(x.numel()))


def test_rough_terminal_variance_is_near_forward_variance():
    """E[V_t] = xi for all t, so E[Var(log S_T)] is xi*T up to the (small)
    leverage and vol-of-vol corrections: the realized vol of the calibrated
    measure must land within a few percent of sqrt(xi)."""
    x = rough_bergomi_log_returns(100_000, N, DT, seed=6, **SPY).double()
    realized = float(x.sum(dim=1).std()) / math.sqrt(T)
    assert abs(realized / math.sqrt(SPY["xi"]) - 1.0) < 0.05, realized


def test_jumps_widen_the_left_tail():
    """Per-step jumps must reach the terminal distribution. Measured with the
    calibrated SPY jump fit (lam 0.85/yr, mu -2.4%, sig 1.2%) the 1% quantile
    of the 30-day log return does NOT move (-0.1405 without, -0.1402 with:
    those jumps are too small and too rare to show against a diffusion whose
    terminal kurtosis is ~50), which is itself a result the experiment
    write-up records. The mechanism is checked with jumps large enough to
    register, the same parameters test_regression uses."""
    x0 = rough_bergomi_log_returns(60_000, N, DT, seed=8, **SPY)
    x1 = rough_bergomi_log_returns(60_000, N, DT, seed=8,
                                   jumps=(25.0, -0.02, 0.03), **SPY)
    q0 = float(torch.quantile(x0.sum(dim=1), 0.01))
    q1 = float(torch.quantile(x1.sum(dim=1), 0.01))
    assert q1 < q0 - 0.01, (q0, q1)


# --------------------------------------------------------------------------- #
#  the linear-regression hedge
# --------------------------------------------------------------------------- #

def test_linear_hedge_reproduces_bs_delta_on_gbm():
    """OLS min-variance fit on cost-free GBM paths returns the BS delta:
    c1 close to 1 and c0, c2 near 0. If the aggregated-gain algebra in
    fit_linear_hedge were wrong this would not come out."""
    engine = H.HedgingEngine(H.ARTIFACTS / "hedger_gbm.pt")
    sigma, rate = 0.20, 0.03
    spots = engine._spots("gbm", sigma, rate, 20_000, 123)
    fit = H.fit_linear_hedge(spots, sigma, rate)
    c0, c1, c2 = fit["coef"]
    assert abs(c1 - 1.0) < 0.02, fit
    assert abs(c0) < 0.02 and abs(c2) < 0.05, fit
    assert fit["r2"] > 0.95
    # ...and the holdings rule built from it IS (numerically) the delta hedge
    fn = H.linear_hedge_fn(fit["coef"], sigma, rate)
    s = np.linspace(0.9, 1.1, 21)
    h = fn(3, 20 * DT, s, np.zeros_like(s))
    d = H.bs_call_delta(s, 1.0, 20 * DT, sigma, rate)
    assert np.max(np.abs(h - d)) < 0.02


def test_linear_hedge_pl_is_linear_in_the_coefficients():
    """The fit rests on P&L being linear in c when costs are ignored, so the
    OLS objective and the book must agree: the fitted rule's cost-free P&L
    variance on its own training paths cannot be beaten by any small
    perturbation of the coefficients."""
    engine = H.HedgingEngine(H.ARTIFACTS / "hedger_gbm.pt")
    sigma, rate = 0.25, 0.02
    spots = engine._spots("gbm", sigma, rate, 4_000, 7)
    fit = H.fit_linear_hedge(spots, sigma, rate)

    def var_of(coef):
        pl, _, _ = engine._run_book(spots, H.linear_hedge_fn(coef, sigma, rate),
                                    0.0, 0.0, rate)
        return float(pl.var())

    base = var_of(fit["coef"])
    for k in range(3):
        for eps in (-0.05, 0.05):
            c = fit["coef"].copy()
            c[k] += eps
            assert var_of(c) >= base * (1 - 1e-9)


# --------------------------------------------------------------------------- #
#  measure wiring in hedging.py
# --------------------------------------------------------------------------- #

def test_rough_measure_params_are_read_from_disk_not_hardcoded():
    import json
    cal = json.loads(H.ROUGH_CALIBRATION_FILE.read_text(encoding="utf-8"))
    fit = json.loads(H.JUMP_FIT_FILE.read_text(encoding="utf-8"))
    p = H.rough_measure_params("rbergomi")
    assert (p["eta"], p["rho"], p["H"], p["xi"], p["rate"]) == (
        cal["eta"], cal["rho"], cal["H"], cal["xi"], cal["rate"])
    assert p["jumps"] is None
    q = H.rough_measure_params("rbergomi_jumps")
    theta = fit["jumps"]["theta"]
    assert (q["eta"], q["rho"], q["H"], q["xi"]) == tuple(theta[:4])
    assert q["jumps"] == tuple(theta[4:])
    assert q["rate"] == cal["rate"]
    with pytest.raises(ValueError):
        H.rough_measure_params("gbm")


def test_rough_train_box_pins_sigma_and_rate_at_calibration():
    box = H.train_box_for("rbergomi")
    p = H.rough_measure_params("rbergomi")
    assert box["sigma"] == (math.sqrt(p["xi"]),) * 2
    assert box["rate"] == (p["rate"],) * 2
    assert box["cost"] == H.TRAIN_BOX["cost"]
    assert H.train_box_for("gbm") == H.TRAIN_BOX


def test_engine_spots_under_rough_measures():
    engine = H.HedgingEngine(H.ARTIFACTS / "hedger_gbm.pt")
    for m in H.ROUGH_MEASURES:
        p = H.rough_measure_params(m)
        s = engine._spots(m, math.sqrt(p["xi"]), p["rate"], 500, 1)
        assert s.shape == (500, N + 1)
        assert np.all(s[:, 0] == 1.0) and np.all(s > 0)
        assert np.array_equal(
            s, engine._spots(m, math.sqrt(p["xi"]), p["rate"], 500, 1))
    with pytest.raises(ValueError, match="unknown measure"):
        engine._spots("heston", 0.2, 0.03, 10, 1)


def test_gbm_spots_are_bit_identical_to_the_pre_rough_code_path():
    """_spots('gbm') was rewritten through measure_log_returns; every seeded
    GBM number in docs/hedging_findings.md depends on the draw order."""
    from backend.quant.generative import gbm_log_returns
    engine = H.HedgingEngine(H.ARTIFACTS / "hedger_gbm.pt")
    gen = torch.Generator().manual_seed(17)
    old = np.exp(np.cumsum(gbm_log_returns(200, 0.3, 0.04, N, generator=gen)
                           .numpy().astype(np.float64), axis=1))
    assert np.array_equal(old, engine._spots("gbm", 0.3, 0.04, 200, 17)[:, 1:])


def test_train_runs_under_a_rough_measure(tmp_path, monkeypatch):
    """A handful of iterations: the differentiable book accepts rough paths
    and the checkpoint carries measure, box, params and train_seconds."""
    monkeypatch.setattr(H, "ARTIFACTS", tmp_path)
    meta = H.train(iters=3, batch=64, measure="rbergomi_jumps",
                   out_name="t.pt", log_every=0)
    assert (tmp_path / "t.pt").exists()
    assert meta["train_measure"] == "rbergomi_jumps"
    assert meta["train_seconds"] >= 0.0
    assert meta["measure_params"]["jumps"] is not None
    assert meta["train_box"]["sigma"][0] == meta["train_box"]["sigma"][1]


@pytest.mark.parametrize("name", ["hedger_rbergomi.pt", "hedger_rbergomi_jumps.pt"])
def test_shipped_rough_checkpoints_load_and_hedge(name):
    ckpt = H.ARTIFACTS / name
    if not ckpt.exists():
        pytest.skip(f"{name} not trained yet")
    engine = H.HedgingEngine(ckpt)
    m = engine.meta["train_measure"]
    assert m in H.ROUGH_MEASURES and "train_seconds" in engine.meta
    p = H.rough_measure_params(m)
    sigma = math.sqrt(p["xi"])
    spots = engine._spots(m, sigma, p["rate"], 300, 3)
    pl, hist, costs = engine._run_book(spots, engine._deep_fn(sigma, p["rate"],
                                                              0.001),
                                       0.02, 0.001, p["rate"])
    assert np.all(np.isfinite(pl)) and hist.shape == (300, N)
    assert np.all((hist >= 0.0) & (hist <= 1.5))
