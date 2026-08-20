"""Regression tests: one per bug actually found and fixed in this codebase.

Each test names the defect it guards against. They are written to FAIL against
the code as it was, not merely to exercise the happy path — a test that would
not have caught the bug is not a regression test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from backend.quant.engine import PricingEngine, ZERO_DTE_CUTOFF
from backend.quant.generative import (PathGenerator, gbm_log_returns,
                                      risk_neutralize, DT, N_STEPS)
from backend.quant.hedging import (MATURITY, bs_call_price, cvar,
                                   cvar_bootstrap_se)
from backend.quant.monte_carlo import price_asian_mc


# --------------------------------------------------------------------------- #
#  monte_carlo.py — the reported standard error ignored antithetic dependence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("control_variate", [False, True])
def test_reported_standard_error_matches_empirical(control_variate):
    """The estimator averages n paths built as concat(z, -z).

    The old code computed se = x.std(ddof=1)/sqrt(n), treating negatively
    correlated antithetic pairs as independent. Measured, that overstated the
    true error by ~45% with the control variate off (ratio 1.44 at 5k paths,
    1.49 at 20k) — and that inflated standard error is what produced the
    README's claimed "~30x" variance reduction, whose honest value is 23.8x.

    This compares the reported standard error against the empirical standard
    deviation of the estimator across independent seeded replications.
    """
    S = K = 100.0
    reps = [price_asian_mc(S, K, 1.0, 0.20, 0.05, n_paths=4_000, n_steps=50,
                           seed=90_000 + i, control_variate=control_variate)
            for i in range(120)]
    reported = float(np.mean([r.std_error for r in reps]))
    empirical = float(np.std([r.price for r in reps], ddof=1))
    ratio = reported / empirical
    # 120 replications leaves a few percent of noise on `empirical` itself;
    # the bug this guards against was a 45% error, far outside this band.
    assert 0.80 < ratio < 1.25, (
        f"reported SE / empirical SE = {ratio:.3f} "
        f"(reported {reported:.6f}, empirical {empirical:.6f})")


def test_control_variate_actually_reduces_variance():
    """The old tolerance (abs(plain - cv) < 0.05) was ~50x too loose to detect
    a 1 bp bias. Compare on the empirical scale instead, and assert the
    variance reduction is real."""
    S = K = 100.0
    kw = dict(n_paths=4_000, n_steps=50)
    plain = [price_asian_mc(S, K, 1.0, 0.20, 0.05, seed=70_000 + i,
                            control_variate=False, **kw) for i in range(120)]
    cv = [price_asian_mc(S, K, 1.0, 0.20, 0.05, seed=70_000 + i,
                         control_variate=True, **kw) for i in range(120)]
    sd_plain = np.std([r.price for r in plain], ddof=1)
    sd_cv = np.std([r.price for r in cv], ddof=1)
    assert sd_cv < sd_plain / 10.0, f"only {sd_plain / sd_cv:.1f}x reduction"
    # Unbiasedness: the two estimators must agree well within the plain SE.
    gap = abs(np.mean([r.price for r in plain]) - np.mean([r.price for r in cv]))
    assert gap < 4.0 * sd_plain / math.sqrt(len(plain)), f"gap {gap:.6f}"


# --------------------------------------------------------------------------- #
#  engine.py — mixed-maturity batches were routed to the wrong model
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def engine():
    return PricingEngine()


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mixed_maturity_batch_matches_scalar(engine, option_type):
    """price_batch gated 0DTE routing on (mat <= cutoff).all() while masking the
    put parity adjustment per element. A batch spanning the cutoff therefore
    priced every call with the Asian net and then applied EUROPEAN parity to
    the short-dated entries — two parity relations on one vector.

    The scalar path was always self-consistent, so batch-vs-scalar agreement is
    the sharpest available check.
    """
    if not engine.has_0dte:
        pytest.skip("no 0DTE surrogate available")
    mats = np.array([0.02, 0.03, 0.5, 1.0, 0.01, 1.5])
    n = len(mats)
    batch = engine.price_batch(np.full(n, 100.0), np.full(n, 100.0), mats,
                               np.full(n, 0.25), np.full(n, 0.04),
                               option_type=option_type)
    scalar = np.array([engine.price_with_greeks(100.0, 100.0, float(t), 0.25,
                                                0.04, option_type)["price"]
                       for t in mats])
    assert np.abs(batch - scalar).max() < 1e-4, (
        f"max |batch - scalar| = {np.abs(batch - scalar).max():.3e}")


def test_in_domain_knows_about_both_surrogates(engine):
    """in_domain checked only the Asian box, so a valid 0DTE query was reported
    out of domain, and the gap between the boxes was invisible."""
    assert engine.in_domain(100, 100, 1.0, 0.25, 0.04)
    if engine.has_0dte:
        assert engine.in_domain(100, 100, 0.02, 0.25, 0.04)
        # Below the 0DTE floor of 1/252.
        assert not engine.in_domain(100, 100, 1e-9, 0.25, 0.04)
    # Dead band: above the 0DTE cutoff, below the Asian training floor.
    assert not engine.in_domain(100, 100, 0.048, 0.25, 0.04)


# --------------------------------------------------------------------------- #
#  generative.py — risk_neutralize produced a non-martingale measure
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def generator():
    from pathlib import Path
    ck = Path(__file__).resolve().parents[1] / "artifacts" / "generator.pt"
    if not ck.exists():
        pytest.skip("generator.pt not available")
    g = PathGenerator()
    g.load_state_dict(torch.load(ck, map_location="cpu",
                                 weights_only=True)["generator"])
    g.eval()
    return g


@pytest.mark.parametrize("sigma", [0.15, 0.30, 0.65])
def test_risk_neutralize_is_a_martingale_with_correct_vol(generator, sigma):
    """Per-step standardization pinned marginals but left cross-step covariance
    free, so paths realized 1.28x the requested volatility and E[S_T] exceeded
    e^{rT} by up to 147 bps. The option was then booked ~30% too cheap."""
    rate, n = 0.03, 20_000
    g = torch.Generator().manual_seed(11)
    z = torch.randn(n, generator.noise_dim, generator=g)
    st = torch.full((n, 1), sigma)
    rt = torch.full((n, 1), rate)
    with torch.no_grad():
        lr = risk_neutralize(generator(z, st, rt), st, rt)
    cum = lr.cumsum(dim=1)
    realized = float(cum[:, -1].std()) / math.sqrt(MATURITY)
    assert abs(realized / sigma - 1.0) < 0.02, (
        f"realized vol {realized:.4f} vs requested {sigma:.4f}")
    drift_bps = (float(torch.exp(cum[:, -1]).mean())
                 - math.exp(rate * MATURITY)) * 1e4
    assert abs(drift_bps) < 1.0, f"E[S_T] - e^rT = {drift_bps:+.2f} bps"


def test_gbm_control_measure_is_risk_neutral():
    """The out-of-sample control measure must itself be sound."""
    sigma, rate, n = 0.25, 0.03, 40_000
    g = torch.Generator().manual_seed(3)
    cum = gbm_log_returns(n, sigma, rate, N_STEPS, generator=g).cumsum(dim=1)
    se = float(torch.exp(cum[:, -1]).std()) / math.sqrt(n)
    drift = float(torch.exp(cum[:, -1]).mean()) - math.exp(rate * MATURITY)
    assert abs(drift) < 4.0 * se, f"drift {drift:.6f} vs 4 SE {4*se:.6f}"
    realized = float(cum[:, -1].std()) / math.sqrt(MATURITY)
    assert abs(realized / sigma - 1.0) < 0.03


# --------------------------------------------------------------------------- #
#  hedging.py — CVaR returned NaN on small samples
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 5, 19, 20, 100])
def test_cvar_is_finite_for_small_samples(n):
    """ceil(0.95*n) == n whenever n <= 19, which sliced an empty tail and
    returned NaN rather than a number or an error."""
    pl = np.linspace(-1.0, 1.0, n)
    v = cvar(pl, 0.95)
    assert np.isfinite(v), f"CVaR is {v} for n={n}"


def test_cvar_bootstrap_se_is_positive_and_finite():
    rng = np.random.default_rng(0)
    pl = rng.standard_normal(2_000) * 0.02
    se = cvar_bootstrap_se(pl, 0.95, n_boot=200, seed=1)
    assert np.isfinite(se) and se > 0.0


def test_cvar_matches_definition():
    """CVaR_95 of a known sample equals the mean of its worst 5% of losses."""
    rng = np.random.default_rng(5)
    pl = rng.standard_normal(10_000)
    losses = np.sort(-pl)
    expected = losses[int(math.ceil(0.95 * losses.size)):].mean()
    assert cvar(pl, 0.95) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
#  Sanity: the shipped artifacts still load and price
# --------------------------------------------------------------------------- #

def test_shipped_checkpoint_prices_sanely(engine):
    out = engine.price_with_greeks(100.0, 100.0, 1.0, 0.20, 0.05, "call")
    assert 0.0 < out["price"] < 100.0
    assert 0.0 < out["greeks"]["delta"] < 1.0
    assert out["greeks"]["gamma"] > 0.0
    # A call is worth at least its discounted intrinsic on the average.
    assert out["price"] > 0.5


def test_asian_call_dominates_geometric(engine):
    """AM-GM: the arithmetic average dominates the geometric one pathwise, so
    C_arith >= C_geo. Checked against the closed form."""
    from backend.quant.monte_carlo import geometric_asian_price
    for m, t, s, r in [(0.8, 0.5, 0.2, 0.03), (1.0, 1.0, 0.3, 0.05),
                       (1.4, 1.5, 0.5, 0.02)]:
        arith = engine.price_with_greeks(100.0 * m, 100.0, t, s, r,
                                         "call")["price"]
        geo = geometric_asian_price(100.0 * m, 100.0, t, s, r,
                                    engine.n_steps, "call")
        assert arith >= geo - 0.05, f"arith {arith:.5f} < geo {geo:.5f}"


# --------------------------------------------------------------------------- #
#  rough_vol.py — the wrong fractional kernel, and the wrong leverage object
# --------------------------------------------------------------------------- #

def test_volterra_covariance_matches_quadrature():
    """The driver was built from the Type-I (Mandelbrot-Van Ness) fBm
    covariance, not the Riemann-Liouville Volterra covariance rough Bergomi
    requires. The two agree only on the diagonal, which is why it survived."""
    from scipy.integrate import quad
    from backend.quant.rough_vol import volterra_covariance
    H, n = 0.1172, 40
    dt = (1.5 / 365) / n
    C = volterra_covariance(n, dt, H, torch.device("cpu")).double().numpy()
    t = np.arange(1, n + 1) * dt

    def exact(u, v):
        f = lambda s: (u - s) ** (H - 0.5) * (v - s) ** (H - 0.5)
        return 2 * H * quad(f, 0, min(u, v), points=[min(u, v)], limit=200)[0]

    for i in (0, 7, 19, 39):
        for j in (0, 3, 25, 39):
            got, want = C[i, j], exact(t[i], t[j])
            assert abs(got - want) / abs(want) < 1e-5, (i, j, got, want)
    # Diagonal is Var[W~_t] = t^{2H} in both parameterizations.
    assert np.abs(np.diag(C) / t ** (2 * H) - 1.0).max() < 1e-5


def test_leverage_correlates_spot_with_the_driving_brownian_motion():
    """chol(C) is NOT the Volterra kernel: W~ is a continuous integral, not a
    linear function of n coarse increments, so factorising C alone forces
    corr(Z_1, W~_t1) = 1 by construction. The truth is sqrt(2H)/(H+1/2).

    Applying rho to the wrong object over-correlates spot and vol at the short
    end, which is exactly where a 0DTE skew fit is identified.
    """
    from backend.quant.rough_vol import joint_factor, volterra_covariance
    H, n = 0.1172, 50
    dt = (1.5 / 365) / n
    L = joint_factor(n, dt, H, torch.device("cpu")).double().numpy()
    S = L @ L.T
    assert np.abs(S[:n, :n] - np.eye(n)).max() < 1e-8, "dW must be i.i.d. N(0,1)"
    expected = math.sqrt(2 * H) / (H + 0.5)
    got = S[n, 0] / math.sqrt(S[n, n])
    assert abs(got - expected) < 1e-6, f"corr {got:.4f} vs {expected:.4f}"
    C = volterra_covariance(n, dt, H, torch.device("cpu")).double().numpy()
    assert np.abs(S[n:, n:] - C / dt ** (2 * H)).max() \
        / np.abs(C / dt ** (2 * H)).max() < 1e-6


def test_rough_bergomi_discounted_spot_is_a_martingale():
    """A pricing measure must satisfy E[S_T] = S_0 e^{rT}; priced with a strike
    of ~0 the call collapses to the discounted expected spot."""
    from backend.quant.rough_vol import rough_bergomi_mc
    S0 = torch.tensor([100.0])
    out = rough_bergomi_mc(S0, torch.tensor([1e-9]), torch.tensor([1.5 / 365]),
                           torch.tensor([0.18 ** 2]), torch.tensor([2.1384]),
                           torch.tensor([-0.7387]), torch.tensor([0.0]),
                           n_paths=200_000, n_steps=50, H=0.1172, seed=5)
    assert abs(float(out[0]) / 100.0 - 1.0) < 1e-4


def test_old_fbm_covariance_fails_loudly():
    """Anything still calling the removed Type-I builder must stop, not
    silently simulate the wrong process."""
    from backend.quant.rough_vol import generate_fbm_covariance
    with pytest.raises(NotImplementedError, match="not the rough Bergomi"):
        generate_fbm_covariance(10, 0.001, 0.1, torch.device("cpu"))


# --------------------------------------------------------------------------- #
#  calibrate.py / dataset_0dte.py — a bad calibration must never be adopted
# --------------------------------------------------------------------------- #

def _ok_fit(**over):
    base = dict(rmse=1.0, eta=2.0, rho=-0.7, H=0.12, xi=0.03,
                stale=False, n_unpriceable=0)
    base.update(over)
    return base


def test_quality_gate_accepts_a_good_fit():
    from backend.quant.calibrate import quality_gate
    accepted, reasons = quality_gate(**_ok_fit())
    assert accepted, reasons


@pytest.mark.parametrize("bad,needle", [
    ({"rmse": 48.7}, "RMSE"),
    ({"stale": True}, "live book"),
    ({"n_unpriceable": 60, "n_quotes": 385}, "no-arb"),
    ({"eta": 4.0}, "pinned"),
    ({"xi": 1e-9}, "pinned"),
])
def test_quality_gate_rejects(bad, needle):
    """Each clause the shipped calibration slipped past.

    The committed rough_calibration.json was stamped 03:43 New York with
    quote_source 'last_trade_market_closed' and accepted=true, because the gate
    tested only RMSE and bound-pinning — never staleness, never xi, never
    unpriceable quotes. Those parameters were then trained into the served model.
    """
    from backend.quant.calibrate import quality_gate
    accepted, reasons = quality_gate(**_ok_fit(**bad))
    assert not accepted
    assert any(needle.lower() in r.lower() for r in reasons), reasons


def test_calibration_is_refused_when_it_predates_the_current_kernel(tmp_path,
                                                                    monkeypatch):
    """A fit is only valid for the driver it was fitted under.

    The shipped calibration was fitted against the Type-I fBm covariance — a
    process that is not rough Bergomi — so its (eta, rho, H) carry no
    information now. Such files have no `kernel` field.
    """
    import json
    from backend.quant import dataset_0dte as d

    monkeypatch.setattr(d, "ARTIFACTS", tmp_path)
    cal = tmp_path / "rough_calibration.json"
    defaults = {"eta": 1.5, "rho": -0.7, "H": 0.1}

    # Legacy: accepted, but no kernel field.
    cal.write_text(json.dumps({"accepted": True, "eta": 2.1384,
                               "rho": -0.7387, "H": 0.1172,
                               "quote_source": "live"}))
    assert d.load_calibrated_dynamics() == defaults

    # Right kernel but market-closed quotes.
    cal.write_text(json.dumps({"accepted": True, "eta": 2.1384,
                               "rho": -0.7387, "H": 0.1172,
                               "kernel": d.KERNEL_ID,
                               "quote_source": "last_trade_market_closed"}))
    assert d.load_calibrated_dynamics() == defaults

    # Right kernel, live quotes, accepted -> adopted.
    cal.write_text(json.dumps({"accepted": True, "eta": 2.0, "rho": -0.6,
                               "H": 0.13, "kernel": d.KERNEL_ID,
                               "quote_source": "live"}))
    assert d.load_calibrated_dynamics() == {"eta": 2.0, "rho": -0.6, "H": 0.13}


def test_calibrate_stamps_the_kernel_it_was_fitted_under():
    """Without this, a fresh, valid calibration would be refused forever."""
    from backend.quant.calibrate import KERNEL_ID as cal_kernel
    from backend.quant.dataset_0dte import KERNEL_ID as ds_kernel
    assert cal_kernel == ds_kernel


# --------------------------------------------------------------------------- #
#  engine.py — the served checkpoint carries a fixed output scale
# --------------------------------------------------------------------------- #

def test_output_scale_is_loaded_and_legacy_checkpoints_default_to_one(engine):
    """The served model conditions its Softplus head to emit a quantity of
    order 1 and carries the magnitude in a fixed scale, which is what halved the
    systematic price bias. Legacy checkpoints fold the magnitude into the
    network and must keep working unchanged."""
    from pathlib import Path
    from backend.quant.engine import ARTIFACTS
    assert engine._output_scale > 0.0
    legacy = ARTIFACTS / "model_legacy_unconditioned_head.pt"
    if legacy.exists():
        assert PricingEngine(legacy)._output_scale == 1.0


def test_greeks_flow_through_the_output_scale(engine):
    """A constant output factor must pass cleanly into every derivative — if it
    were applied outside the autograd graph the price would move and the Greeks
    would not, which is the kind of error that only shows up when someone hedges
    with it."""
    S = K = 100.0
    out = engine.price_with_greeks(S, K, 1.0, 0.20, 0.05, "call")
    h = 0.01
    up = engine.price_with_greeks(S + h, K, 1.0, 0.20, 0.05, "call")["price"]
    dn = engine.price_with_greeks(S - h, K, 1.0, 0.20, 0.05, "call")["price"]
    assert out["greeks"]["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-3)

    hv = 1e-4
    vup = engine.price_with_greeks(S, K, 1.0, 0.20 + hv, 0.05, "call")["price"]
    vdn = engine.price_with_greeks(S, K, 1.0, 0.20 - hv, 0.05, "call")["price"]
    # vega is reported per vol POINT, the difference quotient is per unit vol
    assert out["greeks"]["vega"] * 100.0 == pytest.approx(
        (vup - vdn) / (2 * hv), rel=1e-2)


def test_served_model_beats_its_predecessor_on_price(engine):
    """Guards the promotion itself: the served checkpoint must actually be the
    conditioned one, not silently reverted."""
    meta = engine.meta
    if "promoted_from" not in meta:
        pytest.skip("served checkpoint predates the promotion gate")
    assert meta["output_scale"] != 1.0
    assert meta.get("residual") is False


# --------------------------------------------------------------------------- #
#  calibrate.py — the quality gate had two blind spots a live run exposed
# --------------------------------------------------------------------------- #

def test_pin_tolerance_is_wide_enough_to_actually_fire():
    """PIN_FRAC was 1e-3 — 0.1% of a parameter's range — so tight it could
    essentially never trigger. The first live Deribit BTC calibration returned
    eta = 3.936 against an upper bound of 4.0, i.e. 98.2% of the way across the
    range, and the gate ACCEPTED it. A bound hit is the signal that the model
    cannot reach the market without an extreme parameter; that is the whole
    point of the check.
    """
    from backend.quant.calibrate import BOUNDS, PIN_FRAC, quality_gate
    lo, hi = BOUNDS["eta"]
    near = hi - 0.001 * (hi - lo)          # 99.9% across the range
    accepted, reasons = quality_gate(rmse=1.0, eta=near, rho=-0.7, H=0.12,
                                     xi=0.03, stale=False, n_unpriceable=0)
    assert not accepted
    assert any("pinned" in r for r in reasons), reasons
    # The historically-observed value must also be caught.
    accepted, _ = quality_gate(rmse=1.0, eta=3.936, rho=-0.7, H=0.12, xi=0.03,
                               stale=False, n_unpriceable=0)
    assert not accepted, "eta=3.936 of a [0.5, 4.0] range must read as pinned"
    assert PIN_FRAC >= 0.01


def test_deribit_gate_is_the_shared_gate_not_a_divergent_copy():
    """calibrate_deribit.py used to carry its OWN symmetric market-width rule
    while calibrate.py's gate had been corrected to one-directional (the
    ceiling is max(3 vp, 1.5x median half-spread); a tight book cannot lower
    it). Two gates with different semantics for the same question is how the
    same bug gets fixed once and shipped twice.

    Semantic change, stated rather than hidden: the historic BTC fit
    (2.812 vp RMSE against a ~1 vp median half-spread) was rejected by the old
    local rule as "2.9x outside the spread" and is ACCEPTED by the unified
    gate, because 2.812 is under the 3.0 vp absolute floor. That is the
    deliberate trade adopted on live SPY data: the bid-ask measures the
    market's execution resolution, not the resolution at which four
    parameters can describe a whole surface, so a narrow book must not
    tighten the ceiling below what a low-dimensional model can ever meet.
    """
    from backend.quant.calibrate import quality_gate
    base = dict(eta=2.0, rho=-0.5, H=0.12, xi=0.03, stale=False,
                n_unpriceable=0, n_quotes=300)
    ok, why = quality_gate(rmse=2.812, median_half_spread_iv=0.976, **base)
    assert ok, why
    # The absolute floor still bites on the same book...
    bad, why = quality_gate(rmse=3.2, median_half_spread_iv=0.976, **base)
    assert not bad
    # ...and a genuinely wide crypto book raises the ceiling above the floor.
    wide, why = quality_gate(rmse=4.5, median_half_spread_iv=4.0, **base)
    assert wide, why


def test_deribit_quotes_convert_into_calibrator_quotes():
    """The BTC path must reuse calibrate.py's objective, not reimplement it."""
    from backend.quant.deribit import find_snapshots, latest_snapshot
    from backend.quant.surface import build_surface
    from backend.quant.calibrate import Calibrator
    from backend.quant.calibrate_deribit import quotes_from_surface
    if not find_snapshots():
        pytest.skip("no committed Deribit snapshot")
    quotes, drops = quotes_from_surface(build_surface(latest_snapshot()))
    assert len(quotes) >= 8
    assert len({q.expiry for q in quotes}) >= 2, (
        "H is identified by the term structure of the skew; a single-expiry "
        "fit cannot identify it")
    assert math.isfinite(drops["median_iv_spread_volpts"])
    assert drops["median_iv_spread_volpts"] > 0
    for q in quotes:
        assert q.tau > 0 and q.strike > 0 and q.mid_call > 0
        assert 0.05 <= q.iv <= 0.80
        assert q.vega > 0
    cal = Calibrator(rate=0.0, quotes=quotes, n_paths=200)
    assert len(cal.quotes) == len(quotes)
    assert len(cal.groups) >= 2


# --------------------------------------------------------------------------- #
#  rough_vol.py / calibrate.py — one simulation per smile, and on the GPU
# --------------------------------------------------------------------------- #

def test_strikes_on_one_smile_share_a_single_path_set():
    """Strike does not affect the terminal distribution, so every contract
    sharing (spot, T, xi, eta, rho, rate) must be priced against ONE sample.

    Drawing an independent sample per strike injected noise into the smile
    SHAPE, and shape is what identifies rho, eta and H. Sharing the sample
    makes the call price monotone in the strike by construction.
    """
    from backend.quant.rough_vol import rough_bergomi_mc
    n = 30
    K = torch.linspace(730.0, 800.0, n)
    kw = dict(n_paths=20_000, n_steps=50, H=0.16, seed=5)
    p = rough_bergomi_mc(torch.full((n,), 771.0), K, torch.full((n,), 5 / 365),
                         torch.full((n,), 0.11 ** 2), torch.full((n,), 3.4),
                         torch.full((n,), -0.31), torch.full((n,), 0.037), **kw)
    diffs = (p[1:] - p[:-1]).numpy()
    assert (diffs <= 1e-6).all(), "call price must not rise with strike"


def test_grouping_does_not_change_prices():
    """Same seed, same dynamics: pricing a smile in one call must agree with
    pricing each strike separately."""
    from backend.quant.rough_vol import rough_bergomi_mc
    n = 12
    K = torch.linspace(740.0, 790.0, n)
    kw = dict(n_paths=20_000, n_steps=50, H=0.16, seed=11)
    def args(m):
        return (torch.full((m,), 771.0), torch.full((m,), 5 / 365),
                torch.full((m,), 0.11 ** 2), torch.full((m,), 3.4),
                torch.full((m,), -0.31), torch.full((m,), 0.037))
    s, T, xi, eta, rho, r = args(n)
    grouped = rough_bergomi_mc(s, K, T, xi, eta, rho, r, **kw)
    for i in range(n):
        s1, T1, xi1, eta1, rho1, r1 = args(1)
        one = rough_bergomi_mc(s1, K[i:i + 1], T1, xi1, eta1, rho1, r1, **kw)
        assert abs(float(one[0]) - float(grouped[i])) < 1e-4


def test_distinct_dynamics_are_not_merged():
    """Grouping must key on the dynamics, not just the strike: different
    maturities are different terminal distributions."""
    from backend.quant.rough_vol import rough_bergomi_mc
    p = rough_bergomi_mc(
        torch.full((2,), 771.0), torch.tensor([771.0, 771.0]),
        torch.tensor([5 / 365, 20 / 365]), torch.full((2,), 0.11 ** 2),
        torch.full((2,), 3.4), torch.full((2,), -0.31),
        torch.full((2,), 0.037), n_paths=20_000, n_steps=50, H=0.16, seed=3)
    assert float(p[1]) > float(p[0]), "longer maturity must be worth more"


def test_calibration_runs_on_the_gpu_when_one_is_available():
    """Every tensor in model_prices used to be built without a device, so a fit
    never left the CPU: 15.82 s per objective evaluation against roughly 0.09 s
    of equivalent GPU work, and 2,788 s for one live SPY calibration."""
    from backend.quant.calibrate import Calibrator
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Calibrator.device.type == expected


def test_max_dte_default_admits_a_term_structure():
    """H is identified by the term structure of the skew. The old default of 3
    calendar days retained a SINGLE expiry after the tau floor, so the shipped
    default could not identify one of the four parameters it fits — a live SPY
    run said so in its own output and then reported an H anyway.

    Reads the real default out of the module source rather than restating it.
    """
    import inspect
    import re
    from backend.quant import calibrate

    src = inspect.getsource(calibrate.main)
    m = re.search(r'"--max-dte",\s*type=int,\s*default=(\d+)', src)
    assert m, "could not find the --max-dte default"
    default = int(m.group(1))

    # Weekly expiries, so anything under ~10 days risks a single surviving
    # expiry once the 34.8h tau floor is applied.
    assert default >= 14, f"--max-dte default {default} is too narrow for H"
    # And it must stay inside the surrogate's trained ceiling of 12/252 yr,
    # which is 12/252*365 = 17.38 calendar days.
    assert default <= 12 * 365 / 252, (
        f"--max-dte default {default} admits maturities the surrogate was "
        f"never trained on")


# --------------------------------------------------------------------------- #
#  calibrate.py — xi is a curve, and the gate is not a coin flip
# --------------------------------------------------------------------------- #

def test_a_few_unpriceable_quotes_are_noise_not_a_failure():
    """The gate used to fail on a single unpriceable quote. Holding the
    parameters AND the quotes fixed and varying only the Monte Carlo path set
    gave 1, 4 and 3 unpriceable at 64,000 paths and 1, 0 and 0 at 200,000 — so
    a calibration passed or failed on the random draw, not on fit quality."""
    from backend.quant.calibrate import quality_gate
    ok = dict(rmse=1.0, eta=2.0, rho=-0.5, H=0.12, xi=0.03, stale=False,
              n_quotes=385, median_half_spread_iv=1.0)
    accepted, reasons = quality_gate(n_unpriceable=4, **ok)
    assert accepted, reasons
    accepted, reasons = quality_gate(n_unpriceable=60, **ok)
    assert not accepted
    assert any("no-arb" in r for r in reasons), reasons


def test_market_width_can_only_raise_the_rmse_ceiling_never_lower_it():
    """A wide book earns a looser ceiling. A TIGHT book must not earn a
    stricter one.

    The first version of this criterion was symmetric — RMSE had to sit within
    1.5x the median half-spread either way — and a live SPY run showed why that
    is wrong. Across 677 two-sided quotes in 8 expiries the median half-spread
    was 0.052 vol points, i.e. a 3-cent market on a $3.96 option, which is
    simply how SPY quotes. The symmetric rule demanded a 4-parameter rough
    Bergomi surface price every strike and expiry to within 0.078 vp and threw
    out a 1.579 vp fit for being "32x the spread". The bid-ask measures the
    market's EXECUTION resolution, not the resolution at which four parameters
    can describe a surface.
    """
    from backend.quant.calibrate import quality_gate, MAX_RMSE_VOLPTS
    base = dict(eta=2.0, rho=-0.5, H=0.12, xi=0.03, stale=False,
                n_unpriceable=0, n_quotes=385)

    # Tight book (SPY): the absolute floor governs, the spread does not tighten.
    ok, why = quality_gate(rmse=1.579, median_half_spread_iv=0.052, **base)
    assert ok, why
    # ...and the floor still bites on a genuinely bad fit in the same book.
    bad, why = quality_gate(rmse=MAX_RMSE_VOLPTS + 0.5,
                            median_half_spread_iv=0.052, **base)
    assert not bad and any("RMSE" in r for r in why), why

    # Wide book (crypto / single names): the ceiling rises above the floor.
    wide, why = quality_gate(rmse=MAX_RMSE_VOLPTS + 1.0,
                             median_half_spread_iv=4.0, **base)
    assert wide, why
    # But not without limit — 1.5 x 4.0 = 6.0 vp is still a ceiling.
    too_wide, why = quality_gate(rmse=7.0, median_half_spread_iv=4.0, **base)
    assert not too_wide and any("half-spread" in r for r in why), why


def test_xi_accepts_a_curve_and_pins_each_expiry_to_its_own_atm():
    """xi_0(t) is a forward variance CURVE. Flattening it to a scalar forces one
    ATM vol onto every maturity: measured live, ATM ran 9.18% to 11.75% across
    six SPY expiries, a 2.57 vol point term structure a single number cannot
    represent, and H then absorbed the residual (0.072 to 0.350 across expiries).
    """
    from backend.quant.calibrate import Calibrator, Quote
    quotes = []
    for expiry, tau, atm_iv in (("A", 2 / 365, 0.09), ("B", 10 / 365, 0.12)):
        for k in (0.97, 1.00, 1.03):
            quotes.append(Quote(tau=tau, strike=770.0 * k, mid_call=5.0,
                                iv=atm_iv, vega=60.0, kind="C", expiry=expiry,
                                fwd_pv=770.0, half_spread_iv=0.5))
    cal = Calibrator(rate=0.0, quotes=quotes, n_paths=4_000)
    assert len(cal.groups) == 2

    # A scalar must still work (back-compatible), and a per-group vector must
    # produce DIFFERENT prices from a scalar that cannot match both expiries.
    flat = cal.model_prices(2.0, -0.4, 0.12, 0.01)
    curve = cal.model_prices(2.0, -0.4, 0.12, np.array([0.008, 0.016]))
    assert flat.shape == curve.shape == (len(quotes),)
    assert not np.allclose(flat, curve)

    with pytest.raises(ValueError, match="expiry groups"):
        cal.model_prices(2.0, -0.4, 0.12, np.array([0.01, 0.01, 0.01]))


def test_solve_xi_recovers_a_rising_forward_variance_curve():
    """Given a rising ATM term structure, the profiled xi must rise with it."""
    from backend.quant.calibrate import Calibrator, Quote, bs_call
    F, rate = 770.0, 0.0
    quotes = []
    for expiry, tau, iv in (("A", 3 / 365, 0.09), ("B", 12 / 365, 0.13)):
        for k in (0.98, 1.00, 1.02):
            K = F * k
            quotes.append(Quote(tau=tau, strike=K,
                                mid_call=bs_call(F, K, tau, iv, rate), iv=iv,
                                vega=60.0, kind="C", expiry=expiry, fwd_pv=F,
                                half_spread_iv=0.5))
    cal = Calibrator(rate=rate, quotes=quotes, n_paths=20_000)
    xi = cal.solve_xi(1.5, -0.3, 0.12, n_paths=20_000, seed=3, iters=14)
    assert xi.shape == (2,)
    assert xi[1] > xi[0], f"forward variance must rise: {xi}"
    assert 0.05 ** 2 < xi[0] < 1.2 ** 2


def test_load_calibrated_dynamics_is_market_aware(tmp_path, monkeypatch):
    """A passing BTC fit could never be adopted: calibrate_deribit.py writes
    rough_calibration_btc.json, but the loader read only the SPY file. The
    plumbing existed end to end EXCEPT the last join."""
    import json
    from backend.quant import dataset_0dte as d
    monkeypatch.setattr(d, "ARTIFACTS", tmp_path)
    btc = dict(accepted=True, eta=2.4, rho=-0.5, H=0.09, kernel=d.KERNEL_ID,
               quote_source="live_two_sided_book", market="BTC-DERIBIT")
    (tmp_path / "rough_calibration_btc.json").write_text(json.dumps(btc))

    # BTC is adopted from its own file...
    assert d.load_calibrated_dynamics("BTC") == {"eta": 2.4, "rho": -0.5,
                                                 "H": 0.09}
    # ...the SPY default is NOT cross-contaminated by it...
    assert d.load_calibrated_dynamics() == {"eta": 1.5, "rho": -0.7, "H": 0.1}
    # ...and a typo'd market fails loudly instead of silently defaulting.
    with pytest.raises(ValueError, match="unknown market"):
        d.load_calibrated_dynamics("DOGE")


# --------------------------------------------------------------------------- #
#  rough_vol.py — compensated Merton jumps (the left-tail hypothesis)
# --------------------------------------------------------------------------- #

def test_jumps_off_is_bit_identical():
    """jumps=None must not consume generator state: any extra draw would
    silently change every seeded result in the project."""
    from backend.quant.rough_vol import rough_bergomi_mc
    t = lambda v: torch.tensor([float(v)])
    args = (t(770.0), t(770.0), t(4 / 365), t(0.011), t(2.69), t(-0.33),
            t(0.037))
    a = rough_bergomi_mc(*args, n_paths=5_000, seed=7)
    b = rough_bergomi_mc(*args, n_paths=5_000, seed=7, jumps=None)
    assert torch.equal(a, b)


def test_jumps_preserve_the_martingale_exactly():
    """The compensator -lam*(e^{mu+sig^2/2}-1)*T must hold E[S_T] fixed. A
    call struck at 0 IS disc*E[S_T] = spot, so it checks the drift directly."""
    from backend.quant.rough_vol import rough_bergomi_mc
    t = lambda v: torch.tensor([float(v)])
    args = (t(770.0), t(0.0), t(4 / 365), t(0.011), t(2.69), t(-0.33),
            t(0.037))
    c0, se = rough_bergomi_mc(*args, n_paths=200_000, seed=11,
                              jumps=(25.0, -0.02, 0.03),
                              return_std_error=True)
    assert abs(float(c0) - 770.0) < 4.0 * float(se), (
        f"disc*E[S_T] = {float(c0):.4f} vs 770, SE {float(se):.4f}")


def test_jumps_fatten_the_left_tail():
    """The reason they exist: at 4 days the diffusive model prices a K/F=0.94
    put at ~half a cent while the market carries real premium there (the
    measured -2.1 vp wing bias). Jumps must move that put materially."""
    from backend.quant.rough_vol import rough_bergomi_mc
    t = lambda v: torch.tensor([float(v)])
    K = 0.94 * 770
    args = (t(770.0), t(K), t(4 / 365), t(0.011), t(2.69), t(-0.33), t(0.037))
    disc = math.exp(-0.037 * 4 / 365)
    put = lambda c: float(c) - (770 - K * disc)
    p_nj = put(rough_bergomi_mc(*args, n_paths=100_000, seed=11))
    p_j = put(rough_bergomi_mc(*args, n_paths=100_000, seed=11,
                               jumps=(25.0, -0.02, 0.03)))
    assert p_j > 3.0 * max(p_nj, 1e-4), f"{p_nj:.5f} -> {p_j:.5f}"
