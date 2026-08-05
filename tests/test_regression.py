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
