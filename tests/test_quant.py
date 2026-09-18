import math
import pytest
import torch
import numpy as np
from backend.quant.monte_carlo import price_asian_mc, expected_arithmetic_average
from backend.quant.engine import PricingEngine

@pytest.fixture(scope="module")
def engine():
    return PricingEngine()

def test_asian_put_call_parity():
    """Test Asian parity identity: C - P = exp(-rT) * (E[A] - K)"""
    S, K, T, sig, r = 100.0, 100.0, 1.0, 0.20, 0.05
    n_steps = 50
    
    # Monte Carlo parity
    c_mc = price_asian_mc(S, K, T, sig, r, option_type="call", n_paths=10000, seed=42)
    p_mc = price_asian_mc(S, K, T, sig, r, option_type="put", n_paths=10000, seed=42)
    
    expected_A = expected_arithmetic_average(S, r, T, n_steps)
    parity_rhs = math.exp(-r * T) * (expected_A - K)
    
    assert abs((c_mc.price - p_mc.price) - parity_rhs) < 0.1  # MC noise allowance

def test_neural_asian_parity(engine):
    """Test the neural surrogate obeys Asian parity exactly via the architectural term."""
    S, K, T, sig, r = 100.0, 100.0, 1.0, 0.20, 0.05
    c_nn = engine.price_with_greeks(S, K, T, sig, r, option_type="call")["price"]
    p_nn = engine.price_with_greeks(S, K, T, sig, r, option_type="put")["price"]
    
    expected_A = expected_arithmetic_average(S, r, T, engine.n_steps)
    parity_rhs = math.exp(-r * T) * (expected_A - K)
    
    # NN parity is architectural, should be machine precision
    assert abs((c_nn - p_nn) - parity_rhs) < 1e-5

def test_control_variate_correctness():
    """Test that geometric control variate reduces variance and doesn't bias the price."""
    S, K, T, sig, r = 100.0, 100.0, 1.0, 0.20, 0.05
    
    res_plain = price_asian_mc(S, K, T, sig, r, control_variate=False, n_paths=10000, seed=42)
    res_cv = price_asian_mc(S, K, T, sig, r, control_variate=True, n_paths=10000, seed=42)
    
    # Prices should be statistically identical
    assert abs(res_plain.price - res_cv.price) < 0.05
    
    # Control variate should significantly reduce standard error (at least 10x)
    assert res_cv.std_error < res_plain.std_error / 10.0

def test_greeks_finite_difference(engine):
    """Test that analytical autograd Greeks match finite-difference perturbations."""
    S, K, T, sig, r = 100.0, 100.0, 1.0, 0.20, 0.05
    
    base = engine.price_with_greeks(S, K, T, sig, r, option_type="call")
    
    # Delta: dP / dS
    dS = 0.01
    p_up = engine.price_with_greeks(S + dS, K, T, sig, r, option_type="call")["price"]
    p_dn = engine.price_with_greeks(S - dS, K, T, sig, r, option_type="call")["price"]
    fd_delta = (p_up - p_dn) / (2 * dS)
    
    assert abs(base["greeks"]["delta"] - fd_delta) < 1e-3
    
    # Vega: dP / dsig (scaled to 1 point)
    dsig = 0.001
    p_up = engine.price_with_greeks(S, K, T, sig + dsig, r, option_type="call")["price"]
    p_dn = engine.price_with_greeks(S, K, T, sig - dsig, r, option_type="call")["price"]
    fd_vega = ((p_up - p_dn) / (2 * dsig)) / 100.0
    
    assert abs(base["greeks"]["vega"] - fd_vega) < 1e-3

    # Theta: -dP/dT scaled to ONE TRADING DAY. Every maturity this engine
    # quotes is in trading days - the short-dated cutoff is 12/252, the Asian
    # averages on trading days, the dashboard prints T * 252 - so the day theta
    # charges for has to be the same day, 1/252. A 1/365 calendar theta would
    # be 1.45x too small against the maturity beside it on screen.
    dT = 1e-3
    p_up = engine.price_with_greeks(S, K, T + dT, sig, r, option_type="call")["price"]
    p_dn = engine.price_with_greeks(S, K, T - dT, sig, r, option_type="call")["price"]
    fd_theta = -((p_up - p_dn) / (2 * dT)) / 252.0

    assert abs(base["greeks"]["theta"] - fd_theta) < 1e-3
    # ... and not the 365-day convention, which differs by 45% at every point.
    assert abs(base["greeks"]["theta"] - fd_theta * 252.0 / 365.0) > 1e-4


@pytest.mark.parametrize("S,K,T,sig", [
    (100.0, 100.0, 1.0, 0.20),
    (110.0, 100.0, 1.0, 0.20),
    (90.0, 100.0, 0.5, 0.35),
])
def test_rho_at_zero_rate(engine, S, K, T, sig):
    """Rho survives r = 0, which the API accepts, on both sides of parity.

    The parity term exp(-rT) * (E[A] - K) is where a put's rate sensitivity
    comes from, and at r = 0 it collapses to S - K, because E[A] is S there.
    So the PRICES come out right however the term is evaluated - C - P is
    S - K and nothing about r survives. The derivative is what exposes the
    evaluation:

        d/dr [ exp(-rT) (S * G(r) - K) ] at r = 0
            = -T (S - K) + S T (n + 1) / (2n),   G(r) = (1/n) sum_i e^{r t_i}

    which is 0.51 per rate point at S = K = 100, T = 1, n = 50, and is the
    whole of the gap between the two rhos. Evaluating E[A] through anything
    that holds r away from zero carries no derivative below its threshold,
    deletes exactly that term, and hands the put the call's rho - the call's
    sign, not the put's, and off by half a point per point of rate.
    """
    n = engine.n_steps
    call = engine.price_with_greeks(S, K, T, sig, 0.0, option_type="call")
    put = engine.price_with_greeks(S, K, T, sig, 0.0, option_type="put")
    rho_c, rho_p = call["greeks"]["rho"], put["greeks"]["rho"]

    # Parity at r = 0 reduces to C - P = S - K; the Greeks do not reduce.
    assert (call["price"] - put["price"]) == pytest.approx(S - K, abs=1e-4)
    gap = (-T * (S - K) + S * T * (n + 1) / (2 * n)) / 100.0
    assert abs(rho_c - rho_p) > 0.1
    assert (rho_c - rho_p) == pytest.approx(gap, abs=1e-6)

    # Both rhos match a central finite difference straddling zero. The step
    # has to clear float32 price quantisation - about 5e-7 on a price near 5,
    # i.e. 2e-4 per rate point at h = 1e-5 - without picking up curvature.
    h = 1e-3
    for opt, rho in (("call", rho_c), ("put", rho_p)):
        up = engine.price_with_greeks(S, K, T, sig, h, option_type=opt)["price"]
        dn = engine.price_with_greeks(S, K, T, sig, -h, option_type=opt)["price"]
        assert rho == pytest.approx((up - dn) / (2 * h) / 100.0, abs=1e-4)

    # Nothing kinks at the origin: rho is continuous through it from both
    # sides, at every scale a threshold would have hidden in.
    for eps in (1e-7, 1e-6, 1e-5):
        for side in (1.0, -1.0):
            for opt, rho in (("call", rho_c), ("put", rho_p)):
                near = engine.price_with_greeks(
                    S, K, T, sig, side * eps, option_type=opt)
                assert near["greeks"]["rho"] == pytest.approx(rho, abs=1e-4)
                assert near["price"] == pytest.approx(
                    call["price"] if opt == "call" else put["price"], abs=1e-3)


def test_whalley_wilmott_tuning_paths_are_out_of_sample():
    """The fitted baselines are tuned on paths nothing is scored on.

    Two of the four baselines are fitted: the Whalley-Wilmott no-trade band
    picks its risk aversion off a grid, and the Ruf-Wang linear hedge fits
    OLS coefficients. Both are deliberately generous to the baseline - the
    deep policy has to beat the best band, not a guessed one - but generous
    on its own block of paths is a fair fight, while generous on the block
    every hedger is scored on is not, and it puts a fifth of the reported
    evaluation paths in-sample for the baselines. scripts/deep_hedging_
    regimes.py draws the tuning block at seeds[0] + 5_000; the served
    endpoint draws it at the same offset, so the number on the dashboard is
    the number the published protocol reproduces.

    The simulator and the policy are stubbed out here: this is a statement
    about which seeds compare() draws, not about what the paths contain.
    """
    from backend.quant.hedging import HedgingEngine, N_STEPS

    class _SeedSpy(HedgingEngine):
        def __init__(self):            # deliberately no super(): no checkpoints
            self.seeds_drawn: list[int] = []

        def _spots(self, measure, sigma, rate, n_paths, seed):
            self.seeds_drawn.append(seed)
            rng = np.random.default_rng(seed)
            spots = np.empty((n_paths, N_STEPS + 1))
            spots[:, 0] = 1.0
            spots[:, 1:] = np.exp(np.cumsum(
                rng.normal(0.0, 0.01, size=(n_paths, N_STEPS)), axis=1))
            return spots

        def _deep_fn(self, sigma, rate, cost):
            return lambda i, tau, s, h: np.zeros_like(s)

    eval_seeds = (17, 18, 19)
    spy = _SeedSpy()
    spy.compare(0.25, 0.04, 0.01, n_paths=64, seeds=eval_seeds,
                primary="gbm", measures=("gbm",))

    probe_seed, tune_seed = spy.seeds_drawn[0], spy.seeds_drawn[1]
    assert spy.seeds_drawn[2:] == list(eval_seeds)
    assert tune_seed not in eval_seeds
    assert probe_seed not in eval_seeds
    assert tune_seed == eval_seeds[0] + 5_000   # the offset the scripts publish
    assert probe_seed == eval_seeds[0] + 9_000
