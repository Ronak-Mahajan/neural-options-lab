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
