"""The gamma reference: exact where a closed form exists, self-consistent elsewhere.

Gamma has no pathwise estimator, so the reference is a conditional-Monte-Carlo
density evaluated at the strike. These tests pin the three facts that make it
a reference rather than another approximation: it reproduces Black-Scholes to
machine precision when the contract collapses to a European; the density it
estimates has unit mass and the exact closed-form mean; and an independent,
noisier estimator built on a different identity agrees with it.
"""
import math

import numpy as np
import pytest

from backend.quant.gamma_reference import (black_scholes_gamma,
                                           gamma_conditional,
                                           gamma_finite_difference,
                                           mean_scaled_average,
                                           scaled_average_density)


@pytest.mark.parametrize("m, T, sigma, r", [
    (1.00, 1.0, 0.25, 0.04),
    (1.30, 0.5, 0.40, 0.02),
    (0.70, 2.0, 0.15, 0.08),
    (1.95, 0.1, 0.80, 0.00),
])
def test_one_fixing_is_a_european_and_matches_black_scholes_exactly(m, T, sigma, r):
    """With n = 1 the average is the terminal spot and gamma is closed form.

    The estimator's conditioning variable is then the only variable, so every
    path returns the identical exact lognormal density: no Monte Carlo error
    at all, and the answer must equal Black-Scholes gamma to the last digit.
    """
    out = gamma_conditional(m, T, sigma, r, n_steps=1, n_paths=32, seed=3)
    assert out["gamma"] == pytest.approx(
        black_scholes_gamma(m, 1.0, T, sigma, r), rel=1e-12)
    assert out["se"] == pytest.approx(0.0, abs=1e-15)


def test_the_estimated_density_has_unit_mass_and_the_exact_mean():
    """A density integrates to one; this one's mean is a known series.

    Both are identities, not approximations: E[Atilde] = (1/n) sum exp(r t_i)
    is the geometric series the engine's put-call parity uses. A density
    estimator that gets both right at n = 50 is estimating the right object.
    """
    T, sigma, r, n = 1.0, 0.25, 0.04, 50
    a = np.exp(np.linspace(math.log(1e-3), math.log(40.0), 1500))
    p = scaled_average_density(a, T, sigma, r, n, n_paths=60_000, seed=5)

    mass = np.trapezoid(p, a)
    mean = np.trapezoid(a * p, a)
    assert mass == pytest.approx(1.0, abs=2e-3)
    assert mean == pytest.approx(mean_scaled_average(T, r, n), rel=3e-3)


def test_gamma_is_the_discounted_density_at_the_strike():
    """gamma(m) == exp(-rT) p(1/m) / m^3, on the same paths."""
    m, T, sigma, r, n = 1.15, 0.75, 0.30, 0.03, 50
    g = gamma_conditional(m, T, sigma, r, n, n_paths=40_000, seed=9)
    p = scaled_average_density(np.array([1.0 / m]), T, sigma, r, n,
                               n_paths=40_000, seed=9)[0]
    assert g["gamma"] == pytest.approx(math.exp(-r * T) * p / m ** 3,
                                       rel=1e-12)


def test_an_independent_finite_difference_agrees_to_its_own_precision():
    """Differencing the pathwise delta on common paths reaches the same value.

    The finite difference is a histogram estimate of the same density and is
    biased O(h^2) and noisy, so it corroborates at the percent level and no
    better. Agreement inside that band rules out an algebra error in the
    conditional derivation; it is not the precision the reference claims.
    """
    m, T, sigma, r, n = 1.0, 1.0, 0.25, 0.04, 50
    cond = gamma_conditional(m, T, sigma, r, n, n_paths=200_000, seed=2)
    fd = gamma_finite_difference(m, T, sigma, r, n, n_paths=200_000, seed=2,
                                 rel_step=0.02)
    assert fd["gamma"] == pytest.approx(cond["gamma"], rel=0.02)
    # And the conditional estimator's own error is far tighter than that.
    assert cond["se"] / cond["gamma"] < 0.005


def test_the_estimate_is_reproducible_and_positive():
    m, T, sigma, r, n = 0.9, 1.2, 0.2, 0.05, 50
    first = gamma_conditional(m, T, sigma, r, n, n_paths=20_000, seed=11)
    again = gamma_conditional(m, T, sigma, r, n, n_paths=20_000, seed=11)
    assert first == again
    assert first["gamma"] > 0.0 and first["se"] > 0.0
    # Blocking is a memory detail: the same paths in smaller blocks give the
    # same numbers, because the random stream is consumed identically.
    blocked = gamma_conditional(m, T, sigma, r, n, n_paths=20_000, seed=11,
                                block=4_000)
    assert blocked["gamma"] == pytest.approx(first["gamma"], rel=1e-12)


def test_it_refuses_inputs_that_have_no_density():
    with pytest.raises(ValueError):
        gamma_conditional(1.0, 1.0, 0.0, 0.04, n_steps=50, n_paths=64)
    with pytest.raises(ValueError):
        gamma_conditional(1.0, 1.0, 0.25, 0.04, n_steps=0, n_paths=64)
