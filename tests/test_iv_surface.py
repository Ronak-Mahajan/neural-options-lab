"""Fast checks for backend/quant/iv_surface.py (whole file < 60 s).

  (a) the Durrleman g(k) implementation, driven by autograd, is >= 0 everywhere on
      known arbitrage-free SVI slices and < 0 on the Gatheral-Jacquier (2014)
      "Axel Vogt" slice that is the textbook butterfly-arbitrage example;
  (b) the vectorised torch implied-vol inversion agrees with calibrate.implied_vol
      and flags prices outside the no-arbitrage bounds;
  (c) the differentiable (bisection + 2 Newton) inversion of the served surrogate
      returns implicit first AND second derivatives that match finite differences,
      and the float64 copies reproduce the served float32 price path;
  (d) on a slice, the IV-space conditions (g, dw/dT) and the price-space conditions
      (d2C/dK2, d(C/S)/dT at fixed k) have the same sign point by point;
  (e) the loaded constrained surface has zero butterfly and calendar violations on a
      coarse grid, and its IV RMSE against the teacher on a small random sample is
      below the documented threshold;
  (f) the dashboard API (IVSurface.grid, arbitrage_audit) returns the documented keys
      and shapes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant import iv_surface as ivs  # noqa: E402
from backend.quant.calibrate import implied_vol  # noqa: E402
from backend.quant.engine import PricingEngine  # noqa: E402

SURFACE_PT = ivs.SURFACE_CHECKPOINT
#: documented threshold for the constrained surface's IV RMSE against the teacher
#: on the vega-resolved region.  docs/no_arbitrage_surface.md reports the measured
#: value (0.22 vol points on a 32,768-point held-out sample); this is a loose
#: ceiling above it, not the measurement.
IV_RMSE_THRESHOLD_VOLPTS = 0.50


@pytest.fixture(scope="module")
def teacher() -> ivs.TeacherSurface:
    return ivs.TeacherSurface(PricingEngine())


@pytest.fixture(scope="module")
def student() -> ivs.IVSurface:
    if not SURFACE_PT.exists():
        pytest.skip(f"{SURFACE_PT} not present; run python -m scripts.no_arbitrage_surface")
    return ivs.IVSurface.load(SURFACE_PT)


def _g_of_svi(params, k_min=-1.5, k_max=1.5, n=3001) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.linspace(k_min, k_max, n, dtype=torch.float64)
    one = torch.ones_like(k)
    d = ivs.surface_derivatives(lambda kk, T, s, r: ivs.svi_total_variance(kk, *params),
                                k, one, one, torch.zeros_like(k))
    return k, d["g"]


# --------------------------------------------------------------------------- (a)
def test_durrleman_g_positive_on_arbitrage_free_svi():
    # raw SVI (a, b, rho, m, sigma); both satisfy the Gatheral-Jacquier sufficient
    # conditions comfortably and g stays well above zero over |k| <= 1.5
    for params in ((0.04, 0.10, -0.40, 0.00, 0.10), (0.02, 0.30, -0.50, 0.05, 0.30)):
        k, g = _g_of_svi(params)
        assert torch.isfinite(g).all()
        assert float(g.min()) > 0.1, (params, float(g.min()), float(k[g.argmin()]))


def test_durrleman_g_negative_on_vogt_svi():
    # Axel Vogt's parameters, Gatheral & Jacquier (2014) Section 3: butterfly
    # arbitrage in the right wing although w(k) > 0 everywhere
    k, g = _g_of_svi((-0.0410, 0.1331, 0.3060, 0.3586, 0.4153))
    i = int(g.argmin())
    assert float(g.min()) < -0.01, float(g.min())
    assert 0.6 < float(k[i]) < 1.1, float(k[i])
    # and a crude construction that must fail badly near the money
    _, g2 = _g_of_svi((0.02, 0.5, -0.9, 0.0, 0.02))
    assert float(g2.min()) < -1.0


def test_durrleman_flat_total_variance_gives_g_one():
    k = torch.linspace(-0.2, 0.2, 41, dtype=torch.float64)
    T = torch.full_like(k, 5 / 252)
    d = ivs.surface_derivatives(lambda kk, TT, s, r: s * s * TT, k, T,
                                torch.full_like(k, 0.2), torch.zeros_like(k))
    assert torch.allclose(d["g"], torch.ones_like(k), atol=1e-10)
    assert torch.allclose(d["w_T"], torch.full_like(k, 0.04), atol=1e-10)


# --------------------------------------------------------------------------- (b)
def test_torch_inversion_matches_scalar_bisection_and_flags_bounds():
    m = torch.tensor([0.95, 1.0, 1.05, 0.90, 1.10], dtype=torch.float64)
    T = torch.full((5,), 5 / 252, dtype=torch.float64)
    r = torch.full((5,), 0.04, dtype=torch.float64)
    sig = torch.tensor([0.20, 0.30, 0.15, 0.50, 0.25], dtype=torch.float64)
    p = ivs.bs_call_unit(m, T, sig, r)
    s, defined, capped = ivs.implied_vol_torch(p, m, T, r)
    assert bool(defined.all()) and not bool(capped.any())
    assert float((s - sig).abs().max()) < 1e-10
    for i in range(5):
        ref = implied_vol(float(p[i]), float(m[i]), 1.0, float(T[i]), float(r[i]))
        assert abs(ref - float(s[i])) < 1e-9
    # below intrinsic and above spot: no implied vol
    lower, upper = ivs.price_bounds(m, T, r)
    bad = torch.stack([lower[0] * 0.5 + 0.0, upper[1] * 1.01])
    s2, def2, _ = ivs.implied_vol_torch(bad, m[:2], T[:2], r[:2])
    assert not bool(def2.any()) and bool(torch.isnan(s2).all())


# --------------------------------------------------------------------------- (c)
def test_teacher_implicit_derivatives_match_finite_differences(teacher):
    k0 = torch.tensor([-0.02, 0.0, 0.03], dtype=torch.float64)
    T0 = torch.full((3,), 5 / 252, dtype=torch.float64)
    s0 = torch.full((3,), 0.2, dtype=torch.float64)
    r0 = torch.full((3,), 0.04, dtype=torch.float64)
    d = ivs.surface_derivatives(teacher.total_variance, k0, T0, s0, r0)

    def w(k, T):
        with torch.no_grad():
            return teacher.total_variance(k, T, s0, r0)
    h, hT = 1e-4, 1e-5
    fd_k = (w(k0 + h, T0) - w(k0 - h, T0)) / (2 * h)
    fd_kk = (w(k0 + h, T0) - 2 * w(k0, T0) + w(k0 - h, T0)) / h ** 2
    fd_T = (w(k0, T0 + hT) - w(k0, T0 - hT)) / (2 * hT)
    assert torch.allclose(d["w_k"], fd_k, rtol=1e-4, atol=1e-8)
    assert torch.allclose(d["w_kk"], fd_kk, rtol=1e-3, atol=1e-5)
    assert torch.allclose(d["w_T"], fd_T, rtol=1e-4, atol=1e-8)
    assert torch.isfinite(d["g"]).all()


def test_teacher_float64_copy_matches_served_float32_path(teacher):
    gen = torch.Generator().manual_seed(1)
    k, T, s, r = ivs._sample_box(2000, gen)
    m = torch.exp(-(k + r * T))
    with torch.no_grad():
        p32 = teacher.engine._call_price_torch(m.float(), T.float(), s.float(), r.float())
        p64 = teacher.price_m(m, T, s, r)
    assert float((p32.double() - p64).abs().max()) < 1e-6      # float32 resolution


# --------------------------------------------------------------------------- (d)
def test_price_space_and_iv_space_conditions_agree_on_slice(teacher):
    rep = ivs.arbitrage_audit(teacher, sigmas=(0.10,), rates=(0.04,),
                              k_axis=ivs.default_k_axis(41), T_axis=ivs.default_T_axis(1.0),
                              return_grids=True)
    ps = rep["price_space"]
    assert ps["sign_agreement_butterfly_vs_convexity"] == 1.0
    assert ps["sign_agreement_calendar_w_vs_price"] == 1.0
    # the served surrogate does violate on this slice (that is the audit's finding)
    assert rep["iv_space"]["butterfly_all_defined"]["n_violations"] > 0
    assert rep["price_space"]["convexity_d2C_dK2"]["n_violations"] > 0
    # and every point without an implied vol is a price below intrinsic
    A, P = rep["arrays"], rep["price_arrays"]
    undefined = A["defined"] < 0.5
    assert np.all(P["below_intrinsic"][undefined] > 0)
    assert rep["price_space"]["price_above_spot_bps"]["n_violations"] == 0


# --------------------------------------------------------------------------- (e)
def test_constrained_surface_zero_violations_on_coarse_grid(student):
    k_axis = ivs.default_k_axis(61)
    T_axis = ivs.default_T_axis(1.0)
    for sigma, rate in ((0.05, 0.0), (0.10, 0.04), (0.20, 0.10), (0.80, 0.05)):
        g = student.grid(sigma, rate, k_axis, T_axis)
        assert g["iv"].shape == (T_axis.size, k_axis.size)
        assert np.isfinite(g["g"]).all() and np.isfinite(g["calendar"]).all()
        assert float(g["g_min"]) >= 0.0, (sigma, rate, g["g_min"], g["g_min_at"])
        assert float(g["calendar_min"]) >= 0.0, (sigma, rate, g["calendar_min"], g["calendar_min_at"])
        assert (g["total_variance"] > 0).all()


def test_constrained_surface_iv_rmse_vs_teacher_below_threshold(teacher, student):
    gen = torch.Generator().manual_seed(7)
    k, T, s, r = ivs._sample_box(3000, gen)
    lab = ivs.teacher_labels(teacher, k, T, s, r)
    resolved = lab["defined"] & ~lab["capped"] & (lab["vega"] >= ivs.VEGA_FLOOR)
    assert int(resolved.sum()) > 500
    iv_s = student.iv(k.numpy(), T.numpy(), s.numpy(), r.numpy())
    diff = 100.0 * (iv_s - lab["iv"].numpy())
    rmse = float(np.sqrt(np.mean(diff[resolved.numpy()] ** 2)))
    assert rmse < IV_RMSE_THRESHOLD_VOLPTS, rmse
    # and the surface never leaves the price bounds (evaluated in float32, so the
    # tolerance is float32 resolution on a price of order 0.15, not 1e-12)
    m = torch.exp(-(k + r * T))
    lower, upper = ivs.price_bounds(m, T, r)
    p = student.price(k.numpy(), T.numpy(), s.numpy(), r.numpy())
    assert np.all(p >= lower.numpy() - 2e-6) and np.all(p <= upper.numpy() + 2e-6)


def test_constrained_surface_meta_and_prior_start(student):
    meta = student.meta
    for key in ("ranges", "dynamics", "penalties", "fit_metrics", "architecture"):
        assert key in meta, key
    assert meta["ranges"]["k_convention"].startswith("ln(K/F)")
    # a freshly built net is exactly the flat-vol prior (multiplier == 1)
    net = ivs.IVSurfaceNet()
    k = torch.tensor([0.0, 0.05]); T = torch.tensor([5 / 252, 10 / 252])
    s = torch.tensor([0.2, 0.3]); r = torch.tensor([0.0, 0.05])
    assert torch.allclose(net(k, T, s, r), s * s * T, atol=1e-7)


# --------------------------------------------------------------------------- (f)
def test_grid_api_shapes_and_audit_summary_keys(student):
    k_axis = np.linspace(-0.1, 0.1, 21)
    T_axis = np.array([1, 3, 7, 12]) / 252.0
    g = student.grid(0.25, 0.03, k_axis, T_axis)
    for key in ("iv", "total_variance", "g", "calendar"):
        assert g[key].shape == (4, 21), key
    for key in ("g_min", "calendar_min"):
        assert np.ndim(g[key]) == 0
    assert g["g_min_at"].shape == (2,) and g["calendar_min_at"].shape == (2,)
    assert np.allclose(g["iv"] ** 2 * T_axis[:, None], g["total_variance"])
    rep = ivs.arbitrage_audit(student, sigmas=(0.25,), rates=(0.03,),
                              k_axis=k_axis, T_axis=T_axis)
    assert rep["surface"] == "constrained_iv_surface"
    for key in ("butterfly_all_defined", "butterfly_resolved",
                "calendar_all_defined", "calendar_resolved"):
        assert rep["iv_space"][key]["n_violations"] == 0, key
        assert "worst_at" in rep["iv_space"][key]
    assert rep["coverage"]["iv_defined_fraction"] == 1.0
    assert math.isfinite(rep["elapsed_s"])
