"""Checks on the quant layer's loading and pricing contracts.

  * Every committed checkpoint loads with torch's safe loader
    (weights_only=True), each served loader builds from its artifact, and the
    torch.save helpers in hedging.py and iv_surface.py write checkpoints the
    safe loader reads back. Non-tensor meta that a future retrain adds
    therefore fails here, not at deploy.
  * IVSurface.load fails closed: a checkpoint carrying a pickled callable is
    refused and the callable never runs.
  * PricingEngine.price_with_greeks reports the floor at zero. At a 0DTE
    point where the call is priced below its European parity floor, the
    parity-derived put is negative: `price` is 0.0, `raw_price` is the
    negative value, and `clamped` is True. The expected raw put is rebuilt
    from the call and the closed-form parity term, not taken from the put
    path under test.
  * price_with_greeks accepts numpy scalars and returns the same numbers as
    for Python floats.
  * price_batch(floor_at_zero=False) returns the unfloored values, and the
    default output is their elementwise maximum with zero.
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.quant import hedging  # noqa: E402
from backend.quant import iv_surface as ivs  # noqa: E402
from backend.quant.engine import PricingEngine, ZERO_DTE_CUTOFF  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CHECKPOINTS = sorted(ARTIFACTS.glob("*.pt"))

# A 0DTE point where the ensemble prices the call below S - K e^(-rT), so the
# parity-derived put is negative before the floor.
CLAMPED = dict(spot=101.6557, strike=100.0, maturity=0.04125, sigma=0.05056,
               rate=0.03038)


@pytest.fixture(scope="module")
def engine() -> PricingEngine:
    return PricingEngine()


# --------------------------------------------------------------------------
# Safe checkpoint loading
# --------------------------------------------------------------------------

def test_artifacts_directory_has_checkpoints():
    assert CHECKPOINTS, "no .pt files under artifacts/"


@pytest.mark.parametrize("path", CHECKPOINTS, ids=lambda p: p.name)
def test_every_checkpoint_loads_with_the_safe_loader(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(blob, dict) and blob


def test_served_loaders_build_from_their_artifacts(engine):
    assert engine.n_members >= 1
    assert engine.has_0dte

    surface = ivs.IVSurface.load()
    assert "architecture" in surface.meta

    from backend.quant.calibrate_map import MapPricer
    pricer = MapPricer()
    assert pricer.k_lo < pricer.k_hi

    served = [p for p in CHECKPOINTS
              if p.name == "hedger.pt" or (p.name.startswith("hedger_")
                                          and "_v1_" not in p.name)]
    assert served
    for path in served:
        he = hedging.HedgingEngine(checkpoint=path)
        assert "n_steps" in he.meta


def test_hedger_save_round_trips_through_the_safe_loader(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(hedging, "ARTIFACTS", tmp_path)
    for measure in ("gbm", "rbergomi"):
        name = f"hedger_{measure}_roundtrip.pt"
        meta = hedging.train(iters=1, batch=8, measure=measure,
                             out_name=name, log_every=0)
        blob = torch.load(tmp_path / name, map_location="cpu",
                          weights_only=True)
        assert blob["meta"] == meta
        policy = hedging.HedgePolicy()
        policy.load_state_dict(blob["policy"])


def test_iv_surface_save_round_trips_through_the_safe_loader(tmp_path):
    surface = ivs.IVSurface.load()
    out = tmp_path / "surface.pt"
    surface.save(out)
    again = ivs.IVSurface.load(out)
    assert again.meta == surface.meta
    for (ka, va), (kb, vb) in zip(surface.net.state_dict().items(),
                                  again.net.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb)


_CALLS: list[int] = []


def _record_call() -> int:
    _CALLS.append(1)
    return 0


class _Payload:
    """Pickles as a call to _record_call, which the unsafe loader would run."""

    def __reduce__(self):
        return (_record_call, ())


def test_iv_surface_load_refuses_pickled_code(tmp_path):
    surface = ivs.IVSurface.load()
    meta = dict(surface.meta)
    meta["payload"] = _Payload()
    path = tmp_path / "tampered.pt"
    torch.save({"state_dict": surface.net.state_dict(), "meta": meta}, path)

    _CALLS.clear()
    with pytest.raises(pickle.UnpicklingError):
        ivs.IVSurface.load(path)
    assert _CALLS == []


# --------------------------------------------------------------------------
# The floor at zero
# --------------------------------------------------------------------------

def test_negative_parity_put_is_floored_and_reported(engine):
    c = CLAMPED
    assert c["maturity"] <= ZERO_DTE_CUTOFF
    call = engine.price_with_greeks(**c, option_type="call")
    put = engine.price_with_greeks(**c, option_type="put")

    # Independent reconstruction: European parity on the call.
    forward_gap = c["spot"] - c["strike"] * math.exp(-c["rate"]
                                                     * c["maturity"])
    expected_raw_put = call["raw_price"] - forward_gap
    assert expected_raw_put < 0.0
    assert put["raw_price"] == pytest.approx(expected_raw_put, abs=1e-5)

    assert put["clamped"] is True
    assert put["price"] == 0.0
    assert call["clamped"] is False
    assert call["price"] == call["raw_price"]


@pytest.mark.parametrize("spot,maturity,option_type", [
    (100.0, 1.0, "call"), (100.0, 1.0, "put"),
    (95.0, 5.0 / 252.0, "put"), (105.0, 5.0 / 252.0, "call"),
])
def test_positive_prices_are_not_marked_clamped(engine, spot, maturity,
                                                option_type):
    out = engine.price_with_greeks(spot, 100.0, maturity, 0.25, 0.04,
                                   option_type)
    assert out["raw_price"] > 0.0
    assert out["clamped"] is False
    assert out["price"] == out["raw_price"]


def test_price_batch_unfloored_output(engine):
    c = CLAMPED
    spots = np.array([c["spot"], 100.0, 95.0])
    ones = np.ones_like(spots)
    strikes = np.full_like(spots, c["strike"])
    mats = np.array([c["maturity"], 1.0, 5.0 / 252.0])
    args = (spots, strikes, mats, ones * c["sigma"], ones * c["rate"])

    raw = engine.price_batch(*args, option_type="put", floor_at_zero=False)
    served = engine.price_batch(*args, option_type="put")
    assert raw[0] < 0.0
    assert np.count_nonzero(raw < 0.0) == 1
    np.testing.assert_array_equal(served, np.maximum(raw, 0.0))

    scalar = engine.price_with_greeks(**c, option_type="put")["raw_price"]
    assert raw[0] == pytest.approx(scalar, abs=1e-5)


# --------------------------------------------------------------------------
# numpy scalar inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("maturity", [1.0, 5.0 / 252.0])
def test_price_with_greeks_accepts_numpy_scalars(engine, option_type,
                                                 maturity):
    py = engine.price_with_greeks(100.0, 100.0, maturity, 0.25, 0.04,
                                  option_type)
    npy = engine.price_with_greeks(np.float64(100.0), np.float64(100.0),
                                   np.float64(maturity), np.float64(0.25),
                                   np.float64(0.04), option_type)
    npy32 = engine.price_with_greeks(np.float32(100.0), np.float32(100.0),
                                     np.float32(maturity), np.float32(0.25),
                                     np.float32(0.04), option_type)
    for out in (npy, npy32):
        assert type(out["price"]) is float
        assert out["price"] == pytest.approx(py["price"], rel=1e-6)
        for k, v in py["greeks"].items():
            assert out["greeks"][k] == pytest.approx(v, rel=1e-5, abs=1e-7)
    assert npy["price"] == py["price"]
