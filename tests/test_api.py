"""HTTP-surface tests for backend/api/main.py, run offline against the
shipped checkpoints.

Everything the dashboard reads is checked here at the boundary the browser
actually sees, because three of these behaviours are invisible from the quant
layer:

  (a) /api/model-info republishes the 0DTE checkpoint's OWN provenance - the
      calibrated flag, eta/rho/H, the Volterra kernel stamp and the note
      naming the fit. Those fields have been inside artifacts/model_0dte.pt
      since the live SPY calibration was adopted; until the endpoint returned
      them the site could not state which parameters it was serving. The block
      is resolved once at import, so these tests also pin that no request
      re-runs the git lookup, and that a container without git still answers.
  (b) /api/price and /ws/stream report the EUROPEAN no-arbitrage floor for the
      0DTE regime (discounted intrinsic) next to the served price. The price
      itself is never touched: the tests compare it against the engine call
      the handler makes, so a future "fix" that clamps the price fails here.
  (c) /api/surface holds the same one-at-a-time gate as the Monte Carlo
      endpoints - up to 6,400 rows through a 5-member ensemble is not a free
      request on a 512 MB container. `test_surface_takes_the_heavy_job_gate`
      is the evidence it is gated; it saturates the gate and expects a 503
      rather than an unbounded queue. `test_stream_prices_while_the_heavy_gate
      _is_held` is the other half: the websocket prices off the gate entirely,
      so adding /api/surface to it introduced no cycle.

No test here needs the network: every route is served in-process by
fastapi.testclient against the committed artifacts, so the module carries no
`network` marker and CI's `-m "not network"` runs all of it.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api import main as api  # noqa: E402

#: Small enough to keep the suite fast; /api/price's floor is 1,000.
MC_PATHS = 1_000

#: Inside the 0DTE box: 5 trading days, moneyness 1.0.
ZERO_DTE_T = 5.0 / 252.0
#: Inside the Asian box.
ASIAN_T = 1.0


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(api.app)


def body(**kw) -> dict:
    out = {"spot": 100.0, "strike": 100.0, "maturity": ASIAN_T,
           "sigma": 0.25, "rate": 0.04, "option_type": "call",
           "mc_paths": MC_PATHS}
    out.update(kw)
    return out


# ---------------------------------------------------------------- provenance

def test_model_info_publishes_the_0dte_checkpoint_provenance(client):
    z = client.get("/api/model-info").json()["zero_dte"]
    assert z["available"] is True
    # The three rough-Bergomi parameters and the kernel stamp must come from
    # the checkpoint, not from the API's pre-calibration fallback constants.
    meta = api.ENGINE.meta_0dte
    for key in ("eta", "rho", "H", "kernel"):
        assert z[key] == meta[key]
    assert z["calibrated"] is bool(meta["calibrated"])
    assert z["calibration_note"] == meta["calibration_note"]
    assert z["checkpoint"] == "model_0dte.pt"
    assert z["contract"] == "european"
    assert z["maturity_cutoff_years"] == pytest.approx(api.ZERO_DTE_CUTOFF)
    assert z["maturity_floor_years"] == pytest.approx(api.ZERO_DTE_MIN_MATURITY)
    assert z["moneyness"] == list(api.ZERO_DTE_MONEYNESS)
    assert z["val_rmse_bps_of_strike"] == pytest.approx(meta["val_rmse_bps"])
    # Present whether or not this checkout has a .git; may be null.
    assert "commit" in z and "commit_date" in z


def test_provenance_is_resolved_once_not_per_request(client, monkeypatch):
    """The git stamp shells out. If it ran per request, breaking it would
    break the endpoint; here it cannot, because the block is already built."""
    def explode(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("git lookup ran on a request path")

    monkeypatch.setattr(api, "_checkpoint_git_stamp", explode)
    first = client.get("/api/model-info").json()["zero_dte"]
    second = client.get("/api/model-info").json()["zero_dte"]
    assert first == second
    # Same object every time: no dict is rebuilt per request either.
    assert api.zero_dte_info() is api.zero_dte_info()


def test_git_stamp_degrades_to_nulls_without_git(monkeypatch, tmp_path):
    """The served container ships the artifacts but no .git and no git
    binary. Provenance then reports nulls instead of failing the endpoint."""
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert api._checkpoint_git_stamp(tmp_path / "model_0dte.pt") == {
        "commit": None, "commit_date": None}


# ------------------------------------------------------------- intrinsic floor

def test_intrinsic_fields_flag_a_price_under_the_european_floor():
    """Unit-pins the arithmetic: the flag and the shortfall in bps of strike."""
    floor = api.discounted_intrinsic(115.0, 100.0, ZERO_DTE_T, 0.04, "call")
    assert floor == pytest.approx(115.0 - 100.0 * math.exp(-0.04 * ZERO_DTE_T))

    under = api.intrinsic_fields("rough_bergomi_european", floor - 0.05,
                                 115.0, 100.0, ZERO_DTE_T, 0.04, "call")
    assert under["below_intrinsic"] is True
    assert under["below_intrinsic_bps_of_strike"] == pytest.approx(5.0)

    at_floor = api.intrinsic_fields("rough_bergomi_european", floor,
                                    115.0, 100.0, ZERO_DTE_T, 0.04, "call")
    assert at_floor["below_intrinsic"] is False
    # The Asian regime's floor is a different quantity and is not asserted.
    assert api.intrinsic_fields("asian_gbm", 1.0, 100.0, 100.0, 1.0, 0.04,
                                "call")["intrinsic"] is None


# The last two rows sit in the ITM corner the arbitrage audit measured: at one
# trading day the ensemble prices a 1.05- and 1.10-moneyness call about 1 bp of
# strike UNDER discounted intrinsic. The assertions below are on the arithmetic
# and on the price being untouched, not on the flag's value, so they keep
# holding when a future checkpoint clears that corner.
@pytest.mark.parametrize("spot,maturity,option_type", [
    (100.0, ZERO_DTE_T, "call"),
    (113.0, ZERO_DTE_T, "call"),
    (100.0, ZERO_DTE_T, "put"),
    (88.0, ZERO_DTE_T, "put"),
    (105.0, 1.0 / 252.0, "call"),
    (110.0, 1.0 / 252.0, "call"),
])
def test_price_reports_the_floor_without_moving_the_price(client, spot,
                                                          maturity,
                                                          option_type):
    d = client.post("/api/price", json=body(spot=spot, maturity=maturity,
                                            option_type=option_type)).json()
    assert d["regime"] == "rough_bergomi_european"
    nn = d["nn"]

    floor = api.discounted_intrinsic(spot, 100.0, maturity, 0.04, option_type)
    assert nn["intrinsic"] == pytest.approx(floor)
    assert nn["below_intrinsic"] is (nn["price"] < floor - 1e-9 * 100.0)
    assert nn["below_intrinsic_bps_of_strike"] == pytest.approx(
        max(floor - nn["price"], 0.0) / 100.0 * 1e4)

    # The served price is the raw ensemble output, unchanged by the flag.
    direct = api.ENGINE.price_with_greeks(spot, 100.0, maturity, 0.25, 0.04,
                                          option_type)["price"]
    assert nn["price"] == pytest.approx(direct)


def test_price_leaves_the_asian_regime_floor_unreported(client):
    d = client.post("/api/price", json=body()).json()
    assert d["regime"] == "asian_gbm"
    assert d["nn"]["intrinsic"] is None
    assert d["nn"]["below_intrinsic"] is None


def test_stream_frames_carry_the_floor_in_the_0dte_regime(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"spot": 100, "strike": 100, "maturity": ZERO_DTE_T,
                      "sigma": 0.25, "rate": 0.04, "hz": 15})
        ready = ws.receive_json()
        assert ready["regime"] == "rough_bergomi_european"
        frame = ws.receive_json()
        assert "error" not in frame
        assert frame["intrinsic"] == pytest.approx(
            api.discounted_intrinsic(frame["spot"], 100.0, ZERO_DTE_T, 0.04,
                                     "call"), abs=1e-3)
        assert isinstance(frame["below_intrinsic"], bool)


def test_stream_frames_omit_the_floor_in_the_asian_regime(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"spot": 100, "strike": 100, "maturity": ASIAN_T,
                      "sigma": 0.25, "rate": 0.04, "hz": 15})
        assert ws.receive_json()["regime"] == "asian_gbm"
        frame = ws.receive_json()
        assert "intrinsic" not in frame


# ----------------------------------------------------------------- domain gate

@pytest.mark.parametrize("kw,why", [
    ({"maturity": 0.5 / 252}, "below the 0DTE surrogate's 1-day floor"),
    # The uncovered band is narrow: (12/252 = 0.047619, 0.05).
    ({"maturity": 0.048}, "the dead band above 12/252 and below 0.05"),
    ({"spot": 130.0, "maturity": ZERO_DTE_T}, "0DTE moneyness above 1.15"),
    ({"spot": 80.0, "maturity": ZERO_DTE_T}, "0DTE moneyness below 0.85"),
    ({"spot": 250.0}, "Asian moneyness above 2.0"),
    ({"spot": 40.0}, "Asian moneyness below 0.5"),
    ({"sigma": 0.02}, "sigma below the trained floor of 0.05"),
])
def test_domain_gate_rejects_untrained_requests(client, kw, why):
    assert client.post("/api/price", json=body(**kw)).status_code == 422, why


def test_domain_gate_closes_the_websocket_too(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"spot": 100, "strike": 100, "maturity": 0.5 / 252,
                      "sigma": 0.25, "rate": 0.04})
        assert "error" in ws.receive_json()


# --------------------------------------------------------- put-call parity

def test_put_call_parity_zero_dte(client):
    """European parity: C - P = S - K e^{-rT}. The 0DTE regime prices a
    European contract, so this is exact up to the float32 forward."""
    call = client.post("/api/price", json=body(
        maturity=ZERO_DTE_T, option_type="call")).json()["nn"]["price"]
    put = client.post("/api/price", json=body(
        maturity=ZERO_DTE_T, option_type="put")).json()["nn"]["price"]
    assert call - put == pytest.approx(
        100.0 - 100.0 * math.exp(-0.04 * ZERO_DTE_T), abs=1e-2)


def test_put_call_parity_asian(client):
    """Fixed-strike arithmetic Asian parity: C - P = e^{-rT}(E[A] - K), with
    E[A] the discrete average's expectation under the risk-neutral GBM. The
    monitoring count comes from the served checkpoint's own metadata."""
    meta = client.get("/api/model-info").json()
    n = meta["n_monitoring_steps"]
    r, t, k, s = 0.04, ASIAN_T, 100.0, 100.0
    dt = t / n
    ea = s * math.exp(r * dt) * math.expm1(r * t) / (n * math.expm1(r * dt))

    call = client.post("/api/price", json=body(
        option_type="call")).json()["nn"]["price"]
    put = client.post("/api/price", json=body(
        option_type="put")).json()["nn"]["price"]
    assert call - put == pytest.approx(math.exp(-r * t) * (ea - k), abs=1e-2)


# ------------------------------------------------------------------- surface

@pytest.mark.parametrize("resolution", [10, 23, 40])
def test_surface_honours_the_requested_resolution(client, resolution):
    d = client.post("/api/surface", json={"sigma": 0.25, "rate": 0.04,
                                          "strike": 100.0,
                                          "resolution": resolution}).json()
    assert len(d["moneyness"]) == resolution
    assert len(d["maturity"]) == resolution
    assert len(d["prices"]) == resolution
    assert all(len(row) == resolution for row in d["prices"])
    assert d["n_prices"] == resolution * resolution


def test_surface_resolution_is_bounded(client):
    # 80x80 = 6,400 ensemble rows is the documented ceiling for one request.
    assert client.post("/api/surface",
                       json={"resolution": 81}).status_code == 422
    assert client.post("/api/surface",
                       json={"resolution": 9}).status_code == 422


def test_surface_takes_the_heavy_job_gate(client, monkeypatch):
    """Evidence that /api/surface is queued with the Monte Carlo endpoints
    rather than running unbounded alongside them: with the single gate held,
    the request is refused with the gate's own 503 instead of piling on."""
    monkeypatch.setattr(api, "HEAVY_JOB_TIMEOUT_S", 0.05)
    assert api._HEAVY_JOB_GATE.acquire(timeout=5.0)
    try:
        resp = client.post("/api/surface", json={"resolution": 10})
    finally:
        api._HEAVY_JOB_GATE.release()
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"


def test_stream_prices_while_the_heavy_gate_is_held(client, monkeypatch):
    """The gate /api/surface now takes cannot deadlock against the stream.

    The websocket prices through `anyio.to_thread.run_sync` and never touches
    `_HEAVY_JOB_GATE`, so a surface (or Monte Carlo) job holding the gate
    leaves the stream delivering frames. Held here for the whole exchange:

      * the gate is confirmed still held while the frame is priced, so the
        stream demonstrably did not get it by the holder letting go;
      * a /api/surface request issued in the same window is refused 503, so
        the gate really is the contended one;
      * were the stream ever changed to take the gate, this blocks for
        HEAVY_JOB_TIMEOUT_S and then fails instead of hanging the suite.
        Checked, not assumed: wrapping the handler's pricing call in
        heavy_job() off-tree and re-running this made the frame arrive after
        73 ms as {'tick': 1, 'spot': 99.9992, 'error': 'out of domain'}, so
        the `"error" not in frame` assertion below is the one that fires.

    What this does NOT claim: Starlette dispatches sync handlers on anyio's
    default thread limiter (40 tokens) and the websocket's pricing call draws
    on the same pool, so enough handlers queued on the gate can still make a
    frame wait for a thread. That is bounded starvation - every gate holder
    finishes and releases - not a cycle.
    """
    monkeypatch.setattr(api, "HEAVY_JOB_TIMEOUT_S", 0.05)
    assert api._HEAVY_JOB_GATE.acquire(timeout=5.0)
    try:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"spot": 100, "strike": 100, "maturity": ASIAN_T,
                          "sigma": 0.25, "rate": 0.04, "hz": 15})
            assert ws.receive_json()["status"] == "ready"
            frame = ws.receive_json()
            assert "error" not in frame
            assert frame["price"] > 0.0
            # Still held: the stream priced without acquiring it.
            assert api._HEAVY_JOB_GATE.acquire(blocking=False) is False
            # And the gate it did not take is the one that is contended.
            assert client.post("/api/surface",
                               json={"resolution": 10}).status_code == 503
    finally:
        api._HEAVY_JOB_GATE.release()
