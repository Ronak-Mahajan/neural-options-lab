"""HTTP-surface tests for backend/api/main.py, run offline against the
shipped checkpoints. Each behaviour here exists only at the boundary the
browser sees, so the quant-layer tests cannot cover it:

  * /api/model-info publishes the 0DTE checkpoint's own provenance
    (calibrated flag, eta/rho/H, kernel stamp, calibration note). The block
    is built once at import, so no request runs the git lookup and a
    container without git still answers.
  * /api/price and /ws/stream report the European no-arbitrage floor next to
    the served 0DTE price, which is floored at zero with a `clamped` flag. A
    put is checked against a reconstruction from the call and European
    put-call parity, and one pinned put that the network prices below zero
    must come back clamped and flagged below intrinsic.
  * /api/surface takes the one-at-a-time simulation gate, and the websocket
    prices without it, so the two cannot deadlock.
  * The request boundary: non-finite numbers and malformed desk-note inputs
    are 422s in the JSON error shape, oversized bodies are 413s, every
    pricing route applies the trained-domain gate, and both job gates refuse
    with 503 and Retry-After when their queue is full.
  * /api/health answers 503 without the pricing checkpoint, and /api/hedge
    answers 503 for a dynamics whose served policy is missing.

Every route is served in-process by fastapi.testclient against the committed
artifacts, so the module carries no `network` marker and CI's
`-m "not network"` runs all of it.
"""
from __future__ import annotations

import inspect
import json
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


def test_model_info_publishes_the_0dte_checkpoint_provenance(client):
    z = client.get("/api/model-info").json()["zero_dte"]
    assert z["available"] is True
    # eta, rho, H and the kernel stamp are the checkpoint's own values. The
    # API's fallback constants (1.5, -0.7, 0.1) differ from all three.
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
    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("git lookup ran on a request path")

    monkeypatch.setattr(api, "_checkpoint_git_stamp", explode)
    first = client.get("/api/model-info").json()["zero_dte"]
    second = client.get("/api/model-info").json()["zero_dte"]
    assert first == second
    # Same object every time: no dict is rebuilt per request either.
    assert api.zero_dte_info() is api.zero_dte_info()


def test_git_stamp_degrades_to_nulls_without_git(monkeypatch, tmp_path):
    """The served container ships the artifacts but no .git and no git
    binary. Provenance then reports nulls and the endpoint still answers."""
    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert api._checkpoint_git_stamp(tmp_path / "model_0dte.pt") == {
        "commit": None, "commit_date": None}


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
# strike under discounted intrinsic. The assertions cover the arithmetic and
# the served price and leave the flag's value free, so they hold when a
# future checkpoint clears that corner.
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

    # The expected pre-floor value: the network's call, and for a put the
    # call minus the European parity term S - K e^{-rT}.
    call_raw = api.ENGINE.price_with_greeks(spot, 100.0, maturity, 0.25, 0.04,
                                            "call")["raw_price"]
    raw = (call_raw if option_type == "call"
           else call_raw - (spot - 100.0 * math.exp(-0.04 * maturity)))
    assert nn["price"] == pytest.approx(max(raw, 0.0), abs=1e-4)
    assert nn["clamped"] is (raw < 0.0)

    floor = api.discounted_intrinsic(spot, 100.0, maturity, 0.04, option_type)
    assert nn["intrinsic"] == pytest.approx(floor)
    shortfall = max(floor - raw, 0.0)
    # The reconstruction agrees with the float32 network to about 1e-5, so the
    # flag is asserted wherever the pre-floor value is clear of the floor.
    if floor - raw > 1e-3:
        assert nn["below_intrinsic"] is True
    elif floor - raw < -1e-3:
        assert nn["below_intrinsic"] is False
    assert nn["below_intrinsic_bps_of_strike"] == pytest.approx(
        shortfall / 100.0 * 1e4, abs=1e-2)


def test_price_floors_a_negative_parity_put_and_says_so(client):
    """A 0DTE point where the call sits under its parity floor, so the
    parity-derived put is negative before the zero floor (-0.0821 at a
    $100 strike on the shipped checkpoint). The served price is 0, and the
    response says the floor acted and that the pre-floor value is under
    intrinsic."""
    d = client.post("/api/price", json=body(
        spot=101.6557, maturity=0.04125, sigma=0.05056, rate=0.03038,
        option_type="put")).json()
    assert d["regime"] == "rough_bergomi_european"
    nn = d["nn"]
    assert nn["price"] == 0.0
    assert nn["clamped"] is True
    assert nn["below_intrinsic"] is True
    assert nn["intrinsic"] == 0.0
    assert nn["below_intrinsic_bps_of_strike"] == pytest.approx(8.21, abs=0.05)


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
        assert isinstance(frame["clamped"], bool)


def test_stream_frames_omit_the_floor_in_the_asian_regime(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"spot": 100, "strike": 100, "maturity": ASIAN_T,
                      "sigma": 0.25, "rate": 0.04, "hz": 15})
        assert ws.receive_json()["regime"] == "asian_gbm"
        frame = ws.receive_json()
        assert "intrinsic" not in frame


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


@pytest.mark.parametrize("route", ["/api/explain", "/api/benchmark"])
@pytest.mark.parametrize("kw,why", [
    ({"sigma": 0.02}, "sigma below the trained floor of 0.05"),
    ({"maturity": 0.001}, "below the 0DTE surrogate's 1-day floor"),
    ({"maturity": 0.048}, "the band above 12/252 and below 0.05"),
    ({"spot": 250.0}, "Asian moneyness above 2.0"),
])
def test_domain_gate_covers_explain_and_benchmark(client, route, kw, why):
    """The attribution and latency panels refuse the contracts /api/price
    refuses, and refuse before any simulation runs."""
    params = {k: v for k, v in body(**kw).items() if k != "mc_paths"}
    resp = client.post(route, json=params)
    assert resp.status_code == 422, why
    assert isinstance(resp.json()["detail"], str)


def test_surface_rejects_sigma_below_trained_floor(client):
    """The surface grid sits inside the Asian box; sigma is the one input a
    caller can push out of it."""
    assert client.post("/api/surface", json={"sigma": 0.02,
                                             "resolution": 10}).status_code == 422
    assert client.post("/api/surface", json={"sigma": 0.05,
                                             "resolution": 10}).status_code == 200


def test_domain_messages_print_distinct_operands(client):
    """A value a hair outside a bound is printed at full precision, so the
    message cannot read `sigma 0.05 outside [0.05, 0.8]`."""
    detail = client.post("/api/price",
                         json=body(sigma=0.0499999)).json()["detail"]
    assert "0.0499999" in detail
    detail = client.post("/api/price",
                         json=body(maturity=1 / 252 - 1e-9)).json()["detail"]
    assert repr(1 / 252 - 1e-9) in detail and "1/252" in detail


NON_FINITE = [
    ("/api/price", b'{"spot": NaN}', "spot"),
    ("/api/price", b'{"strike": Infinity}', "strike"),
    ("/api/price", b'{"sigma": 1e400}', "sigma"),
    ("/api/implied-vol", b'{"price": NaN}', "price"),
    ("/api/convergence", b'{"rate": NaN}', "rate"),
    ("/api/surface", b'{"sigma": NaN}', "sigma"),
    ("/api/iv-surface", b'{"rate": -Infinity}', "rate"),
    ("/api/benchmark", b'{"maturity": NaN}', "maturity"),
    ("/api/hedge", b'{"cost": NaN}', "cost"),
    ("/api/explain", b'{"spot": NaN}', "spot"),
    ("/api/risk-report", b'{"nn_price": NaN, "bs_cvar": 1, "deep_cvar": 1,'
                         b' "attributions": {}}', "nn_price"),
]


@pytest.mark.parametrize("route,payload,field", NON_FINITE)
def test_non_finite_numbers_are_422(client, route, payload, field):
    """Python's JSON parser admits NaN and Infinity. Every request model
    rejects them, and the error body carries no echo of the input, which
    could not be serialised."""
    resp = client.post(route, content=payload,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/json")
    errors = resp.json()["detail"]
    assert {"loc": ["body", field], "type": "finite_number",
            "msg": "Input should be a finite number"} in errors
    assert all(set(e) == {"loc", "msg", "type"} for e in errors)


def test_validation_error_omits_request_body(client):
    marker = "echo-marker-7f3a"
    resp = client.post("/api/risk-report", json={"contract": marker})
    assert resp.status_code == 422
    assert marker not in resp.text


def test_spot_and_strike_are_bounded(client):
    """At 1e308 the float32 price overflows to inf, which is not JSON."""
    resp = client.post("/api/price", json=body(spot=1e308, strike=1e308))
    assert resp.status_code == 422


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
    """/api/surface queues with the Monte Carlo endpoints: with the single
    gate held, the request times out with the gate's own 503."""
    monkeypatch.setattr(api, "HEAVY_JOB_TIMEOUT_S", 0.05)
    assert api._HEAVY_JOB_GATE.acquire(timeout=5.0)
    try:
        resp = client.post("/api/surface", json={"resolution": 10})
    finally:
        api._HEAVY_JOB_GATE.release()
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"


def test_stream_prices_while_the_heavy_gate_is_held(client, monkeypatch):
    """The stream prices through `anyio.to_thread.run_sync` and never takes
    `_HEAVY_JOB_GATE`, so a job holding the gate leaves frames flowing.

    The gate is held for the whole exchange. A frame is priced while it is
    still held, and a /api/surface request in the same window is refused
    with 503, so the held gate is the contended one. A stream that took the
    gate would wait HEAVY_JOB_TIMEOUT_S and send an error frame, and the
    `"error" not in frame` assertion would fail.

    Scope: the stream's pricing call shares anyio's 40-token thread limiter
    with the sync handlers, so a frame can wait for a thread. The capped
    queues in main.py bound that wait.
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


def test_full_simulation_queue_refuses_at_once(client, monkeypatch):
    """Every waiter holds a pool thread, so the queue is capped. With the cap
    at zero the request is refused without waiting for the timeout."""
    monkeypatch.setattr(api, "HEAVY_QUEUE_MAX", 0)
    monkeypatch.setattr(api, "HEAVY_JOB_TIMEOUT_S", 60.0)
    resp = client.post("/api/surface", json={"resolution": 10})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "5"
    assert api._HEAVY_QUEUE.waiting == 0


@pytest.mark.parametrize("route,payload", [
    ("/api/explain", {}),
    ("/api/implied-vol", {"price": 5.0}),
])
def test_inference_gate_bounds_explain_implied_vol(client, monkeypatch,
                                                       route, payload):
    monkeypatch.setattr(api, "LIGHT_JOB_TIMEOUT_S", 0.05)
    held = 0
    try:
        while api._LIGHT_JOB_GATE.acquire(blocking=False):
            held += 1
        resp = client.post(route, json=payload)
    finally:
        for _ in range(held):
            api._LIGHT_JOB_GATE.release()
    assert held == 4
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"
    assert client.post(route, json=payload).status_code == 200


def test_oversized_body_is_413(client):
    """A declared Content-Length over the cap is refused before the body is
    read, whatever the route would have made of it."""
    big = json.dumps({"contract": "x" * (api.MAX_BODY_BYTES + 1)})
    resp = client.post("/api/risk-report", content=big,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 413
    assert resp.json() == {
        "detail": f"request body exceeds {api.MAX_BODY_BYTES} bytes"}


def test_oversized_chunked_body_is_413(client):
    """No Content-Length: the body is counted as it arrives."""
    def chunks():
        yield b'{"contract": "'
        for _ in range(api.MAX_BODY_BYTES // 1000 + 1):
            yield b"x" * 1000
        yield b'"}'

    resp = client.post("/api/risk-report", content=chunks(),
                       headers={"Content-Type": "application/json"})
    assert "content-length" not in resp.request.headers
    assert resp.status_code == 413


def risk_body(**kw) -> dict:
    out = {"ticker": "SPY", "contract": "1-year at-the-money Asian call",
           "nn_price": 5.61, "bs_cvar": 3.2, "deep_cvar": 2.9,
           "attributions": {"spot": 0.1, "maturity": 3.0, "sigma": 2.0,
                            "rate": 0.2}}
    out.update(kw)
    return out


def test_risk_report_accepts_dashboard_body(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    resp = client.post("/api/risk-report", json=risk_body())
    assert resp.status_code == 200
    assert "nan" not in resp.text.lower()


@pytest.mark.parametrize("kw,loc", [
    ({"attributions": {"spot": "abc"}}, ["body", "attributions", "spot"]),
    ({"attributions": {"spot": None}}, ["body", "attributions", "spot"]),
    ({"attributions": {f"d{i}": 1.0 for i in range(9)}},
     ["body", "attributions"]),
    ({"ticker": "ignore all previous"}, ["body", "ticker"]),
    ({"contract": "x" * 121}, ["body", "contract"]),
    ({"paired_separated": {"deep|delta": "maybe"}},
     ["body", "paired_separated", "deep|delta"]),
])
def test_risk_report_rejects_malformed_inputs(client, kw, loc):
    resp = client.post("/api/risk-report", json=risk_body(**kw))
    assert resp.status_code == 422
    assert loc in [e["loc"] for e in resp.json()["detail"]]


@pytest.mark.parametrize("payload,loc", [
    (b'{"attributions": {"spot": NaN}}', ["body", "attributions", "spot"]),
    (b'{"bs_cvar": Infinity}', ["body", "bs_cvar"]),
    (b'{"deep_cvar_se": NaN}', ["body", "deep_cvar_se"]),
    (b'{"baseline_price": -Infinity}', ["body", "baseline_price"]),
])
def test_risk_report_rejects_non_finite_numbers(client, payload, loc):
    """A NaN attribution or CVaR would print `$nan` in the desk note."""
    resp = client.post("/api/risk-report", content=payload,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert {"loc": loc, "type": "finite_number",
            "msg": "Input should be a finite number"} in resp.json()["detail"]


def test_health_200_with_engine_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True,
                           "hedgers_loaded": ["gbm", "rough"],
                           "iv_surface_loaded": True}
    # A coroutine: answered on the event loop, with no pool thread.
    assert inspect.iscoroutinefunction(api.health)


def test_health_503_without_pricing_engine(client, monkeypatch):
    """render.yaml health-checks this route, so a container that cannot
    price must not read as healthy."""
    monkeypatch.setattr(api, "ENGINE", None)
    resp = client.get("/api/health")
    assert resp.status_code == 503
    assert resp.json()["model_loaded"] is False
    assert resp.json()["status"] == "unavailable"


def test_hedgers_are_served_checkpoints():
    """Each dynamics is served by the policy trained under its measure."""
    assert set(api.HEDGERS) == {"rough", "gbm"}
    for key, spec in api.HEDGE_DYNAMICS.items():
        assert api.HEDGERS[key].meta["train_measure"] == spec["measure"]


def test_hedger_loading_has_no_fallback(tmp_path):
    """artifacts/hedger.pt is the GAN-measure policy and serves neither
    dynamics. A directory without the served checkpoints loads nothing."""
    assert (api.ARTIFACTS / "hedger.pt").exists()
    assert api.load_hedgers(tmp_path) == {}


def test_hedge_503_for_missing_policy(client, monkeypatch):
    monkeypatch.setattr(api, "HEDGERS", {"rough": api.HEDGERS["rough"]})
    resp = client.post("/api/hedge", json={"dynamics": "gbm"})
    assert resp.status_code == 503
    assert "hedger_gbm.pt" in resp.json()["detail"]


class CountingHedger:
    """Stands in for a HedgingEngine: compare() is a few seconds of Monte
    Carlo, and these tests are about the handler around it."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta
        self.calls: list[tuple] = []

    def compare(self, sigma, rate, cost, primary, measures):
        self.calls.append((sigma, rate, cost))
        block = {"deep": {"cvar95": -0.01, "pnl": [0.0, 0.1]}, "note": "n"}
        return {"cost": cost, "measure": primary, "deep": block["deep"],
                "by_measure": {primary: block}}


def test_hedge_reply_is_memoised(client, monkeypatch):
    """compare() is seeded, so one run serves every repeat of a question.
    Under the rough dynamics sigma and rate come from the calibration, and
    the sliders do not enter the key."""
    rough = CountingHedger({"train_measure": "rbergomi_jumps",
                            "measure_params": {"xi": 0.04, "rate": 0.03}})
    gbm = CountingHedger({"train_measure": "gbm"})
    monkeypatch.setattr(api, "HEDGERS", {"rough": rough, "gbm": gbm})
    monkeypatch.setattr(api, "_HEDGE_CACHE", api._ResponseCache(max_entries=2))

    first = client.post("/api/hedge", json={"dynamics": "rough", "sigma": 0.2})
    again = client.post("/api/hedge", json={"dynamics": "rough", "sigma": 0.6})
    assert first.status_code == 200 and first.content == again.content
    assert rough.calls == [(pytest.approx(0.2), 0.03, 0.01)]
    d = first.json()
    assert d["sigma_source"] == "SPY calibration"
    assert d["policy_trained_on"] == "rbergomi_jumps"
    # by_measure keeps the statistics and drops its copy of the P&L array.
    assert d["deep"]["pnl"] == [0.0, 0.1]
    assert d["by_measure"]["rbergomi_jumps"]["deep"] == {"cvar95": -0.01}

    # Under Black-Scholes the sliders are the question.
    client.post("/api/hedge", json={"dynamics": "gbm", "sigma": 0.2})
    client.post("/api/hedge", json={"dynamics": "gbm", "sigma": 0.2})
    client.post("/api/hedge", json={"dynamics": "gbm", "sigma": 0.3})
    assert [c[0] for c in gbm.calls] == [0.2, 0.3]

    # Two entries: the rough reply was evicted by the two gbm ones.
    client.post("/api/hedge", json={"dynamics": "rough"})
    assert len(rough.calls) == 2


def test_stream_refuses_beyond_client_cap(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_STREAM_CLIENTS", 0)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"spot": 100, "strike": 100})
        assert ws.receive_json() == {
            "error": "stream capacity reached; try again shortly"}
    assert api._stream_clients == 0


@pytest.mark.parametrize("config", [
    '{"spot": NaN}', '{"sigma": 0.9}', '{"spot": null}', '{"hz": "fast"}',
    "[1, 2]", '"hi"', "null", "not json",
])
def test_stream_config_errors_fixed_messages(client, config):
    """The refusal names the offending fields and the expected shape. The
    validator's own text (library name, version, documentation URL) and
    Python exception text stay on the server."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(config)
        error = ws.receive_json()["error"]
    assert error == "config must be a JSON object" or (
        error.startswith("invalid config (")
        and error.endswith(api.STREAM_CONFIG_HELP))
    for leak in ("pydantic", "http", "object has no attribute", "float()"):
        assert leak not in error


# ---------------------------------------------------------------------------
# The simulation gate on every heavy route
# ---------------------------------------------------------------------------

# /api/price, /api/convergence and /api/benchmark in flight together have
# exhausted the container's memory, so each heavy route must queue on the one
# gate. The two cached routes get an empty cache, so the request reaches the
# gate instead of answering from memory.
HEAVY_ROUTES = [
    ("/api/price", body()),
    ("/api/convergence", {"path_counts": [1_000, 2_000]}),
    ("/api/benchmark", {}),
    ("/api/hedge", {"dynamics": "gbm", "sigma": 0.3, "cost": 0.02}),
    ("/api/iv-surface", {"resolution": 11}),
    ("/api/surface", {"resolution": 10}),
]


@pytest.mark.parametrize("route,payload", HEAVY_ROUTES,
                         ids=[r for r, _ in HEAVY_ROUTES])
def test_every_heavy_route_takes_the_simulation_gate(client, monkeypatch,
                                                     route, payload):
    """With the single gate held, each route times out with the gate's own
    503 rather than running its simulation or batch alongside."""
    monkeypatch.setattr(api, "HEAVY_JOB_TIMEOUT_S", 0.05)
    monkeypatch.setattr(api, "_CONVERGENCE_CACHE", api._ResponseCache(4))
    monkeypatch.setattr(api, "_HEDGE_CACHE", api._ResponseCache(4))
    assert api._HEAVY_JOB_GATE.acquire(timeout=5.0)
    try:
        resp = client.post(route, json=payload)
    finally:
        api._HEAVY_JOB_GATE.release()
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "10"
    assert resp.json()["detail"] == "Simulation queue is saturated; retry shortly"


# ---------------------------------------------------------------------------
# Response headers, caching and methods
# ---------------------------------------------------------------------------

def csp_directives(csp: str) -> dict[str, list[str]]:
    return {d.split()[0]: d.split()[1:] for d in csp.split("; ")}


@pytest.mark.parametrize("method,path", [
    ("GET", "/"), ("GET", "/methodology"), ("GET", "/app.js"),
    ("GET", "/api/health"), ("HEAD", "/api/health"), ("GET", "/api/nope"),
    ("GET", "/missing.html"),
])
def test_security_headers_on_every_response(client, method, path):
    resp = client.request(method, path)
    for name, value in api.SECURITY_HEADERS.items():
        assert resp.headers.get(name) == value, (path, name)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_api_doc_pages_carry_every_header_but_the_page_csp(client, path):
    """Swagger UI and ReDoc load from cdn.jsdelivr.net and start with an
    inline script, which the dashboard's policy blocks, so their pages get
    no CSP. Every other header still applies."""
    resp = client.get(path)
    assert resp.status_code == 200, path
    assert "Content-Security-Policy" not in resp.headers
    for name, value in api.SECURITY_HEADERS.items():
        if name != "Content-Security-Policy":
            assert resp.headers.get(name) == value, (path, name)
    csp = client.get("/openapi.json").headers["Content-Security-Policy"]
    assert csp == api.CONTENT_SECURITY_POLICY


def test_security_headers_on_a_refused_body(client):
    """The 413 is written by BodySizeLimit, inside the header middleware."""
    resp = client.post("/api/risk-report",
                       content=b"x" * (api.MAX_BODY_BYTES + 1),
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 413
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"


def test_content_security_policy_is_enforcing_and_narrow(client):
    headers = client.get("/").headers
    csp = headers["Content-Security-Policy"]
    directives = csp_directives(csp)
    assert directives["default-src"] == ["'self'"]
    assert directives["script-src"] == ["'self'", "https://cdn.plot.ly"]
    assert directives["connect-src"] == ["'self'"]
    assert directives["frame-ancestors"] == ["'none'"]
    assert directives["object-src"] == ["'none'"]
    # The pinned Plotly build, gl3d surfaces included, runs without eval.
    assert "'unsafe-eval'" not in csp
    # Inline script is never allowed; inline style is (Plotly writes it).
    assert "'unsafe-inline'" not in directives["script-src"]
    assert "Content-Security-Policy-Report-Only" not in headers


def test_every_external_source_the_pages_load_is_allowed(client):
    """Each third-party origin index.html and methodology.html reference sits
    in the directive that governs it, so a new CDN tag cannot ship without
    its CSP entry."""
    import re
    directives = csp_directives(
        client.get("/").headers["Content-Security-Policy"])
    checked = 0
    for page in ("/", "/methodology"):
        html = client.get(page).text
        for src in re.findall(r'<script[^>]+src="(https?://[^"]+)"', html):
            assert "/".join(src.split("/")[:3]) in directives["script-src"], src
            checked += 1
        for tag in re.findall(r"<link[^>]+>", html):
            href = re.search(r'href="(https?://[^"]+)"', tag)
            if href and 'rel="stylesheet"' in tag:
                origin = "/".join(href.group(1).split("/")[:3])
                assert origin in directives["style-src"], href.group(1)
                checked += 1
    assert checked >= 3
    assert "https://fonts.gstatic.com" in directives["font-src"]
    # The favicon is a data: URI.
    assert "data:" in directives["img-src"]


@pytest.mark.parametrize("path", ["/", "/index.html", "/app.js",
                                  "/styles.css", "/methodology",
                                  "/methodology.html"])
def test_pages_scripts_and_styles_revalidate_on_every_load(client, path):
    """No content hash in these names, so a cached copy is revalidated (a 304
    against the ETag) and a deploy is seen at once."""
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-cache"


def test_images_are_cached_for_an_hour(client):
    resp = client.get("/hero.png")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "public, max-age=3600"


def test_revalidation_returns_304_with_the_same_policy(client):
    first = client.get("/app.js")
    again = client.get("/app.js",
                       headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.headers["Cache-Control"] == "no-cache"


def test_api_responses_carry_no_static_cache_policy(client):
    assert "Cache-Control" not in client.get("/api/health").headers


def test_head_methodology_and_health(client):
    """Link checkers and uptime monitors that probe with HEAD see the page
    and the service as up."""
    get = client.get("/methodology")
    head = client.head("/methodology")
    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-length"] == get.headers["content-length"]
    health = client.head("/api/health")
    assert health.status_code == 200 and health.content == b""


@pytest.mark.parametrize("method,path,allow", [
    ("GET", "/api/price", "POST"),
    ("HEAD", "/api/price", "POST"),
    ("GET", "/api/hedge", "POST"),
    ("PUT", "/api/price", "POST"),
    ("POST", "/api/health", "GET, HEAD"),
])
def test_method_mismatch_on_an_api_route_is_405(client, method, path, allow):
    resp = client.request(method, path)
    assert resp.status_code == 405
    assert set(resp.headers["Allow"].split(", ")) == set(allow.split(", "))
    if method != "HEAD":
        assert resp.json() == {"detail": "Method Not Allowed"}


def test_unknown_api_path_is_a_json_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_static_site_still_served_beside_the_api_guard(client):
    assert client.get("/").status_code == 200
    assert client.get("/methodology.html").status_code == 200
    assert client.head("/app.js").status_code == 200


# ---------------------------------------------------------------------------
# WebSocket handshake origin and message size
# ---------------------------------------------------------------------------

STREAM_CONFIG = {"spot": 100, "strike": 100, "maturity": ASIAN_T,
                 "sigma": 0.25, "rate": 0.04, "hz": 15}


@pytest.mark.parametrize("origin", [
    "https://evil.example", "null", "https://testserver.evil.example",
    "http://evil.example:8000", "file://",
])
def test_stream_refuses_a_cross_origin_handshake(client, origin):
    """CORS does not cover WebSocket handshakes. A page on another origin is
    refused before the socket is accepted, so it takes no stream slot."""
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream",
                                      headers={"origin": origin}):
            pass  # pragma: no cover
    assert exc.value.code == 1008
    assert api._stream_clients == 0


@pytest.mark.parametrize("origin", [
    "http://testserver",                        # the request's own host
    "https://neural-options-lab.onrender.com",  # the deployed site
    "http://localhost:5173",                    # a local frontend dev server
    "http://127.0.0.1:8123",
])
def test_stream_accepts_same_origin_and_local_dev(client, origin):
    with client.websocket_connect("/ws/stream",
                                  headers={"origin": origin}) as ws:
        ws.send_json(STREAM_CONFIG)
        assert ws.receive_json()["status"] == "ready"


@pytest.mark.parametrize("text", [
    json.dumps({**STREAM_CONFIG, "pad": [0] * 400}),
    json.dumps({**STREAM_CONFIG, "pad": "x" * (2 * 1024 * 1024)}),
    # Under the cap in characters, over it in UTF-8 bytes.
    json.dumps({"spot": 100, "pad": "€" * 400}, ensure_ascii=False),
], ids=["padded-array", "2MiB-string", "multibyte"])
def test_stream_refuses_an_oversized_config_before_parsing(client, text):
    from starlette.websockets import WebSocketDisconnect
    assert len(text.encode("utf-8")) > api.MAX_STREAM_MESSAGE_BYTES
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(text)
        assert ws.receive_json() == {
            "error": f"config exceeds {api.MAX_STREAM_MESSAGE_BYTES} bytes"}
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1009


def test_dashboard_config_fits_the_message_cap():
    """The dashboard's config at the widest values the schema admits stays
    well under the cap."""
    widest = {"spot": 999999.99, "strike": 999999.99, "sigma": 0.8,
              "rate": 0.1, "maturity": 1.9999999999999998,
              "option_type": "call", "hz": 15}
    assert len(json.dumps(widest)) < api.MAX_STREAM_MESSAGE_BYTES // 4


def test_stream_refuses_a_binary_config(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(b'{"spot": 100}')
        assert ws.receive_json() == {"error": "config must be a JSON object"}


def test_stream_closes_on_a_second_message(client):
    """The stream takes one config message. A later one closes it and frees
    the slot, so nothing piles up unread in the server's receive buffer."""
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(STREAM_CONFIG)
        assert ws.receive_json()["status"] == "ready"
        ws.send_text("x" * 64)
        closing = None
        with pytest.raises(WebSocketDisconnect) as exc:
            for _ in range(200):
                msg = ws.receive_json()
                if "tick" not in msg:
                    closing = msg
    assert closing == {"error": "the stream takes one config message"}
    assert exc.value.code == 1008
    assert api._stream_clients == 0


def test_stream_slot_is_freed_when_the_client_leaves(client):
    import time
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json(STREAM_CONFIG)
        assert ws.receive_json()["status"] == "ready"
        assert ws.receive_json()["tick"] == 1
        assert api._stream_clients == 1
    for _ in range(50):
        if api._stream_clients == 0:
            break
        time.sleep(0.05)
    assert api._stream_clients == 0
