"""Offline tests for the testnet execution layer.

Everything runs against a FakeExchange transport — no network, no credentials.
The exchange fake is deliberately stateful (orders rest, fills mutate
positions) because the OMS's whole job is reconciling against a venue whose
state moves without telling you.
"""
from __future__ import annotations

import math

import pytest

from backend.trading.oms import OMS
from backend.trading.quoter import VolEstimator, as_quotes, round_to_tick
from backend.trading.risk import (KillSwitch, OrderBlocked, RiskLimits,
                                  RiskManager)
from backend.trading.testnet import (MissingCredentials, TestnetClient,
                                     TESTNET_BASE)


class FakeExchange:
    """Stateful stand-in for test.deribit.com."""

    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.position = 0.0
        self.next_id = 0

    def __call__(self, method: str, params: dict, token: str) -> object:
        if method == "public/auth":
            return {"access_token": "tok", "expires_in": 900}
        if method in ("private/buy", "private/sell"):
            self.next_id += 1
            oid = f"o{self.next_id}"
            order = {"order_id": oid, "price": params["price"],
                     "amount": params["amount"], "filled_amount": 0.0,
                     "order_state": "open",
                     "side": method.split("/")[1]}
            self.orders[oid] = order
            return {"order": order}
        if method == "private/get_open_orders_by_instrument":
            return [o for o in self.orders.values()
                    if o["order_state"] == "open"]
        if method == "private/get_order_state":
            return self.orders[params["order_id"]]
        if method == "private/get_position":
            return {"size": self.position}
        if method == "private/cancel_all_by_instrument":
            for o in self.orders.values():
                if o["order_state"] == "open":
                    o["order_state"] = "cancelled"
            return len(self.orders)
        raise AssertionError(f"unexpected call {method}")

    def fill(self, order_id: str, amount: float | None = None) -> None:
        o = self.orders[order_id]
        amount = o["amount"] if amount is None else amount
        o["filled_amount"] = o["filled_amount"] + amount
        if o["filled_amount"] >= o["amount"] - 1e-12:
            o["order_state"] = "filled"
        self.position += amount if o["side"] == "buy" else -amount


@pytest.fixture()
def venue():
    return FakeExchange()


@pytest.fixture()
def oms(venue):
    client = TestnetClient(key="k", secret="s", transport=venue)
    return OMS(client, "BTC-TEST-C")


def test_client_is_structurally_testnet_only():
    """The safety property the whole package leans on: there is no way to
    point the authenticated client at real money."""
    assert "test.deribit.com" in TESTNET_BASE
    c = TestnetClient(key="k", secret="s", transport=lambda *a: None)
    assert "test.deribit.com" in c.base


def test_missing_credentials_fail_loudly_with_instructions(venue, monkeypatch):
    monkeypatch.delenv("DERIBIT_TESTNET_KEY", raising=False)
    monkeypatch.delenv("DERIBIT_TESTNET_SECRET", raising=False)
    c = TestnetClient(transport=venue)
    with pytest.raises(MissingCredentials, match="test.deribit.com"):
        c.open_orders("BTC-TEST-C")


def test_order_lifecycle_place_fill_reconcile(oms, venue):
    """The exchange fills an order without telling us; sync() must find the
    fill AND take the venue's position as truth, not accumulate its own."""
    o = oms.place("buy", 0.5, 0.0055)
    assert o is not None and o.state == "open"
    assert oms.sync() == []                      # nothing happened yet

    venue.fill(o.order_id)                       # fills while we weren't looking
    deltas = oms.sync()
    assert len(deltas) == 1
    assert deltas[0]["amount"] == pytest.approx(0.5)
    assert oms.orders[o.order_id].state == "filled"
    assert oms.position_amount == pytest.approx(0.5)

    # A second sync reports nothing new — fills must not double-count.
    assert oms.sync() == []
    assert oms.position_amount == pytest.approx(0.5)


def test_partial_fill_is_a_delta_not_a_state_change(oms, venue):
    o = oms.place("sell", 1.0, 0.0060)
    venue.fill(o.order_id, amount=0.4)
    deltas = oms.sync()
    assert deltas[0]["amount"] == pytest.approx(0.4)
    assert oms.orders[o.order_id].state == "open"       # still working
    assert oms.position_amount == pytest.approx(-0.4)   # signed: we sold


def test_kill_switch_is_one_way(venue):
    risk = RiskManager(RiskLimits(max_order_size=1.0))
    with pytest.raises(KillSwitch):
        risk.pre_trade(side="buy", amount=5.0, position=0.0, n_open_orders=0)
    # Once tripped, even a well-formed order is refused — forever.
    with pytest.raises(KillSwitch):
        risk.pre_trade(side="buy", amount=0.1, position=0.0, n_open_orders=0)
    assert risk.tripped


def test_position_cap_blocks_the_growing_side_only():
    """At the cap, the maker must still be able to REDUCE: blocking both sides
    freezes the inventory it exists to manage."""
    risk = RiskManager(RiskLimits(max_position=1.0))
    with pytest.raises(OrderBlocked):
        risk.pre_trade(side="buy", amount=0.5, position=0.8, n_open_orders=0)
    # The reducing side goes through, and nothing tripped.
    risk.pre_trade(side="sell", amount=0.5, position=0.8, n_open_orders=0)
    assert risk.tripped is None


def test_session_loss_trips_on_equity_drift():
    risk = RiskManager(RiskLimits(max_session_loss=0.05))
    risk.note_equity(10.00)
    risk.note_equity(9.97)
    assert risk.tripped is None
    risk.note_equity(9.94)          # -0.06 from anchor
    assert risk.tripped is not None


def test_as_quotes_is_dimensionless():
    """Found live: absolute-unit A-S with dollar-scale defaults quoted
    0.0001/0.4521 around a 0.0058 BTC mark — a 77x-the-price half-spread.
    In relative units the SAME parameters must produce the same relative
    spread at any price scale, lean against inventory, and stay ordered."""
    for mark in (0.0058, 100.0, 72_000.0):
        bid, ask = as_quotes(mark, 0.0, 2.0, 1e-4, gamma=2.0, kappa=40.0)
        assert 0 < bid < mark < ask
        rel = (ask - bid) / mark
        assert 0.01 < rel < 0.15, f"relative spread {rel:.4f} at mark {mark}"
    b0, a0 = as_quotes(1.0, 0.0, 2.0, 1e-4, gamma=2.0, kappa=40.0)
    b_long, a_long = as_quotes(1.0, +3.0, 2.0, 1e-4, gamma=2.0, kappa=40.0)
    assert b_long < b0 and a_long < a0, "long inventory must lean quotes DOWN"


def test_round_to_tick_never_crosses_by_rounding():
    assert round_to_tick(0.00557, 0.0001, "buy") == pytest.approx(0.0055)
    assert round_to_tick(0.00551, 0.0001, "sell") == pytest.approx(0.0056)
    # Rounding must never produce a zero/negative price.
    assert round_to_tick(0.00003, 0.0001, "buy") == pytest.approx(0.0001)


def test_vol_estimator_needs_data_before_it_speaks():
    v = VolEstimator()
    assert v.annualized(floor=0.2) == pytest.approx(0.2)
    for i in range(20):
        v.update(float(i), 100.0 * math.exp(0.001 * (i % 2)))
    assert v.annualized(floor=0.2) > 0.2
