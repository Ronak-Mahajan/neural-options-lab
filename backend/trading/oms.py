"""Order and position tracking with reconciliation against the exchange.

The unglamorous 70% of a trading system. The strategy's view of the world
("I am quoting 761 bid / 764 ask, flat") and the exchange's view ("your bid
filled 40 seconds ago") drift apart the moment anything happens, and every
serious failure in a small trading system is some version of acting on the
stale view. The rule here is that THE EXCHANGE IS RIGHT: local state is a
cache, sync() refreshes it, and inventory is read back from the venue rather
than inferred from what we think our fills were.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .testnet import TestnetClient


@dataclass
class Order:
    order_id: str
    instrument: str
    side: str                  # "buy" | "sell"
    price: float
    amount: float
    filled: float = 0.0
    state: str = "open"        # open | filled | cancelled
    created: float = field(default_factory=time.time)


class OMS:
    """Track our open orders and position on ONE instrument."""

    def __init__(self, client: TestnetClient, instrument: str) -> None:
        self.client = client
        self.instrument = instrument
        self.orders: dict[str, Order] = {}
        self.position_amount = 0.0     # signed contracts, exchange's number
        self.fills: list[dict] = []    # observed fill deltas, for the log

    # -- actions ------------------------------------------------------------ #

    def place(self, side: str, amount: float, price: float,
              label: str = "") -> Order | None:
        """Place a post-only limit order; returns None if the venue rejected
        it (e.g. post-only would cross). A maker being rejected for crossing
        is routine, not an error."""
        from .testnet import TestnetAPIError
        try:
            result = self.client.limit_order(self.instrument, side, amount,
                                             price, label=label)
        except TestnetAPIError as exc:
            if str(exc.code) in ("10006", "post_only_reject"):
                return None
            raise
        o = result["order"]
        order = Order(order_id=o["order_id"], instrument=self.instrument,
                      side=side, price=float(o["price"]),
                      amount=float(o["amount"]),
                      filled=float(o.get("filled_amount", 0.0)),
                      state="open" if o["order_state"] in ("open", "untriggered")
                      else o["order_state"])
        self.orders[order.order_id] = order
        return order

    def cancel_all(self) -> None:
        self.client.cancel_all(self.instrument)
        for o in self.orders.values():
            if o.state == "open":
                o.state = "cancelled"

    # -- reconciliation ------------------------------------------------------ #

    def sync(self) -> list[dict]:
        """Refresh local state from the exchange. Returns fill deltas seen.

        Any order we track that the venue no longer lists is either filled or
        cancelled; the venue's filled_amount decides which, and the position
        is ALWAYS overwritten with the venue's number rather than accumulated
        locally - accumulate and you eventually double-count a fill you also
        inferred, which is how a hedger ends up twice as long as it thinks.
        """
        open_now = {o["order_id"]: o
                    for o in self.client.open_orders(self.instrument)}
        deltas: list[dict] = []
        for oid, order in self.orders.items():
            if order.state != "open":
                continue
            if oid in open_now:
                new_filled = float(open_now[oid].get("filled_amount", 0.0))
                if new_filled > order.filled + 1e-12:
                    deltas.append({"order_id": oid, "side": order.side,
                                   "price": order.price,
                                   "amount": new_filled - order.filled})
                    order.filled = new_filled
            else:
                final = self._final_state(oid)
                got = float(final.get("filled_amount", 0.0))
                if got > order.filled + 1e-12:
                    deltas.append({"order_id": oid, "side": order.side,
                                   "price": order.price,
                                   "amount": got - order.filled})
                order.filled = got
                order.state = ("filled" if final.get("order_state") == "filled"
                               else "cancelled")
        pos = self.client.position(self.instrument)
        # Deribit reports option size in contracts, signed (+long / -short).
        self.position_amount = float(pos.get("size", 0.0))
        self.fills.extend(deltas)
        return deltas

    def _final_state(self, order_id: str) -> dict:
        return self.client._private("get_order_state", order_id=order_id)

    @property
    def open(self) -> list[Order]:
        return [o for o in self.orders.values() if o.state == "open"]
