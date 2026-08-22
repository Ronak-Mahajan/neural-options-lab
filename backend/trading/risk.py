"""Pre-trade risk checks and the kill switch.

Every check here exists because its absence has a specific failure story: a
quoting loop with a sign bug builds a position forever (max_position); a
mis-priced quote for 100x the intended size fills instantly (max_order_size);
a re-quote loop that never cancels leaves a ladder of stale orders working
(max_open_orders); and a strategy that is simply wrong bleeds until stopped
(max_session_loss). The kill switch is one-way BY DESIGN: once tripped, the
only permitted action is cancelling - a limit that un-trips itself is a limit
the next bug walks straight through. Re-arming requires constructing a new
session, i.e. a human restarting the process.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position: float = 5.0        # |contracts|, net
    max_order_size: float = 1.0      # contracts per order
    max_open_orders: int = 4
    max_session_loss: float = 0.05   # in the account currency (BTC on testnet)


class KillSwitch(Exception):
    """A hard breach: the loop must cancel everything and STOP."""


class OrderBlocked(Exception):
    """A soft refusal: skip THIS order, keep running. Position and open-order
    caps are soft because hitting them is the risk system working, not a
    malfunction - a maker at its inventory cap should stop growing the
    position, not shut down."""


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.tripped: str | None = None
        self._equity_start: float | None = None

    def note_equity(self, equity: float) -> None:
        """Feed the account equity each tick; the first call sets the anchor.

        Session P&L is measured as equity drift from the anchor, which charges
        the strategy for fees and mark moves on inventory - the two costs a
        realized-only P&L conveniently forgets.
        """
        if self._equity_start is None:
            self._equity_start = equity
        if (self._equity_start - equity) > self.limits.max_session_loss:
            self.trip(f"session loss {self._equity_start - equity:.4f} > "
                      f"{self.limits.max_session_loss}")

    def trip(self, reason: str) -> None:
        if self.tripped is None:
            self.tripped = reason

    def pre_trade(self, *, side: str, amount: float, position: float,
                  n_open_orders: int) -> None:
        """KillSwitch = stop the session. OrderBlocked = skip this order.

        A malformed order (bad size) trips the switch: it means the code
        computing orders is wrong, and a wrong order generator must not get a
        second attempt. Hitting a position or order-count cap only blocks:
        that is the limit doing its job.
        """
        if self.tripped:
            raise KillSwitch(self.tripped)
        if amount <= 0 or amount > self.limits.max_order_size:
            self.trip(f"malformed order size {amount} "
                      f"(cap {self.limits.max_order_size})")
            raise KillSwitch(self.tripped)
        would_be = position + (amount if side == "buy" else -amount)
        if abs(would_be) > self.limits.max_position:
            raise OrderBlocked(
                f"position would become {would_be:+.2f}, cap "
                f"{self.limits.max_position}")
        if n_open_orders >= self.limits.max_open_orders:
            raise OrderBlocked(f"{n_open_orders} open orders >= cap "
                               f"{self.limits.max_open_orders}")
