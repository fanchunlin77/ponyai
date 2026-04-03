"""Fill: represents a completed trade execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Fill:
    """Record of a single executed trade.

    Attributes:
        symbol: Ticker or instrument identifier.
        side: ``"buy"`` or ``"sell"``.
        quantity: Number of units traded.
        price: Execution price per unit.
        timestamp: When the fill occurred.
        commission: Transaction cost.
    """

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0

    @property
    def cost(self) -> float:
        """Total cash outflow (positive) or inflow (negative) of this fill."""
        sign = 1.0 if self.side == "buy" else -1.0
        return sign * self.quantity * self.price + self.commission

    @property
    def pnl(self) -> float:
        """Realised PnL is meaningful only when netting against a position.

        For a single fill the *signed notional* is returned so that
        ``sum(fill.pnl for fill in fills)`` gives the net cash flow.
        """
        sign = -1.0 if self.side == "buy" else 1.0
        return sign * self.quantity * self.price - self.commission
