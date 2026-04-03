"""Portfolio: tracks positions and cash balance."""

from __future__ import annotations

from dataclasses import dataclass, field

from ponyai.broker.fill import Fill


@dataclass
class Portfolio:
    """Simple portfolio tracker.

    Attributes:
        cash: Available cash balance.
        positions: Mapping of symbol → signed quantity (positive = long).
        fills: Chronological list of all executed fills.
    """

    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill) -> None:
        """Update positions and cash from a fill."""
        self.fills.append(fill)
        self.cash -= fill.cost

        current = self.positions.get(fill.symbol, 0.0)
        if fill.side == "buy":
            self.positions[fill.symbol] = current + fill.quantity
        else:
            self.positions[fill.symbol] = current - fill.quantity

    def market_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value at given market prices.

        Args:
            prices: Mapping of symbol → current price.

        Returns:
            Cash plus the mark-to-market value of all positions.
        """
        position_value = sum(
            qty * prices.get(sym, 0.0) for sym, qty in self.positions.items()
        )
        return self.cash + position_value

    @property
    def total_commission(self) -> float:
        """Sum of all commissions paid."""
        return sum(f.commission for f in self.fills)
