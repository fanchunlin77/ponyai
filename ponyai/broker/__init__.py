"""Paper-trading broker — portfolio and order execution simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ponyai.strategy import Order, OrderSide, OrderType


@dataclass
class Position:
    """Current holding in a single symbol."""

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost


@dataclass
class Fill:
    """Executed trade record."""

    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: pd.Timestamp
    commission: float = 0.0
    cost_basis: float = 0.0  # avg cost per share at the time of a SELL fill

    @property
    def pnl(self) -> float:
        """Realised P&L for a SELL fill (0 for BUY fills)."""
        if self.side == OrderSide.SELL:
            return (self.price - self.cost_basis) * self.quantity - self.commission
        return 0.0


class Portfolio:
    """Maintains cash, positions and trade history."""

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash: float = initial_cash
        self._positions: Dict[str, Position] = {}
        self._fills: List[Fill] = []

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def fills(self) -> List[Fill]:
        return list(self._fills)

    def equity(self, prices: Dict[str, float]) -> float:
        """Total portfolio value at current *prices*."""
        pos_value = sum(
            pos.quantity * prices.get(sym, pos.avg_cost)
            for sym, pos in self._positions.items()
        )
        return self.cash + pos_value

    def position(self, symbol: str) -> float:
        """Current quantity held for *symbol* (0 if none)."""
        return self._positions.get(symbol, Position(symbol)).quantity

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions after a fill."""
        if fill.symbol not in self._positions:
            self._positions[fill.symbol] = Position(fill.symbol)
        pos = self._positions[fill.symbol]
        cost = fill.price * fill.quantity + fill.commission
        if fill.side == OrderSide.BUY:
            new_qty = pos.quantity + fill.quantity
            if new_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill.price * fill.quantity) / new_qty
            pos.quantity = new_qty
            self.cash -= cost
        else:
            pos.quantity -= fill.quantity
            self.cash += fill.price * fill.quantity - fill.commission
            if pos.quantity <= 0:
                del self._positions[fill.symbol]
        self._fills.append(fill)


class PaperBroker:
    """Simulated broker that fills orders at bar close price.

    Parameters
    ----------
    commission_rate:
        Fraction of trade value charged as commission (e.g. 0.001 = 0.1%).
    slippage_rate:
        Additional fraction of price applied as slippage (e.g. 0.0005).
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def execute(
        self,
        order: Order,
        bar: "ponyai.data.Bar",  # type: ignore[name-defined]  # noqa: F821
        portfolio: Portfolio,
    ) -> Optional[Fill]:
        """Attempt to execute *order* against *bar*; return Fill or None."""
        if order.symbol != bar.symbol:
            return None
        if order.quantity <= 0:
            return None
        base_price = bar.close
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            if order.side == OrderSide.BUY and base_price > order.limit_price:
                return None
            if order.side == OrderSide.SELL and base_price < order.limit_price:
                return None
            base_price = order.limit_price
        slippage = base_price * self.slippage_rate
        fill_price = (
            base_price + slippage
            if order.side == OrderSide.BUY
            else base_price - slippage
        )
        commission = fill_price * order.quantity * self.commission_rate
        # Clamp buy quantity to affordable amount
        if order.side == OrderSide.BUY:
            max_qty = portfolio.cash / (fill_price * (1 + self.commission_rate))
            quantity = min(order.quantity, max_qty)
            if quantity <= 0:
                return None
            cost_basis = 0.0
        else:
            held = portfolio.position(order.symbol)
            quantity = min(order.quantity, held)
            if quantity <= 0:
                return None
            # Capture avg cost before applying the fill for accurate PnL
            cost_basis = portfolio._positions[order.symbol].avg_cost
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            timestamp=bar.timestamp,
            commission=commission,
            cost_basis=cost_basis,
        )
        portfolio.apply_fill(fill)
        return fill
