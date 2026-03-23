"""Core strategy abstractions for the PonyAI quantitative platform."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SignalType(Enum):
    """Trading signal type."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Signal:
    """A trading signal produced by a strategy."""

    symbol: str
    signal_type: SignalType
    strength: float = 1.0  # 0.0 – 1.0, relative confidence
    metadata: Dict = field(default_factory=dict)


@dataclass
class Order:
    """A trade order generated from a signal."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # required for LIMIT orders
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Represents a current holding in a symbol."""

    symbol: str
    quantity: float
    avg_cost: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost


class Strategy(abc.ABC):
    """Abstract base class for all quantitative strategies.

    Sub-classes must implement :meth:`generate_signals`.  The engine will call
    :meth:`on_bar` for every bar of historical (or live) data, collect signals
    and translate them to orders via :meth:`signals_to_orders`.

    Parameters
    ----------
    name:
        Human-readable strategy identifier.
    params:
        Hyper-parameters that control the strategy behaviour.  Storing them
        here (rather than in instance variables) ensures every run is
        fully described by ``(strategy_class, params)``.
    """

    def __init__(self, name: str, params: Optional[Dict] = None) -> None:
        self.name = name
        self.params: Dict = params or {}
        self._positions: Dict[str, Position] = {}
        self._bars_seen: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def generate_signals(self, bar: Dict) -> List[Signal]:
        """Return a list of :class:`Signal` objects for the given *bar*.

        Parameters
        ----------
        bar:
            A dictionary with at least ``symbol``, ``open``, ``high``,
            ``low``, ``close`` and ``volume`` keys.
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the first bar is processed."""

    def on_end(self) -> None:
        """Called once after the last bar has been processed."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def on_bar(self, bar: Dict) -> List[Signal]:
        """Process a single bar and return signals."""
        self._bars_seen += 1
        return self.generate_signals(bar)

    def signals_to_orders(
        self,
        signals: List[Signal],
        bar: Dict,
        capital: float,
        max_position_pct: float = 0.10,
    ) -> List[Order]:
        """Convert signals to concrete orders using a simple equal-weight sizer.

        Parameters
        ----------
        signals:
            Signals produced by :meth:`generate_signals`.
        bar:
            The current market bar used for market-order price reference.
        capital:
            Available cash capital.
        max_position_pct:
            Maximum fraction of *capital* to allocate per position.
        """
        orders: List[Order] = []
        price = bar.get("close", 0.0)
        if price <= 0:
            return orders

        for sig in signals:
            if sig.signal_type == SignalType.BUY:
                alloc = capital * max_position_pct * sig.strength
                qty = alloc / price
                if qty > 0:
                    orders.append(
                        Order(
                            symbol=sig.symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=qty,
                        )
                    )
            elif sig.signal_type == SignalType.SELL:
                pos = self._positions.get(sig.symbol)
                if pos and pos.quantity > 0:
                    orders.append(
                        Order(
                            symbol=sig.symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=pos.quantity,
                        )
                    )
        return orders

    def update_position(self, order: Order, fill_price: float) -> None:
        """Update internal position after an order is filled."""
        symbol = order.symbol
        pos = self._positions.get(symbol, Position(symbol=symbol, quantity=0.0, avg_cost=0.0))

        if order.side == OrderSide.BUY:
            total_cost = pos.quantity * pos.avg_cost + order.quantity * fill_price
            pos.quantity += order.quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0.0
        elif order.side == OrderSide.SELL:
            pos.quantity = max(0.0, pos.quantity - order.quantity)
            if pos.quantity == 0:
                pos.avg_cost = 0.0

        self._positions[symbol] = pos

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)
