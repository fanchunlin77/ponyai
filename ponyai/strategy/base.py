"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """Represents an order emitted by a strategy."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price must be set for LIMIT orders")


@dataclass
class Position:
    """Tracks a single symbol position."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0

    def update(self, side: OrderSide, quantity: float, fill_price: float) -> None:
        """Update position after a fill."""
        if side == OrderSide.BUY:
            total_cost = self.avg_price * self.quantity + fill_price * quantity
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity else 0.0
        else:
            self.quantity -= quantity
            if self.quantity < 0:
                self.quantity = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Sub-classes must implement :meth:`on_bar`.

    Attributes
    ----------
    name:
        Human-readable strategy identifier.
    params:
        Strategy hyper-parameters.  Override in sub-class or pass at
        construction time.
    """

    name: str = "BaseStrategy"

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params: dict = params or {}
        self._orders: List[Order] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_bar(self, bar: "Bar", context: "StrategyContext") -> None:  # type: ignore[name-defined]
        """Called on every new bar.

        Implementations should call ``context.submit_order(...)`` to place
        orders.
        """

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def on_start(self, context: "StrategyContext") -> None:  # type: ignore[name-defined]
        """Called once before the first bar."""

    def on_end(self, context: "StrategyContext") -> None:  # type: ignore[name-defined]
        """Called once after the last bar."""


@dataclass
class StrategyContext:
    """Execution context passed to strategy callbacks.

    The engine populates this object before each :meth:`on_bar` call.
    """

    current_bar: Optional["Bar"] = None  # type: ignore[name-defined]
    cash: float = 0.0
    positions: dict = field(default_factory=dict)  # symbol -> Position
    pending_orders: List[Order] = field(default_factory=list)

    def submit_order(self, order: Order) -> None:
        """Queue an order for execution by the engine."""
        self.pending_orders.append(order)

    def get_position(self, symbol: str) -> Position:
        """Return the :class:`Position` for *symbol* (creates if absent)."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
