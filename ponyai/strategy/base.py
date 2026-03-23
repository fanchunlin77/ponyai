"""Strategy abstractions: Order, Portfolio, and the Strategy base class.

Reproducibility is wired in at construction time — each :class:`Strategy`
accepts an optional ``seed`` parameter.  When a seed is provided the Python
and NumPy random-number generators are both initialised with that seed so that
any stochastic logic inside the strategy is fully deterministic.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import numpy.random as npr

from ponyai.data.feed import Bar


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """A market order request produced by a :class:`Strategy`.

    Parameters
    ----------
    symbol:
        Ticker / instrument identifier.
    side:
        :data:`OrderSide.BUY` or :data:`OrderSide.SELL`.
    quantity:
        Number of shares / contracts (must be positive).
    """

    symbol: str
    side: OrderSide
    quantity: float

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")


@dataclass
class Position:
    """Tracks an open position in a single instrument."""

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value — updated externally with the latest price."""
        return self.quantity * self.avg_cost

    def apply_fill(self, side: OrderSide, quantity: float, price: float) -> None:
        """Update the position after an order fill."""
        if side == OrderSide.BUY:
            total_cost = self.avg_cost * self.quantity + price * quantity
            self.quantity += quantity
            self.avg_cost = total_cost / self.quantity if self.quantity else 0.0
        else:
            self.quantity -= quantity
            if self.quantity < 0:
                raise ValueError("Short selling is not supported in this version.")
            if self.quantity == 0:
                self.avg_cost = 0.0


class Portfolio:
    """Tracks cash and open positions for a strategy.

    Parameters
    ----------
    initial_capital:
        Starting cash balance.
    commission_rate:
        Fraction of trade value charged as commission (e.g. ``0.001`` = 0.1 %).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.commission_rate = commission_rate
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[float] = []

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Trade execution (called by the backtest engine)
    # ------------------------------------------------------------------

    def fill_order(self, order: Order, price: float) -> float:
        """Apply an order fill at *price* and return the commission paid."""
        trade_value = order.quantity * price
        commission = trade_value * self.commission_rate

        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need {total_cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= total_cost
        else:
            self.cash += trade_value - commission

        position = self.get_position(order.symbol)
        position.apply_fill(order.side, order.quantity, price)
        return commission

    # ------------------------------------------------------------------
    # Equity snapshot
    # ------------------------------------------------------------------

    def record_equity(self, prices: Dict[str, float]) -> float:
        """Compute and record total equity at current *prices*."""
        holdings_value = sum(
            self.get_position(sym).quantity * price for sym, price in prices.items()
        )
        equity = self.cash + holdings_value
        self._equity_curve.append(equity)
        return equity

    @property
    def equity_curve(self) -> List[float]:
        return list(self._equity_curve)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Portfolio(cash={self.cash:.2f}, positions={list(self._positions.keys())})"


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Parameters
    ----------
    seed:
        Optional integer seed for per-instance random-number generators.
        When provided, ``self.rng`` (:class:`random.Random`) and
        ``self.np_rng`` (:class:`numpy.random.Generator`) are both initialised
        with that seed so that any stochastic logic inside the strategy is fully
        deterministic regardless of external global state.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        # Per-instance generators — do NOT touch the global random/np.random
        # state so that multiple Strategy instances are fully independent.
        self.rng: random.Random = random.Random(seed)
        self.np_rng: npr.Generator = npr.default_rng(seed)

    @abstractmethod
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        """React to a new :class:`~ponyai.data.feed.Bar` and return orders.

        Parameters
        ----------
        bar:
            The latest OHLCV candlestick.
        portfolio:
            The current portfolio state.

        Returns
        -------
        list of :class:`Order`
            Zero or more market orders to execute.  Return an empty list when
            the strategy has no action to take.
        """
