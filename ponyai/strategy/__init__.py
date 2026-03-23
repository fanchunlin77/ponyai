"""Strategy base class and order types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """A trading order emitted by a strategy."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement :meth:`on_bar` which is called for every bar by
    the backtesting engine.  Orders are returned as a list from that method.

    Parameters
    ----------
    name:
        Human-readable strategy name.
    version:
        Strategy version string used for reproducibility and canary routing.
    params:
        Arbitrary hyper-parameter dict (must be JSON-serialisable for
        reproducibility logs).
    seed:
        Random seed; fix this to guarantee reproducible order generation when
        the strategy uses randomness.
    """

    def __init__(
        self,
        name: str = "strategy",
        version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.version = version
        self.params: Dict[str, Any] = params or {}
        self.seed = seed
        self._rng = __import__("numpy").random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Reset mutable state; called before every backtest run."""
        self._rng = __import__("numpy").random.default_rng(self.seed)

    @abstractmethod
    def on_bar(self, bar: Any, portfolio: Any) -> List[Order]:
        """Process a new price bar and return a (possibly empty) list of orders.

        Parameters
        ----------
        bar:
            The current :class:`~ponyai.data.Bar` object.
        portfolio:
            The current :class:`~ponyai.broker.Portfolio` snapshot.

        Returns
        -------
        List[Order]
            Orders to submit to the broker.
        """

    def on_start(self) -> None:
        """Called once before the backtest loop begins."""

    def on_end(self) -> None:
        """Called once after the backtest loop ends."""
