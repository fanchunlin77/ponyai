"""Strategy base – common interface for all trading strategies.

Every strategy must subclass :class:`BaseStrategy` and implement
:meth:`on_bar`.  The strategy receives bars one at a time from the
back-test engine (or live runner) and emits :class:`Signal` objects.

A strategy carries a ``params`` dict that is part of its identity for
reproducibility – the same (class, params, data) triple must always
produce the same signals.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ponyai.data.feed import BarData


class SignalDirection(enum.Enum):
    """Direction of a trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"  # close / stay-out


@dataclass
class Signal:
    """A trading signal emitted by a strategy.

    Attributes
    ----------
    symbol:
        The instrument ticker this signal refers to.
    direction:
        :class:`SignalDirection` – BUY, SELL, or FLAT.
    size:
        Desired position size (number of shares/contracts).  A value of
        ``0`` combined with ``FLAT`` means "close the position".
    price:
        Suggested limit price, or ``None`` for market order.
    meta:
        Arbitrary strategy-specific metadata for audit / debugging.
    """

    symbol: str
    direction: SignalDirection
    size: float = 1.0
    price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all ponyai strategies.

    Parameters
    ----------
    params:
        Hyper-parameters that control strategy behaviour.  These are
        stored verbatim and included in :meth:`describe` so that every
        run is fully reproducible.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    # ------------------------------------------------------------------
    # Lifecycle hooks (override as needed)
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the first bar is delivered."""

    def on_stop(self) -> None:
        """Called once after the last bar has been processed."""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_bar(self, bar: BarData) -> list[Signal]:
        """Process a single bar and return zero or more signals.

        Parameters
        ----------
        bar:
            The latest OHLCV bar.

        Returns
        -------
        list[Signal]
            Signals to act on.  Return an empty list when no action is
            required this bar.
        """

    # ------------------------------------------------------------------
    # Identity / introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable strategy name (class name by default)."""
        return type(self).__name__

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this strategy instance.

        Used to tag back-test results for reproducibility (可复现).
        """
        return {"strategy": self.name, "params": dict(self.params)}
