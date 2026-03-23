"""Built-in example strategies shipped with PonyAI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ponyai.strategy import Order, OrderSide, Strategy


class BuyAndHold(Strategy):
    """Buy a fixed quantity on the first bar and hold forever."""

    def __init__(self, quantity: float = 100.0, **kwargs: Any) -> None:
        super().__init__(name="BuyAndHold", **kwargs)
        self.quantity = quantity
        self._bought = False

    def reset(self) -> None:
        super().reset()
        self._bought = False

    def on_bar(self, bar: Any, portfolio: Any) -> List[Order]:
        if not self._bought:
            self._bought = True
            return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=self.quantity)]
        return []


class MovingAverageCrossover(Strategy):
    """Classic dual moving-average crossover strategy.

    Parameters
    ----------
    fast_period:
        Lookback for the fast (short) moving average.
    slow_period:
        Lookback for the slow (long) moving average.
    quantity:
        Number of shares per trade.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="MovingAverageCrossover",
            params={"fast_period": fast_period, "slow_period": slow_period},
            **kwargs,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        self._prices: List[float] = []
        self._position = 0.0

    def reset(self) -> None:
        super().reset()
        self._prices = []
        self._position = 0.0

    def on_bar(self, bar: Any, portfolio: Any) -> List[Order]:
        self._prices.append(bar.close)
        if len(self._prices) < self.slow_period:
            return []

        fast_ma = sum(self._prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self._prices[-self.slow_period :]) / self.slow_period

        orders: List[Order] = []
        held = portfolio.position(bar.symbol)

        if fast_ma > slow_ma and held == 0:
            orders.append(Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=self.quantity))
            self._position = self.quantity
        elif fast_ma < slow_ma and held > 0:
            orders.append(Order(symbol=bar.symbol, side=OrderSide.SELL, quantity=held))
            self._position = 0.0

        return orders
