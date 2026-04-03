"""CanaryRouter: gradual roll-out of new strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from ponyai.strategy.base import Signal, Strategy


@dataclass
class CanaryRouter:
    """Route a fraction of traffic to a *canary* strategy.

    This enables safe, gradual deployment of new strategies alongside an
    existing *baseline* strategy.

    Args:
        baseline: The current production strategy.
        canary: The new strategy being tested.
        canary_weight: Fraction of bars routed to the canary (0.0 to 1.0).
        seed: Random seed for reproducibility.
    """

    baseline: Strategy
    canary: Strategy
    canary_weight: float = 0.1
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def route(self, bar: pd.Series) -> list[Signal]:
        """Decide which strategy processes this bar.

        Returns signals from either the canary or the baseline strategy
        depending on the random draw against ``canary_weight``.
        """
        if self._rng.random() < self.canary_weight:
            return self.canary.on_bar(bar)
        return self.baseline.on_bar(bar)

    def set_canary_weight(self, weight: float) -> None:
        """Adjust canary traffic share (0.0 to 1.0)."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.canary_weight = weight
