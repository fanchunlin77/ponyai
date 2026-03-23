# PonyAI — 可复现、可回测、可灰度上线的量化平台

PonyAI is a lightweight, event-driven quantitative trading platform built around three first-class properties:

| Property | Chinese | What it means |
|---|---|---|
| **Reproducible** | 可复现 | Identical `(feed_version, strategy_version, seed)` always produces identical results |
| **Backtestable** | 可回测 | Full event-driven backtesting engine with performance metrics |
| **Canary-deployable** | 可灰度上线 | Gradually shift capital from a stable to a candidate strategy via `CanaryRouter` |

---

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from ponyai.data import DataFeed
from ponyai.backtest import BacktestEngine
from ponyai.strategy.examples import MovingAverageCrossover

# 1. Create a deterministic synthetic feed (pin seed for reproducibility)
feed = DataFeed.synthetic(symbol="SYN", n_bars=252, seed=42)
print(f"Feed version: {feed.version}")   # content-hash → reproducible

# 2. Backtest a strategy
strategy = MovingAverageCrossover(fast_period=10, slow_period=30, seed=0)
engine = BacktestEngine(initial_cash=1_000_000)
result = engine.run(strategy, feed)
print(result)
```

---

## Architecture

```
ponyai/
├── data/          # DataFeed — versioned OHLCV bars, synthetic generator
├── strategy/      # Strategy ABC + example strategies (BuyAndHold, MA Crossover)
├── broker/        # PaperBroker — fills at close ± slippage; Portfolio
├── backtest/      # BacktestEngine — event loop, BacktestResult
├── risk/          # compute_metrics — Sharpe, max drawdown, Calmar, win rate
└── deploy/        # CanaryRouter — gray deployment / A–B capital split
```

---

## Reproducibility (可复现)

Every `DataFeed` carries a **16-character SHA-256 content hash** (`feed.version`).
Strategies declare a `version` string and accept a `seed` for their internal RNG.
Fixing `(feed.version, strategy.version, seed)` guarantees byte-identical equity
curves across runs, machines, and time.

```python
r1 = engine.run(strategy, feed)
r2 = engine.run(strategy, feed)   # reset() called automatically
assert list(r1.equity_curve) == list(r2.equity_curve)  # always True
```

---

## Backtesting (可回测)

`BacktestEngine` drives an event loop over bars, calls `strategy.on_bar(bar, portfolio)`,
and forwards returned `Order` objects to `PaperBroker` for simulated execution.

```python
from ponyai.backtest import BacktestEngine
from ponyai.strategy.examples import BuyAndHold

result = engine.run(BuyAndHold(quantity=100), feed)
print(result.metrics)
# --- Performance Metrics ---
#   Total Return        : +28.34%
#   Sharpe Ratio        : 1.247
#   Max Drawdown        : -14.21%
#   ...
```

Writing a custom strategy requires implementing one method:

```python
from ponyai.strategy import Strategy, Order, OrderSide

class MyStrategy(Strategy):
    def on_bar(self, bar, portfolio):
        if bar.close > 120 and portfolio.position(bar.symbol) == 0:
            return [Order(bar.symbol, OrderSide.BUY, quantity=50)]
        return []
```

---

## Canary / Gray Deployment (可灰度上线)

`CanaryRouter` runs two strategies in parallel, splitting capital by
`candidate_weight` (0 → 100 % stable, 1 → 100 % candidate).

```python
from ponyai.deploy import CanaryRouter
from ponyai.strategy.examples import BuyAndHold, MovingAverageCrossover

router = CanaryRouter(
    stable=BuyAndHold(quantity=100),
    candidate=MovingAverageCrossover(fast_period=5, slow_period=20),
    candidate_weight=0.1,        # 10 % allocated to new strategy
    initial_cash=1_000_000,
)
result = router.run(feed)
print(result)

# Promote to 50/50 once the candidate looks good
router.candidate_weight = 0.5
result2 = router.run(feed)
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 46 tests should pass.
