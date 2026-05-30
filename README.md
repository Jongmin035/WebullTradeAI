# WebullTradeAI

An autonomous ML trading bot that runs on AWS EC2, trades S&P 500 stocks through the Webull Open API, and retrains itself weekly. It uses an LSTM to rank stocks by expected return, a regime-aware portfolio allocator to decide how aggressively to deploy capital, and a set of safeguards to limit downside risk.

---

## How It Works

Every weekday at 9:35 AM ET the bot wakes up on EC2, runs inference on all 500 S&P 500 symbols, and rebalances the portfolio. Every Saturday it retrains the LSTM on the latest data and uploads new model artifacts to S3. The EC2 instance shuts itself down after each run to minimize cost.

```
Saturday (weekly)                  Weekdays (daily)
─────────────────                  ────────────────
EventBridge → EC2 start            EventBridge → EC2 start
retrain.py                         main.py
  ↓ download data from S3            ↓ restore state from S3
  ↓ update sentiment (Alpaca+FinBERT)↓ fetch latest prices (yfinance)
  ↓ walk-forward LSTM eval           ↓ LSTM inference (500 symbols)
  ↓ train final model on all data    ↓ regime + VIX check
  ↓ optimize allocator params        ↓ compute target weights (Kelly + allocator)
  ↓ upload artifacts to S3           ↓ run safety checks
EC2 auto-shutdown                    ↓ execute rebalancing orders (Webull API)
                                     ↓ upload state to S3
                                   EC2 auto-shutdown
```

---

## Model

### LSTM Architecture

The core model is a dual-head LSTM trained on 20 years of S&P 500 daily OHLCV data (2006–present).

- **Input**: sequences of 20 trading days × 27 features per symbol
- **Architecture**: 2-layer LSTM → hidden size 32 → dropout → linear output head
- **Classifier head** (`clf_prob`): probability the stock outperforms the median S&P 500 stock tomorrow (binary cross-entropy loss)
- **Regressor head** (`reg_pred`): expected next-day return as a z-score (MSE loss)
- **Device**: NVIDIA A10G GPU (g5.xlarge) — training takes ~1–3 hours for 500 symbols

### Features (27 per timestep)

| Category | Features |
|---|---|
| Price/Volume | `close`, `volume` |
| Trend | `sma20`, `sma50`, `close_vs_sma20`, `close_vs_sma50` |
| Momentum | `return_lag1/2/3`, `momentum`, `zscore20` |
| Oscillators | `rsi`, `macd`, `signal`, `histogram` |
| Volatility | `volatility10`, `price_range` |
| Cross-sectional ranks | `rsi_rank`, `volume_rank`, `momentum_rank`, `zscore20_rank`, `volatility10_rank` |
| Sentiment | `sentiment_1d`, `sentiment_3d`, `sentiment_7d` |

Cross-sectional rank features normalize each indicator relative to all 500 stocks on the same day, helping the LSTM learn relative strength rather than absolute levels.

### Training

- **Data**: per-symbol Parquet files stored in S3 (`s3://webull-trade-ai/indicators/`), one file per symbol covering 2006–present
- **Walk-forward evaluation**: the last 12 months are held out for evaluation (one forward pass per month, trained only on past data)
- **Training window cap**: 36 months of data fed to LSTM at a time to avoid OOM on 16 GB GPU RAM
- **Early stopping**: patience = 10 epochs, max 200 epochs
- **Optimizer**: Adam, lr = 1e-3, batch size = 512

### Sentiment Features

News sentiment is computed offline using [FinBERT](https://huggingface.co/ProsusAI/finbert) on headlines fetched from the Alpaca News API. Each symbol gets three rolling sentiment scores: 1-day, 3-day, and 7-day weighted averages of `positive_prob − negative_prob`. Sentiment parquets are stored in S3 and updated weekly during retrain.

---

## Portfolio Allocation

Predictions from the LSTM feed into a two-stage allocation system.

### Stage 1 — Regime Detection

Market regime is detected daily from SPY using four signals:

| Signal | Role |
|---|---|
| SMA50 vs SMA200 (±1% buffer) | Structural trend direction |
| ADX14 > 25 | Confirms a real trend exists (not chop) |
| 21-day SPY rolling return | Short-term momentum confirmation |
| VIX level | Fear modifier — VIX ≥ 30 downgrades bull→sideways; VIX ≥ 40 forces bear |

**Labels**: `bull` / `sideways` / `bear`

### Stage 2 — Bucket Allocator

A 13-parameter walk-forward optimizer (Nelder-Mead) maps `(regime, VIX)` to four portfolio buckets:

| Bucket | Contents | Typical allocation |
|---|---|---|
| **Venture** | Top 15 S&P 500 stocks by Kelly criterion | 30–50% |
| **Safety** | SPY, XLP, XLU (equal-weighted) | 20–40% |
| **Hedge** | GLDM, SH, SQQQ (equal-weighted) | 0–15% |
| **Cash** | Uninvested | 0–60% |

The allocator is contrarian (Buffett-style): it holds more cash in bull markets and deploys aggressively during bear markets / recoveries. A VIX slope parameter shifts venture allocation toward hedge as VIX rises above 20.

### Stage 3 — Position Sizing (Kelly Criterion)

Within the venture bucket, each stock's weight is set by half-Kelly:

```
kelly = clf_prob - (1 - clf_prob)   # edge / odds (simplified)
weight_i = kelly_i / sum(kelly)  ×  venture_pct
```

Positions are capped to the top 15 stocks by Kelly to ensure each position is large enough to clear the 2% rebalance threshold.

### Rebalancing

A trade is only placed if the difference between a stock's target weight and its current portfolio weight exceeds **2%**. This prevents excessive turnover from small price drifts.

---

## Safety Mechanisms

Four independent checks run before every rebalance:

| Check | Trigger | Action |
|---|---|---|
| VIX circuit breaker | VIX ≥ 40 | Sell all positions, halt trading |
| VIX reduction | VIX ≥ 25 | Halve all Kelly weights |
| Drawdown halt | Portfolio down ≥ 15% from peak | Sell all, halt trading |
| ATR trailing stop-loss | Price falls 2×ATR14 below its high-water mark | Sell that position |

The ATR trailing stop adapts to each stock's volatility — a low-volatility stock has a tighter stop than a high-volatility one. High-water marks are persisted to `position_highs.json` and backed up to S3 after every rebalance.

---

## APIs

| API | Purpose |
|---|---|
| **Webull Open API** (official SDK) | Live account positions, order execution, trade calendar |
| **yfinance** | Daily OHLCV for 500 S&P 500 symbols (historical + live inference), VIX |
| **Alpaca News API** | Financial news headlines for FinBERT sentiment scoring |
| **AWS S3** | Persistent storage for model artifacts, training data, and bot state |

### Webull API

The bot uses the official Webull Open API SDK. It calls three main endpoints:

- `get_positions()` — current holdings and market values
- `place_order()` — market orders for rebalancing
- `get_trade_calendar()` — check whether today is a US trading day (used to skip holidays)

All credentials are loaded from environment variables (`.env`) and never hardcoded.

---

## AWS Infrastructure

| Component | Purpose |
|---|---|
| EC2 g5.xlarge | GPU instance for training (NVIDIA A10G, 16 GB VRAM) |
| S3 `webull-trade-ai` | Model artifacts, training data parquets, bot state |
| EventBridge | Starts EC2 on schedule (weekdays 14:30 UTC, Saturdays 11:55 UTC) |
| systemd `bot.service` | Runs `main.py` on weekdays, shuts down EC2 when done |
| systemd `retrain.service` | Runs `retrain.py` on Saturdays, shuts down EC2 when done |

**Cost**: ~$25/month (g5.xlarge billed only during runs — ~10 min/day + ~2 hr/week retrain).

---

## Project Structure

```
src/
├── aws/           # EC2 entry points
│   ├── main.py        # daily trading run
│   ├── retrain.py     # weekly retrain pipeline
│   ├── cleanup.py     # delete stale model artifacts
│   ├── healthcheck.py # exits 0/1 based on last run status
│   ├── bot.service / bot.timer
│   └── retrain.service / retrain.timer
├── core/          # shared library
│   ├── trader.py      # Webull order execution + Kelly sizing
│   ├── predict.py     # daily inference engine
│   ├── safeguards.py  # VIX / drawdown / stop-loss checks
│   ├── model_store.py # save/load artifacts (local + S3)
│   ├── dashboard_logger.py  # S3 state persistence + dashboard upload
│   └── controls.py    # runtime config (emergency stop, force sell, etc.)
├── pipeline/      # data + features
│   ├── data_pipeline.py  # load parquets + merge yfinance + sentiment
│   ├── indicators.py     # technical indicator computation
│   ├── sentiment.py      # Alpaca News + FinBERT sentiment pipeline
│   ├── backtest.py       # walk-forward evaluation
│   └── metrics.py        # Kelly criterion, Sharpe, Sortino
├── models/        # model definitions
│   ├── regime_pipeline.py  # SPY regime detection
│   └── allocator.py        # bucket allocator (venture/safety/hedge/cash)
├── dashboard/     # reporting
│   ├── index.html    # live dashboard (S3 static site)
│   ├── history.html  # trade history page (auto-generated)
│   └── history.py    # HTML generation from trade_log.csv
└── state/         # local bot state (backed up to S3)
    ├── trade_log.csv
    ├── balance_history.csv
    ├── position_highs.json
    ├── peak_portfolio_value.json
    └── commands.json       # runtime controls (max_capital, emergency_stop, etc.)
```

---

## Configuration

Runtime behavior is controlled via `src/state/commands.json` (synced to S3 — changes are picked up the next morning without redeploying):

```json
{
  "max_capital": 8000.0,
  "manual_symbols": [],
  "force_sell": [],
  "emergency_stop": false
}
```

| Field | Effect |
|---|---|
| `max_capital` | Maximum dollars the bot will deploy |
| `manual_symbols` | Symbols the bot will not touch (manually managed positions) |
| `force_sell` | Symbols to sell at next rebalance regardless of model signal |
| `emergency_stop` | If true, bot sells everything and halts |

---

## Setup

1. Clone the repo and install dependencies into a virtualenv
2. Copy `.env.example` to `.env` and fill in your Webull, AWS, and Alpaca credentials
3. Upload historical S&P 500 parquets to S3 (see `src/pipeline/data_pipeline.py`)
4. Run `src/aws/setup.sh` on your EC2 instance to install the bot
5. Set up EventBridge rules to start EC2 on the weekday and Saturday schedules

See `src/aws/setup.sh` for the full EC2 setup script.
