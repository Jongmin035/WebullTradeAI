# WebullTradeAI

An autonomous ML trading bot that runs on AWS EC2, trades S&P 500 stocks through the Webull Open API, and retrains itself weekly. It uses an LSTM to rank stocks by expected return, a regime-aware portfolio allocator to decide how aggressively to deploy capital, and a set of safeguards to limit downside risk.

---

## How It Works

Every weekday at 11:00 AM ET (8:00 AM PT) the bot wakes up on EC2, runs inference on all 500 S&P 500 symbols, and rebalances the portfolio. Every Saturday it retrains the LSTM on the latest data and uploads new model artifacts to S3. The EC2 instance shuts itself down after each run to minimize cost.

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

- **Input**: sequences of 20 trading days × 38 features per symbol
- **Architecture**: 2-layer LSTM → hidden size 32 → dropout → linear output head
- **Classifier head** (`clf_prob`): probability the stock outperforms the median S&P 500 stock over the next 5 days (binary cross-entropy loss)
- **Regressor head** (`reg_pred`): expected 5-day forward return as a z-score (MSE loss)
- **Device**: NVIDIA A10G GPU (g5.xlarge) — training takes ~1–3 hours for 500 symbols

### Features (38 per timestep)

| Category | Features |
|---|---|
| Price/Volume | `close`, `volume` |
| Trend | `sma20`, `sma50`, `close_vs_sma20`, `close_vs_sma50`, `sma50_vs_sma200` |
| Momentum | `return_lag1/2/3`, `recovery_slope` (5d), `momentum` (12-1 mo), `zscore20` |
| Oscillators | `rsi`, `macd`, `signal`, `histogram` |
| Volatility | `volatility10`, `price_range`, `atr14_pct` |
| Fear index | `vix` |
| Entry/reversal signals | `gap`, `pct_from_high_20d`, `range_tightness`, `donchian_55_pos`, `obv_zscore` |
| Cross-sectional ranks | `rsi_rank`, `volume_rank`, `momentum_rank`, `zscore20_rank`, `volatility10_rank`, `pct_from_high_20d_rank`, `recovery_slope_rank` |
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

Market regime is detected daily from SPY using two primary signals:

| Signal | Role |
|---|---|
| SMA50 vs SMA200 (±0.5% buffer) | Structural trend direction — primary signal for bull |
| VIX level | Fear modifier — VIX ≥ 30 downgrades bull→sideways; VIX ≥ 40 forces bear |
| ADX14 > 25 + 21-day SPY return | Used only for bear confirmation — avoids calling bear on a short-term dip |

**Labels**: `bull` / `sideways` / `bear`

- **Bull**: SMA50 > SMA200 AND VIX < 30
- **Bear**: (SMA50 < SMA200 AND ADX > 25 AND 21-day return < 0) OR VIX ≥ 40
- **Sideways**: everything else

### Stage 2 — Bucket Allocator

Regime maps to hardcoded portfolio bucket targets:

| Regime | Venture | Safety (SPY) | Hedge (SH / SQQQ) | Cash |
|---|---|---|---|---|
| **Bull** | 60% | 25% | 0% | 15% |
| **Sideways** | 40% | 20% | 0% | 40% |
| **Bear** | 15% | 0% | 30% | 55% |

The goal in bull markets is to track the index: 60% in ML-selected stocks that should outperform, 25% directly in SPY as a neutral filler. In bear, SH and SQQQ provide short exposure that profits from further declines.

A VIX slope parameter in the underlying model shifts venture allocation toward hedge as VIX rises above 20 — this compounds with the VIX circuit breaker in `safeguards.py` (see below).

Safety and hedge ETFs are not subject to the same ranking system as venture stocks. The allocator's `safety_pct` / `hedge_pct` determine how much to hold; the ETFs within each bucket are always held at equal weight.

### Stage 3 — Position Sizing (Kelly Criterion)

Within the venture bucket, each stock's weight is set by half-Kelly:

```
kelly = clf_prob - (1 - clf_prob)   # edge / odds (simplified)
weight_i = kelly_i / sum(kelly)  ×  venture_pct
```

### Rebalancing

Stocks are ranked by Kelly score each day and fall into one of three zones:

| Zone | Criteria | Action |
|---|---|---|
| **Buy zone** | Top 10 by Kelly (clf_prob ≥ 0.60, reg_pred > 0) | Buy/hold at proportional Kelly weight |
| **Hold zone** | Ranks 11–30, currently held | Keep at current weight — no trade triggered |
| **Sell zone** | Rank > 30, or below confidence threshold | Full exit |

The hold zone reduces unnecessary daily turnover: a stock that slips from rank 9 to rank 12 is held rather than sold and potentially repurchased the next day.

A trade is only placed if the weight delta between target and current exceeds **2%** (except full exits, which always execute regardless of size).

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
| EC2 g5.xlarge | GPU instance (NVIDIA A10G, 16 GB VRAM) — runs both the daily bot and weekly retrain |
| ECR `webull-bot` / `webull-retrain` | Docker image registry — GitHub Actions pushes here on every commit |
| S3 `webull-trade-ai` | Model artifacts, training data parquets, bot state, systemd service files |
| EventBridge | Starts EC2 on schedule (weekdays 14:45 UTC, Saturdays 11:55 UTC) |
| systemd `bot.service` | Runs the bot container on weekdays, shuts down EC2 when done |
| systemd `retrain.service` | Runs the retrain container on Saturdays |

**Cost**: ~$13/month (g5.xlarge billed only during runs — ~20 min/day + ~2 hr/week retrain). ECR lifecycle policy keeps the last 3 tagged images per repo and expires untagged images after 1 day.

### Deployment pipeline

```
git push → GitHub Actions
             ├── docker build Dockerfile.bot   → ECR webull-bot:latest
             └── docker build Dockerfile.retrain → ECR webull-retrain:latest

EC2 boot (triggered by EventBridge)
  └── config-pull.service runs startup.sh
        ├── downloads latest service/timer files from S3
        └── docker pull webull-bot:latest + webull-retrain:latest

bot.timer fires
  └── docker run webull-bot:latest  (GIT_SHA env var shows exact commit)
```

No SSH, no manual steps. A code change is live on the next scheduled run.

---

## Project Structure

```
├── docker/
│   ├── Dockerfile.bot      # python:3.11-slim + CPU torch (~2 GB image)
│   └── Dockerfile.retrain  # python:3.11-slim + CUDA torch (~6.5 GB image)
├── .github/workflows/
│   └── deploy.yml          # builds + pushes both images to ECR on push to main
├── requirements.txt        # all Python dependencies except torch
└── src/
    ├── aws/               # EC2 entry points + infrastructure
    │   ├── main.py            # daily trading run
    │   ├── retrain.py         # weekly retrain pipeline
    │   ├── bot.service / bot.timer
    │   ├── retrain.service / retrain.timer
    │   ├── startup.sh         # runs on every boot: pulls service files + Docker images
    │   ├── bootstrap.sh       # first-boot setup: installs Docker + nvidia-container-toolkit
    │   └── deploy.sh          # uploads service/timer/startup files to S3
    ├── core/              # shared library
    │   ├── trader.py          # Webull order execution + Kelly sizing
    │   ├── predict.py         # daily inference engine
    │   ├── safeguards.py      # VIX / drawdown / stop-loss checks
    │   ├── model_store.py     # save/load artifacts (local + S3)
    │   ├── dashboard_logger.py  # S3 state persistence + dashboard upload
    │   └── controls.py        # runtime config (emergency stop, force sell, etc.)
    ├── pipeline/          # data + features
    │   ├── data_pipeline.py   # load parquets + merge yfinance + sentiment
    │   ├── indicators.py      # technical indicator computation
    │   ├── sentiment.py       # Alpaca News + FinBERT sentiment pipeline
    │   ├── backtest.py        # walk-forward evaluation
    │   └── metrics.py         # Kelly criterion, Sharpe, Sortino
    ├── models/            # model definitions
    │   ├── regime_pipeline.py  # SPY regime detection
    │   └── allocator.py        # bucket allocator (venture/safety/hedge/cash)
    └── dashboard/         # reporting
        ├── index.html         # live dashboard (S3 static site)
        ├── history.html       # trade history page (auto-generated)
        └── history.py         # HTML generation from trade_log.csv
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

### Prerequisites

- AWS account with an EC2 g5.xlarge, an S3 bucket, and two ECR repositories (`webull-bot`, `webull-retrain`)
- Webull Open API credentials and an Alpaca API key
- GitHub repository with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` set as Actions secrets

### Deploy

```bash
# 1. Fill in credentials
cp .env.example .env   # add Webull, AWS, and Alpaca keys

# 2. Push to main — GitHub Actions builds and pushes both Docker images to ECR automatically
git push origin main

# 3. Deploy infrastructure files to S3 (service/timer/startup files)
bash src/aws/deploy.sh
```

On the next EC2 boot, `startup.sh` pulls the latest images from ECR and the bot runs as a Docker container. No SSH or manual steps required for subsequent deploys — `git push` is sufficient.
