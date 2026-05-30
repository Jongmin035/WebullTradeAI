"""
Sentiment pipeline: Alpaca News + FinBERT.

Fetches news for stock symbols from Alpaca Markets, scores each article with
FinBERT (ProsusAI/finbert), and aggregates to daily sentiment scores per symbol.

Storage:  src/data/sentiment/{SYMBOL}.parquet  — columns: date, symbol, sentiment_score
Rolling features (sentiment_1d/3d/7d) are computed at load time, not stored.

CLI:
  python sentiment.py --build                      # full build, all S&P 500 (from 2017)
  python sentiment.py --build --symbols AAPL XOM   # specific symbols
  python sentiment.py --update                     # refresh last 30 days
  python sentiment.py --upload                     # upload parquets to S3
  python sentiment.py --download                   # download parquets from S3

Environment variables (add to .env):
  ALPACA_API_KEY
  ALPACA_API_SECRET
  AWS_S3_BUCKET  (already used by data pipeline)

Note: Alpaca News has reliable coverage from ~2017. Dates before that will be 0.0 (neutral).
"""

import os
import sys
import time
import argparse
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_here = os.path.dirname(os.path.abspath(__file__))

SENTIMENT_DIR = os.path.join(_here, "..", "data", "sentiment")
ALPACA_URL    = "https://data.alpaca.markets/v1beta1/news"
ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET", "")

SYMBOL_BATCH = 50    # symbols per Alpaca request
PAGE_LIMIT   = 50    # Alpaca max items per page
RATE_SLEEP   = 0.35  # seconds between paginated requests (~170/min, under 200 limit)


# --- Alpaca fetch ---

def _headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def fetch_news_batch(symbols, start_date, end_date):
    """
    Fetch all news mentioning any symbol in `symbols` between start_date and end_date.
    Returns list of {date: str, symbol: str, text: str}.
    """
    symbol_set = set(symbols)
    articles   = []
    page_token = None
    page_num   = 0

    while True:
        params = {
            "symbols": ",".join(symbols),
            "start":   start_date,
            "end":     end_date,
            "limit":   PAGE_LIMIT,
            "sort":    "asc",
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(ALPACA_URL, headers=_headers(), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("news", []):
            text = f"{item.get('headline', '')} {item.get('summary', '')}".strip()
            if not text:
                continue
            date_str = item["created_at"][:10]  # "YYYY-MM-DD"
            for sym in item.get("symbols", []):
                if sym in symbol_set:
                    articles.append({"date": date_str, "symbol": sym, "text": text})

        page_num  += 1
        page_token = data.get("next_page_token")
        if not page_token:
            break
        if page_num % 10 == 0:
            print(f"    page {page_num}, {len(articles)} articles so far...", flush=True)
        time.sleep(RATE_SLEEP)

    return articles


# --- FinBERT scoring ---

def load_finbert():
    """Load FinBERT from HuggingFace. Returns (model, tokenizer, device)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name      = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModelForSequenceClassification.from_pretrained(name)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device).eval()
    print(f"FinBERT loaded on {device}")
    return model, tokenizer, device


def score_texts(texts, model, tokenizer, device, batch_size=64):
    """
    Score texts with FinBERT. Returns float array of shape (n,).
    Score = positive_prob - negative_prob  (range roughly -1 to 1).
    FinBERT label order: positive=0, negative=1, neutral=2.
    """
    import torch

    scores = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())

    return np.array(scores)


# --- Aggregation ---

def _aggregate_daily(articles_df, symbol, trading_dates):
    """
    Mean FinBERT score per trading day. Days with no news get 0.0 (neutral).
    Returns DataFrame: date, symbol, sentiment_score.
    """
    base = pd.DataFrame({"date": pd.to_datetime(trading_dates)})

    if articles_df.empty:
        base["symbol"]          = symbol
        base["sentiment_score"] = 0.0
        return base[["date", "symbol", "sentiment_score"]]

    daily = (
        articles_df
        .assign(date=pd.to_datetime(articles_df["date"]))
        .groupby("date")["score"].mean()
        .reset_index()
        .rename(columns={"score": "sentiment_score"})
    )
    out = base.merge(daily, on="date", how="left")
    out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
    out["symbol"]          = symbol
    return out[["date", "symbol", "sentiment_score"]]


# --- I/O ---

def _path(symbol):
    return os.path.join(SENTIMENT_DIR, f"{symbol}.parquet")


def load_sentiment(symbol):
    """
    Load sentiment for one symbol with rolling features computed on the fly.
    Returns DataFrame: date, symbol, sentiment_1d, sentiment_3d, sentiment_7d.
    Returns None if file is missing.
    """
    path = _path(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["sentiment_1d"] = df["sentiment_score"]
    df["sentiment_3d"] = df["sentiment_score"].rolling(3, min_periods=1).mean()
    df["sentiment_7d"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
    return df[["date", "symbol", "sentiment_1d", "sentiment_3d", "sentiment_7d"]]


# --- Build / update ---

def _trading_dates(start_year, end_year):
    """
    Get market-open dates. Uses SPY indicator parquet if available,
    otherwise falls back to pandas business days (close enough for aggregation).
    """
    sys.path.insert(0, _here)
    from data_pipeline import load_symbol
    spy = load_symbol("SPY")
    if spy is not None:
        mask = spy["date"].dt.year.between(start_year, end_year)
        return spy.loc[mask, "date"].tolist()

    # Fallback: US business days (excludes weekends, not holidays — acceptable for aggregation)
    dates = pd.bdate_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
    return pd.Series(dates).tolist()


def build_all(symbols, start_year=2017, end_year=None):
    """
    Full sentiment build. Fetches news from Alpaca and scores with FinBERT.
    Saves one parquet per symbol to SENTIMENT_DIR.
    Coverage before 2017 is sparse on Alpaca — those dates default to 0.0.
    """
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise EnvironmentError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")

    end_year = end_year or pd.Timestamp.today().year
    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    model, tokenizer, device = load_finbert()
    dates = _trading_dates(start_year, end_year)

    # Generate monthly chunks to keep pagination manageable
    months = pd.period_range(start=f"{start_year}-01", end=f"{end_year}-12", freq="M")

    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        print(f"\nSymbols {i+1}–{min(i+SYMBOL_BATCH, len(symbols))} / {len(symbols)}", flush=True)

        all_articles = []
        for month in months:
            start_str = month.start_time.strftime("%Y-%m-%d")
            end_str   = month.end_time.strftime("%Y-%m-%d")
            print(f"  {month} fetching...", end=" ", flush=True)
            try:
                raw = fetch_news_batch(batch, start_str, end_str)
            except Exception as e:
                print(f"error: {e}", flush=True)
                raw = []
            print(f"{len(raw)} articles", flush=True)
            all_articles.extend(raw)
            time.sleep(RATE_SLEEP)

        if all_articles:
            art_df          = pd.DataFrame(all_articles)
            print(f"  Scoring {len(art_df)} articles with FinBERT...", flush=True)
            art_df["score"] = score_texts(art_df["text"].tolist(), model, tokenizer, device)
        else:
            art_df = pd.DataFrame(columns=["date", "symbol", "text", "score"])

        for sym in batch:
            sym_art = art_df[art_df["symbol"] == sym] if not art_df.empty else art_df
            daily   = _aggregate_daily(sym_art, sym, dates)
            daily.to_parquet(_path(sym), index=False)
            n = (daily["sentiment_score"] != 0).sum()
            print(f"  {sym}: saved ({n} days with news)", flush=True)


def update_all(symbols, days=30):
    """Refresh the last `days` days of sentiment for all symbols."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise EnvironmentError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")

    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    model, tokenizer, device = load_finbert()

    today     = pd.Timestamp.today()
    start_str = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end_str   = today.strftime("%Y-%m-%d")

    sys.path.insert(0, _here)
    from data_pipeline import load_symbol
    spy    = load_symbol("SPY")
    recent = spy["date"][spy["date"] >= start_str].tolist() if spy is not None else []

    cutoff = pd.Timestamp(start_str)

    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        print(f"Updating sentiment: symbols {i+1}–{min(i+SYMBOL_BATCH, len(symbols))} / {len(symbols)}")

        try:
            raw = fetch_news_batch(batch, start_str, end_str)
        except Exception as e:
            print(f"  Fetch error: {e} — skipping batch.")
            continue

        if raw:
            art_df          = pd.DataFrame(raw)
            art_df["score"] = score_texts(art_df["text"].tolist(), model, tokenizer, device)
        else:
            art_df = pd.DataFrame(columns=["date", "symbol", "text", "score"])

        for sym in batch:
            sym_art  = art_df[art_df["symbol"] == sym] if not art_df.empty else art_df
            new_rows = _aggregate_daily(sym_art, sym, recent)
            path     = _path(sym)

            if os.path.exists(path):
                old = pd.read_parquet(path)
                old["date"] = pd.to_datetime(old["date"])
                merged = (
                    pd.concat([old[old["date"] < cutoff], new_rows])
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                merged.to_parquet(path, index=False)
            else:
                new_rows.to_parquet(path, index=False)

        time.sleep(RATE_SLEEP)


# --- S3 ---

def upload_to_s3(symbols=None):
    """Upload sentiment parquets to s3://{AWS_S3_BUCKET}/data/sentiment/."""
    import boto3
    bucket = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")
    s3     = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

    if symbols:
        parquets = [f"{sym}.parquet" for sym in symbols if os.path.exists(_path(sym))]
    else:
        parquets = [f for f in os.listdir(SENTIMENT_DIR) if f.endswith(".parquet")]

    print(f"Uploading {len(parquets)} sentiment files to S3...")
    for fname in parquets:
        s3.upload_file(os.path.join(SENTIMENT_DIR, fname), bucket, f"data/sentiment/{fname}")
    print("Upload complete.")


def download_from_s3():
    """Download sentiment parquets from s3://{AWS_S3_BUCKET}/data/sentiment/."""
    import boto3
    bucket = os.getenv("AWS_S3_BUCKET", "webull-trade-ai")
    s3     = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix="data/sentiment/")
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    print(f"Downloading {len(keys)} sentiment files from S3...")
    for key in keys:
        local = os.path.join(SENTIMENT_DIR, os.path.basename(key))
        if not os.path.exists(local):
            s3.download_file(bucket, key, local)
    print("Download complete.")


# --- CLI ---

def main():
    sys.path.insert(0, _here)
    from data_pipeline import fetch_sp500_symbols

    parser = argparse.ArgumentParser(description="Sentiment pipeline: Alpaca News + FinBERT")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build",    action="store_true", help="Full build from scratch")
    group.add_argument("--update",   action="store_true", help="Refresh last 30 days")
    group.add_argument("--upload",   action="store_true", help="Upload parquets to S3")
    group.add_argument("--download", action="store_true", help="Download parquets from S3")
    parser.add_argument("--symbols",    nargs="+", default=None)
    parser.add_argument("--start-year", type=int, default=2017,
                        help="Start year for full build (default: 2017, Alpaca coverage starts here)")
    parser.add_argument("--end-year",   type=int, default=pd.Timestamp.today().year)
    args = parser.parse_args()

    symbols = args.symbols or fetch_sp500_symbols()

    if args.build:
        build_all(symbols, args.start_year, args.end_year)
    elif args.update:
        update_all(symbols)
    elif args.upload:
        upload_to_s3(symbols)
    elif args.download:
        download_from_s3()


if __name__ == "__main__":
    main()
