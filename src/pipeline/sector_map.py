"""
Symbol -> GICS sector mapping, used by trader.py's sector concentration cap.

Source: Wikipedia's "List of S&P 500 companies" (GICS Sector column) — same
page/table fetch_sp500_symbols() already uses in data_pipeline.py. ETFs (not
S&P 500 constituents, so absent from that table) are tagged "ETF" and excluded
from sector-cap grouping, same as the single-position cap exemption in
safeguards.py's _BUCKET_ETFS.

Storage: src/pipeline/sector_map.csv — columns: symbol, sector. Checked into
git and shipped with the Docker image (not under src/data/, which is volume-
mounted from EBS at runtime and would shadow a baked-in copy).

CLI:
  python sector_map.py --build   # scrape Wikipedia + fill gaps via yfinance, save CSV
"""

import os
import io
import sys
import argparse
import logging
import requests
import pandas as pd

log = logging.getLogger(__name__)

_here = os.path.dirname(os.path.abspath(__file__))
SECTOR_MAP_FILE = os.path.join(_here, "sector_map.csv")
WIKI_URL        = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Not S&P 500 constituents — won't appear in the Wikipedia table. Tagged "ETF"
# so the sector cap skips them (same intent as safeguards.py's _BUCKET_ETFS).
KNOWN_ETFS = ["SPY", "QQQ", "XLP", "XLV", "XLU", "GLD", "XLI", "IWM", "SH", "SQQQ"]

_cache = None  # module-level cache so repeated calls in one rebalance don't re-read the file


def build_sector_map(extra_symbols=None):
    """
    Scrape the Wikipedia S&P 500 table for Symbol -> GICS Sector, add known
    ETFs as "ETF", and fall back to yfinance for any requested symbol still
    missing (e.g. a very recent index addition Wikipedia hasn't caught up on).
    Saves the result to SECTOR_MAP_FILE.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp    = requests.get(WIKI_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    symbols = table["Symbol"].str.replace(".", "-", regex=False)
    sector_map = dict(zip(symbols, table["GICS Sector"]))
    print(f"Wikipedia: {len(sector_map)} symbols mapped to GICS sectors.")

    for etf in KNOWN_ETFS:
        sector_map[etf] = "ETF"

    missing = [s for s in (extra_symbols or []) if s not in sector_map]
    if missing:
        print(f"Falling back to yfinance for {len(missing)} symbol(s) not on Wikipedia: {missing}")
        import yfinance as yf
        for sym in missing:
            try:
                sector = yf.Ticker(sym).info.get("sector")
                if sector:
                    sector_map[sym] = sector
                else:
                    print(f"  {sym}: yfinance returned no sector — leaving unmapped")
            except Exception as e:
                print(f"  {sym}: yfinance lookup failed ({e}) — leaving unmapped")

    df = pd.DataFrame(sorted(sector_map.items()), columns=["symbol", "sector"])
    df.to_csv(SECTOR_MAP_FILE, index=False)
    print(f"Saved {len(df)} symbol->sector mappings to {SECTOR_MAP_FILE}")
    return sector_map


def load_sector_map():
    """Load symbol -> GICS sector dict from SECTOR_MAP_FILE. Cached after first call."""
    global _cache
    if _cache is None:
        df = pd.read_csv(SECTOR_MAP_FILE)
        _cache = dict(zip(df["symbol"], df["sector"]))
    return _cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Scrape Wikipedia and save sector_map.csv")
    args = parser.parse_args()

    if args.build:
        sys.path.insert(0, _here)
        from data_pipeline import fetch_sp500_symbols, ETF_SYMBOLS
        universe = [s for s in set(fetch_sp500_symbols()) | set(ETF_SYMBOLS) if "-" not in s]
        result = build_sector_map(extra_symbols=universe)
        covered = [s for s in universe if s in result]
        print(f"\nCoverage: {len(covered)}/{len(universe)} live-universe symbols mapped.")
        uncovered = sorted(set(universe) - set(covered))
        if uncovered:
            print(f"Unmapped (sector cap will skip these): {uncovered}")
