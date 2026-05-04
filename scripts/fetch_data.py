"""
scripts/fetch_data.py
---------------------
Downloads stock price data via yfinance and saves processed
train/val/test splits ready for both Python baselines and HeatWave ingestion.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --config configs/benchmark.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "configs/benchmark.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_tickers(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for all tickers and return a combined DataFrame."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    frames = []
    for ticker in tickers:
        log.info(f"Fetching {ticker} ...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            log.warning(f"No data returned for {ticker}, skipping.")
            continue
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.insert(0, "ticker", ticker)
        frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched for any ticker.")

    combined = pd.concat(frames).sort_index()
    log.info(f"Combined dataset: {len(combined):,} rows across {len(tickers)} tickers")
    return combined


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicator features per ticker."""
    result = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.copy().sort_index()
        close = grp["close"]

        # Returns
        grp["return_1d"] = close.pct_change(1)
        grp["return_5d"] = close.pct_change(5)

        # Rolling statistics
        for w in [5, 10, 20]:
            grp[f"sma_{w}"] = close.rolling(w).mean()
            grp[f"std_{w}"] = close.rolling(w).std()

        # Volatility proxy
        grp["hl_spread"] = (grp["high"] - grp["low"]) / grp["close"]

        # Volume change
        grp["vol_change_1d"] = grp["volume"].pct_change(1)

        result.append(grp)

    out = pd.concat(result).sort_index()
    n_before = len(out)
    out = out.dropna()
    log.info(f"Dropped {n_before - len(out)} rows with NaN after feature engineering. "
             f"Final: {len(out):,} rows")
    return out


def split_data(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple:
    """
    Time-aware split: preserves temporal order within each ticker.
    Returns (train_df, val_df, test_df).
    """
    trains, vals, tests = [], [], []
    for _, grp in df.groupby("ticker"):
        grp = grp.sort_index()
        n = len(grp)
        i_train = int(n * train_ratio)
        i_val = int(n * (train_ratio + val_ratio))
        trains.append(grp.iloc[:i_train])
        vals.append(grp.iloc[i_train:i_val])
        tests.append(grp.iloc[i_val:])

    return pd.concat(trains), pd.concat(vals), pd.concat(tests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]

    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch
    df = fetch_tickers(ds["tickers"], ds["start_date"], ds["end_date"])
    raw_path = raw_dir / "ohlcv_raw.parquet"
    df.to_parquet(raw_path)
    log.info(f"Raw data saved → {raw_path}")

    # 2. Feature engineering
    df = add_features(df)

    # 3. Split
    train, val, test = split_data(df, ds["train_ratio"], ds["val_ratio"])
    log.info(f"Split sizes → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = proc_dir / f"{name}.parquet"
        split.to_parquet(path)
        log.info(f"Saved {name} split → {path}")

    # 4. Also export CSV for HeatWave ingestion
    hw_path = proc_dir / "heatwave_import.csv"
    df.reset_index().to_csv(hw_path, index=False)
    log.info(f"HeatWave CSV export → {hw_path}")

    # 5. Quick summary
    print("\n── Dataset Summary ───────────────────────────────")
    print(f"  Tickers   : {', '.join(df['ticker'].unique())}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Features  : {[c for c in df.columns if c != 'ticker']}")
    print(f"  Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print("──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
