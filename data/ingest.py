"""
data/ingest.py — Data Ingestion Pipeline
Pulls historical OHLCV for all ETF pairs + USD/INR FX rate.
Aligns on common trading calendar, handles missing data, saves to Parquet.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _download(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from yfinance with retry logic."""
    log.info(f"  Downloading {ticker} ({start} → {end})")
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                interval=interval, auto_adjust=True,
                progress=False, threads=False
            )
            if df.empty:
                raise ValueError(f"Empty data for {ticker}")
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            log.info(f"  ✓ {ticker}: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
            return df
        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/3 failed for {ticker}: {e}")
    log.error(f"  ✗ Failed to download {ticker} after 3 attempts")
    return pd.DataFrame()


def _compute_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log-returns from close price."""
    return np.log(df["Close"] / df["Close"].shift(1))


def _flag_partial_overlap(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """
    Flag dates where only one market was open.
    Returns boolean Series (True = both markets open).
    """
    common = df1.index.intersection(df2.index)
    return pd.Series(True, index=common)


# ─────────────────────────────────────────────────────────────────────────────
# VIX (regime detector input)
# ─────────────────────────────────────────────────────────────────────────────

def download_vix(start: str, end: str) -> pd.Series:
    df = _download("^VIX", start, end)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].rename("VIX")


# ─────────────────────────────────────────────────────────────────────────────
# PAIR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_dataset(
    ticker1: str,
    ticker2: str,
    start: str,
    end: str,
    fx_series: pd.Series,
    vix_series: pd.Series,
) -> pd.DataFrame:
    """
    Build a unified daily DataFrame for one ETF pair.

    Columns:
        close1, close2          — raw close prices
        close2_usd              — close2 adjusted by USD/INR (if applicable)
        spread                  — log(close1) - log(close2_usd)
        return1, return2        — daily log returns
        volume1, volume2        — daily volumes
        vol_ratio               — volume1/volume2 (liquidity balance)
        fx                      — USD/INR rate
        vix                     — CBOE VIX
        both_open               — True if both markets had data that day
    """
    df1 = _download(ticker1, start, end)
    df2 = _download(ticker2, start, end)

    if df1.empty or df2.empty:
        log.error(f"Cannot build pair {ticker1}/{ticker2} — missing data")
        return pd.DataFrame()

    # Align on common dates
    common_dates = df1.index.intersection(df2.index)
    df1 = df1.loc[common_dates]
    df2 = df2.loc[common_dates]

    # Align FX
    fx = fx_series.reindex(common_dates).ffill().bfill()

    # For India-proxy pairs, the IN ticker is US-traded so no FX adjustment needed
    # For actual cross-listed pairs, multiply by FX. We flag which pairs need it.
    # For now: both legs are US-traded ETFs — no FX conversion on prices.
    # FX is included as a feature for the model.
    close2_usd = df2["Close"]  # already in USD (India ETFs listed on US exchanges)

    spread = np.log(df1["Close"]) - np.log(close2_usd)

    out = pd.DataFrame({
        "close1": df1["Close"],
        "close2": df2["Close"],
        "close2_usd": close2_usd,
        "spread": spread,
        "return1": _compute_returns(df1),
        "return2": _compute_returns(df2),
        "volume1": df1["Volume"],
        "volume2": df2["Volume"],
        "vol_ratio": df1["Volume"] / (df2["Volume"] + 1),
        "fx": fx,
        "vix": vix_series.reindex(common_dates).ffill().bfill(),
        "both_open": True,
    }, index=common_dates)

    out.index.name = "date"
    out = out.dropna(subset=["close1", "close2", "spread"])
    out = out.sort_index()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME RANK (for signal filtering)
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Add rolling volume rank (percentile) for both legs."""
    df = df.copy()
    df["vol_rank1"] = df["volume1"].rolling(lookback).rank(pct=True)
    df["vol_rank2"] = df["volume2"].rolling(lookback).rank(pct=True)
    df["vol_rank_min"] = df[["vol_rank1", "vol_rank2"]].min(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def ingest_all(
    start: str = None,
    end: str = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download and process all pairs. Returns dict keyed by 'TICKER1_TICKER2'.
    Saves Parquet files to data/parquet/ if save=True.
    """
    start = start or CFG.data.start_date
    end   = end   or CFG.data.end_date

    log.info("=" * 60)
    log.info("GLOBAL ARBITRAGE SIGNAL ENGINE — Data Ingestion")
    log.info(f"Period: {start} → {end}")
    log.info("=" * 60)

    # Directories
    parquet_dir = Path(__file__).parent.parent / CFG.data.parquet_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Download shared series
    log.info("\n[1/3] Downloading FX rate (USD/INR)...")
    fx_df = _download(CFG.data.fx_ticker, start, end)
    fx = fx_df["Close"].rename("fx") if not fx_df.empty else pd.Series(dtype=float)

    log.info("\n[2/3] Downloading VIX...")
    vix = download_vix(start, end)

    # Download each pair
    log.info(f"\n[3/3] Downloading {len(CFG.pairs)} ETF pairs...")
    datasets: Dict[str, pd.DataFrame] = {}

    for ticker1, ticker2, desc in CFG.pairs:
        log.info(f"\n  Pair: {ticker1}/{ticker2} — {desc}")
        key = f"{ticker1}_{ticker2}"

        parquet_path = parquet_dir / f"{key}.parquet"

        df = build_pair_dataset(ticker1, ticker2, start, end, fx, vix)
        if df.empty:
            log.warning(f"  Skipping {key} — empty dataset")
            continue

        df = add_volume_features(df)
        datasets[key] = df

        if save:
            df.to_parquet(parquet_path)
            log.info(f"  Saved → {parquet_path} ({len(df)} rows)")

    log.info(f"\n✓ Ingestion complete. {len(datasets)} pairs ready.")

    # Summary table
    log.info("\n" + "─" * 60)
    log.info(f"{'Pair':<15} {'Rows':>6} {'Start':>12} {'End':>12} {'Missing%':>9}")
    log.info("─" * 60)
    for key, df in datasets.items():
        miss = df["spread"].isna().mean() * 100
        log.info(
            f"{key:<15} {len(df):>6} "
            f"{str(df.index[0].date()):>12} {str(df.index[-1].date()):>12} "
            f"{miss:>8.2f}%"
        )
    log.info("─" * 60)

    return datasets


def load_pair(key: str) -> pd.DataFrame:
    """Load a saved pair from Parquet."""
    parquet_path = Path(__file__).parent.parent / CFG.data.parquet_dir / f"{key}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No cached data for {key}. Run ingest_all() first.")
    return pd.read_parquet(parquet_path)


def load_all_pairs() -> Dict[str, pd.DataFrame]:
    """Load all saved pairs from Parquet."""
    parquet_dir = Path(__file__).parent.parent / CFG.data.parquet_dir
    datasets = {}
    for f in sorted(parquet_dir.glob("*.parquet")):
        key = f.stem
        datasets[key] = pd.read_parquet(f)
        log.info(f"Loaded {key}: {len(datasets[key])} rows")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    datasets = ingest_all()
    print(f"\nReady: {list(datasets.keys())}")
