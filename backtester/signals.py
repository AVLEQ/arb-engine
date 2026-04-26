"""
backtester/signals.py — Signal Generation
Computes all entry/exit signals on top of raw pair data.
Outputs a signals DataFrame with z-score, cointegration, and ML confidence.
"""

import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("signals")


# ─────────────────────────────────────────────────────────────────────────────
# SPREAD & Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread_zscore(
    df: pd.DataFrame,
    lookback: int = None,
) -> pd.DataFrame:
    """
    Compute rolling z-score of the spread.
    Uses OLS hedge ratio estimated on a rolling window (not fixed 1:1 ratio).
    This is the proper pairs trading spread, not a naive log-price difference.
    """
    lookback = lookback or CFG.signal.zscore_lookback
    df = df.copy()

    # Rolling OLS hedge ratio (beta) — how many units of leg2 per unit of leg1
    betas, alphas = [], []
    for i in range(len(df)):
        if i < lookback:
            betas.append(np.nan)
            alphas.append(np.nan)
            continue
        window = df.iloc[i - lookback : i]
        # OLS: log(close1) = alpha + beta * log(close2) + eps
        y = np.log(window["close1"])
        x = np.log(window["close2"])
        x_const = np.column_stack([np.ones(len(x)), x])
        try:
            result = np.linalg.lstsq(x_const, y, rcond=None)
            a, b = result[0]
            alphas.append(a)
            betas.append(b)
        except Exception:
            betas.append(np.nan)
            alphas.append(np.nan)

    df["hedge_ratio"] = betas
    df["ols_alpha"] = alphas

    # Spread using rolling hedge ratio
    df["spread_ols"] = (
        np.log(df["close1"])
        - df["hedge_ratio"] * np.log(df["close2"])
        - df["ols_alpha"]
    )

    # Rolling z-score
    roll = df["spread_ols"].rolling(lookback)
    df["spread_mean"] = roll.mean()
    df["spread_std"]  = roll.std()
    df["zscore"] = (df["spread_ols"] - df["spread_mean"]) / df["spread_std"]

    # Spread in basis points (simpler metric for reporting)
    df["spread_bps"] = (
        (df["close1"] / df["close2_usd"] - df["close1"].shift(lookback) / df["close2_usd"].shift(lookback))
        * 10000
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# COINTEGRATION (rolling)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cointegration(
    df: pd.DataFrame,
    lookback: int = None,
    freq: int = 5,          # recompute every N days (expensive otherwise)
) -> pd.DataFrame:
    """
    Rolling Engle-Granger cointegration test.
    Adds 'coint_pval' column. Only computed every `freq` days for speed.
    """
    lookback = lookback or CFG.signal.lookback_coint
    df = df.copy()
    pvals = [np.nan] * len(df)

    for i in range(lookback, len(df)):
        if i % freq != 0:                       # reuse last computed value
            pvals[i] = pvals[i - 1] if i > 0 else np.nan
            continue
        window = df.iloc[i - lookback : i]
        try:
            s, pval, _ = coint(
                np.log(window["close1"]),
                np.log(window["close2"]),
                trend="c"
            )
            pvals[i] = pval
        except Exception:
            pvals[i] = 1.0   # treat as not cointegrated on error

    df["coint_pval"] = pvals
    df["is_cointegrated"] = df["coint_pval"] < CFG.signal.cointegration_pval
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ADF STATIONARITY (spread should be stationary if cointegrated)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_adf(
    df: pd.DataFrame,
    lookback: int = 60,
    freq: int = 5,
) -> pd.DataFrame:
    """Augmented Dickey-Fuller test on the spread residual."""
    df = df.copy()
    adf_pvals = [np.nan] * len(df)
    for i in range(lookback, len(df)):
        if i % freq != 0:
            adf_pvals[i] = adf_pvals[i - 1] if i > 0 else np.nan
            continue
        window = df["spread_ols"].iloc[i - lookback : i].dropna()
        if len(window) < 20:
            adf_pvals[i] = 1.0
            continue
        try:
            result = adfuller(window, maxlags=1, autolag=None)
            adf_pvals[i] = result[1]
        except Exception:
            adf_pvals[i] = 1.0
    df["adf_pval"] = adf_pvals
    df["spread_stationary"] = df["adf_pval"] < 0.10
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM FEATURES (for ML)
# ─────────────────────────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and vol features used as ML model inputs."""
    df = df.copy()

    # Spread momentum
    df["spread_mom_5d"]  = df["spread_ols"].diff(5)
    df["spread_mom_10d"] = df["spread_ols"].diff(10)

    # Leg momentum
    df["leg1_mom_10d"] = np.log(df["close1"] / df["close1"].shift(10))
    df["leg2_mom_10d"] = np.log(df["close2"] / df["close2"].shift(10))

    # FX vol
    df["fx_ret"] = np.log(df["fx"] / df["fx"].shift(1))
    df["fx_vol_20d"] = df["fx_ret"].rolling(20).std() * np.sqrt(252)

    # Days since last signal fired (tracks clustering)
    zscore_threshold = CFG.signal.zscore_entry
    signal_days = df["zscore"].abs() > zscore_threshold
    days_since = []
    last = 999
    for s in signal_days:
        if s:
            last = 0
        else:
            last += 1
        days_since.append(last)
    df["days_since_signal"] = days_since

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY / EXIT SIGNAL FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def generate_raw_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry and exit signal flags.

    signal:
        +1 = long leg1, short leg2 (spread too low, expect reversion up)
        -1 = short leg1, long leg2 (spread too high, expect reversion down)
         0 = no position
    """
    df = df.copy()
    cfg = CFG.signal

    # Volume filter: minimum liquidity gate
    vol_ok = df.get("vol_rank_min", pd.Series(1.0, index=df.index)) >= cfg.min_volume_pct

    # Cointegration gate
    coint_ok = df.get("is_cointegrated", pd.Series(True, index=df.index))

    # Z-score signal
    long_signal  = (df["zscore"] < -cfg.zscore_entry) & vol_ok & coint_ok
    short_signal = (df["zscore"] >  cfg.zscore_entry) & vol_ok & coint_ok
    exit_signal  = df["zscore"].abs() < cfg.zscore_exit

    df["raw_signal"]  = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    df["exit_flag"]   = exit_signal
    df["signal_valid"] = vol_ok & coint_ok

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL SIGNAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full pipeline: raw data → signals DataFrame.
    Runs all signal computations in order.
    """
    if verbose:
        log.info("Computing spread and z-score...")
    df = compute_spread_zscore(df)

    if verbose:
        log.info("Running rolling cointegration tests...")
    df = rolling_cointegration(df)

    if verbose:
        log.info("Running rolling ADF tests...")
    df = rolling_adf(df)

    if verbose:
        log.info("Adding momentum features...")
    df = add_momentum_features(df)

    if verbose:
        log.info("Generating entry/exit flags...")
    df = generate_raw_signals(df)

    if verbose:
        n_long  = (df["raw_signal"] == 1).sum()
        n_short = (df["raw_signal"] == -1).sum()
        coint_pct = df["is_cointegrated"].mean() * 100
        log.info(f"  Long signals: {n_long}, Short signals: {n_short}")
        log.info(f"  Cointegrated (% of days): {coint_pct:.1f}%")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# THEORETICAL EDGE (bps)
# ─────────────────────────────────────────────────────────────────────────────

def compute_theoretical_edge(df: pd.DataFrame) -> pd.Series:
    """
    Theoretical edge = expected spread reversion in bps.
    = |zscore| * spread_std * 10000, adjusted for cost.
    """
    gross_edge = df["zscore"].abs() * df["spread_std"] * 10000
    # subtract total round-trip cost
    total_cost_bps = (
        (CFG.costs.us_commission_pct + CFG.costs.us_bid_ask_pct) * 2
        + (CFG.costs.in_stt_pct + CFG.costs.in_bid_ask_pct + CFG.costs.in_brokerage_pct) * 2
        + CFG.costs.fx_spread_pct * 2
    ) * 10000
    return (gross_edge - total_cost_bps).clip(lower=0)
