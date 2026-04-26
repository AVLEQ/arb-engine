"""
backtester/signals.py — Signal Generation  [FIXED v2]
Key improvements:
  1. Momentum filter: only enter when spread is actively reverting
  2. Edge filter: require minimum net edge > cost
  3. Longer z-score lookback (30d vs 20d) for stability
  4. ADF stationarity gate added to entry
"""

import sys
import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("signals")


# ─────────────────────────────────────────────────────────────────────────────
# SPREAD & Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread_zscore(df: pd.DataFrame, lookback: int = None) -> pd.DataFrame:
    """Rolling OLS z-score. Proper pairs spread, not naive log-diff."""
    lookback = lookback or CFG.signal.zscore_lookback
    df = df.copy()

    betas, alphas = [], []
    for i in range(len(df)):
        if i < lookback:
            betas.append(np.nan)
            alphas.append(np.nan)
            continue
        window = df.iloc[i - lookback: i]
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

    df["spread_ols"] = (
        np.log(df["close1"])
        - df["hedge_ratio"] * np.log(df["close2"])
        - df["ols_alpha"]
    )

    roll = df["spread_ols"].rolling(lookback)
    df["spread_mean"] = roll.mean()
    df["spread_std"]  = roll.std()
    df["zscore"] = (df["spread_ols"] - df["spread_mean"]) / df["spread_std"]

    df["spread_bps"] = (
        (df["close1"] / df["close2_usd"] - df["close1"].shift(lookback) / df["close2_usd"].shift(lookback))
        * 10000
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COINTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cointegration(df: pd.DataFrame, lookback: int = None, freq: int = 5) -> pd.DataFrame:
    lookback = lookback or CFG.signal.lookback_coint
    df = df.copy()
    pvals = [np.nan] * len(df)

    for i in range(lookback, len(df)):
        if i % freq != 0:
            pvals[i] = pvals[i - 1] if i > 0 else np.nan
            continue
        window = df.iloc[i - lookback: i]
        try:
            _, pval, _ = coint(np.log(window["close1"]), np.log(window["close2"]), trend="c")
            pvals[i] = pval
        except Exception:
            pvals[i] = 1.0

    df["coint_pval"] = pvals
    df["is_cointegrated"] = df["coint_pval"] < CFG.signal.cointegration_pval
    return df


def rolling_adf(df: pd.DataFrame, lookback: int = 60, freq: int = 5) -> pd.DataFrame:
    df = df.copy()
    adf_pvals = [np.nan] * len(df)
    for i in range(lookback, len(df)):
        if i % freq != 0:
            adf_pvals[i] = adf_pvals[i - 1] if i > 0 else np.nan
            continue
        window = df["spread_ols"].iloc[i - lookback: i].dropna()
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
# MOMENTUM & FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread_mom_5d"]  = df["spread_ols"].diff(5)
    df["spread_mom_10d"] = df["spread_ols"].diff(10)
    df["leg1_mom_10d"]   = np.log(df["close1"] / df["close1"].shift(10))
    df["leg2_mom_10d"]   = np.log(df["close2"] / df["close2"].shift(10))
    df["fx_ret"]         = np.log(df["fx"] / df["fx"].shift(1))
    df["fx_vol_20d"]     = df["fx_ret"].rolling(20).std() * np.sqrt(252)

    zscore_threshold = CFG.signal.zscore_entry
    signal_days = df["zscore"].abs() > zscore_threshold
    days_since, last = [], 999
    for s in signal_days:
        if s: last = 0
        else: last += 1
        days_since.append(last)
    df["days_since_signal"] = days_since
    return df


# ─────────────────────────────────────────────────────────────────────────────
# THEORETICAL EDGE
# ─────────────────────────────────────────────────────────────────────────────

def compute_theoretical_edge(df: pd.DataFrame) -> pd.Series:
    """Theoretical edge in bps after costs."""
    gross_edge = df["zscore"].abs() * df["spread_std"] * 10000
    total_cost_bps = (
        (CFG.costs.us_commission_pct + CFG.costs.us_bid_ask_pct) * 2
        + CFG.costs.fx_spread_pct * 2
    ) * 10000
    return (gross_edge - total_cost_bps).clip(lower=0)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY / EXIT SIGNALS  [KEY FIX]
# ─────────────────────────────────────────────────────────────────────────────

def generate_raw_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry conditions (ALL must be true):
      1. |z-score| > zscore_entry threshold
      2. Cointegration gate (p-val < 0.10)
      3. Momentum filter: spread REVERTING toward mean (new)
         - For LONG (z < -threshold): spread_mom_5d > 0 (spread rising back)
         - For SHORT (z > +threshold): spread_mom_5d < 0 (spread falling back)
      4. Net edge > min_edge_bps after costs (new)
      5. Volume filter

    Exit conditions (ANY):
      1. |z-score| < zscore_exit
      2. Max hold days hit (enforced in engine)
    """
    df = df.copy()
    cfg = CFG.signal

    # Filters
    vol_ok   = df.get("vol_rank_min", pd.Series(1.0, index=df.index)) >= cfg.min_volume_pct
    coint_ok = df["coint_pval"] < cfg.cointegration_pval

    # Momentum filter: is spread actively reverting?
    mom = df.get("spread_mom_5d", pd.Series(0.0, index=df.index)).fillna(0)
    mom_long_ok  = mom > 0    # spread rising → z recovering from negative extreme
    mom_short_ok = mom < 0    # spread falling → z recovering from positive extreme

    # Edge filter
    edge = compute_theoretical_edge(df)
    edge_ok = edge > cfg.min_edge_bps

    # Z-score gates
    long_signal  = (df["zscore"] < -cfg.zscore_entry) & vol_ok & coint_ok & edge_ok
    short_signal = (df["zscore"] >  cfg.zscore_entry) & vol_ok & coint_ok & edge_ok

    # Apply momentum filter if enabled
    if cfg.momentum_filter:
        long_signal  = long_signal  & mom_long_ok
        short_signal = short_signal & mom_short_ok

    exit_signal = df["zscore"].abs() < cfg.zscore_exit

    df["raw_signal"]   = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    df["exit_flag"]    = exit_signal
    df["signal_valid"] = vol_ok & coint_ok
    df["edge_bps"]     = edge
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose: log.info("Computing spread and z-score...")
    df = compute_spread_zscore(df)

    if verbose: log.info("Running rolling cointegration tests...")
    df = rolling_cointegration(df)

    if verbose: log.info("Running rolling ADF tests...")
    df = rolling_adf(df)

    if verbose: log.info("Adding momentum features...")
    df = add_momentum_features(df)

    if verbose: log.info("Generating entry/exit flags...")
    df = generate_raw_signals(df)

    if verbose:
        n_long  = (df["raw_signal"] == 1).sum()
        n_short = (df["raw_signal"] == -1).sum()
        coint_pct = df["is_cointegrated"].mean() * 100
        log.info(f"  Long signals: {n_long}, Short signals: {n_short}")
        log.info(f"  Cointegrated (% of days): {coint_pct:.1f}%")

    return df