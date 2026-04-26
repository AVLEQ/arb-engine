"""
risk/framework.py — Risk Framework
Position sizing logic, circuit breakers, and regime detection.
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("risk")


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING REGIME MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class RegimeMonitor:
    """
    Monitors strategy health in real-time. Flags:
    - Rolling Sharpe degradation
    - Cointegration breakdown
    - FX volatility spike
    - VIX regime change
    """

    def __init__(self):
        self.state = "HEALTHY"
        self.flags: Dict[str, bool] = {
            "low_sharpe": False,
            "coint_broken": False,
            "fx_spike": False,
            "high_vix": False,
        }

    def update(self, results_df: pd.DataFrame, signals_df: pd.DataFrame) -> str:
        """
        Update regime state from latest data.
        Returns: "HEALTHY" | "DEGRADED" | "SUSPENDED"
        """
        # Rolling Sharpe (last 30 days of returns)
        recent_ret = results_df["returns"].dropna().iloc[-CFG.risk.rolling_sharpe_window:]
        if len(recent_ret) >= 10:
            roll_sharpe = (recent_ret.mean() / max(recent_ret.std(), 1e-6)) * np.sqrt(252)
            self.flags["low_sharpe"] = roll_sharpe < CFG.risk.rolling_sharpe_min
            log.info(f"Rolling Sharpe (30d): {roll_sharpe:.2f} | Flag: {self.flags['low_sharpe']}")

        # Cointegration (most recent value)
        if "coint_pval" in signals_df.columns:
            latest_pval = signals_df["coint_pval"].dropna().iloc[-1]
            self.flags["coint_broken"] = latest_pval > CFG.risk.coint_pval_max
            log.info(f"Latest cointegration p-value: {latest_pval:.4f} | Flag: {self.flags['coint_broken']}")

        # FX vol spike
        if "fx_vol_20d" in signals_df.columns:
            fx_vol = signals_df["fx_vol_20d"].dropna()
            if len(fx_vol) > 30:
                recent_fxvol = fx_vol.iloc[-1]
                mean_fxvol = fx_vol.iloc[:-1].mean()
                std_fxvol  = fx_vol.iloc[:-1].std()
                z_fxvol = (recent_fxvol - mean_fxvol) / max(std_fxvol, 1e-6)
                self.flags["fx_spike"] = z_fxvol > CFG.risk.fx_vol_sigma
                log.info(f"FX vol z-score: {z_fxvol:.2f} | Flag: {self.flags['fx_spike']}")

        # VIX regime
        if "vix" in signals_df.columns:
            latest_vix = signals_df["vix"].dropna().iloc[-1]
            self.flags["high_vix"] = latest_vix > CFG.risk.vix_high
            log.info(f"VIX: {latest_vix:.1f} | High vol flag: {self.flags['high_vix']}")

        # Determine overall regime
        n_flags = sum(self.flags.values())
        if n_flags == 0:
            self.state = "HEALTHY"
        elif n_flags == 1:
            self.state = "DEGRADED"
        else:
            self.state = "SUSPENDED"

        log.info(f"Regime state: {self.state} ({n_flags} flags active)")
        return self.state

    def size_multiplier(self) -> float:
        """Return position size multiplier based on regime."""
        if self.state == "SUSPENDED":
            return 0.0
        elif self.state == "DEGRADED":
            return 0.5
        else:
            return 1.0

    def report(self) -> Dict:
        return {
            "state": self.state,
            "flags": self.flags,
            "size_multiplier": self.size_multiplier(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING — STANDALONE (for documentation)
# ─────────────────────────────────────────────────────────────────────────────

def fractional_kelly(
    expected_return: float,
    variance: float,
    fraction: float = None,
) -> float:
    """
    Fractional Kelly criterion.
    f* = (E[r] / Var[r]) * fraction

    Returns: optimal fraction of capital to deploy (0 to 1)
    """
    fraction = fraction or CFG.sizing.kelly_fraction
    if variance <= 0 or expected_return <= 0:
        return 0.0
    full_kelly = expected_return / variance
    return min(full_kelly * fraction, CFG.sizing.max_position_pct)


def vol_target_size(
    base_fraction: float,
    realized_vol: float,
    target_vol: float = None,
) -> float:
    """
    Scale position to hit a target annualised volatility.
    scalar = target_vol / realized_vol, clamped to [0.25, 2.0]
    """
    target_vol = target_vol or CFG.sizing.target_vol_annual
    if realized_vol <= 0:
        return base_fraction
    scalar = np.clip(target_vol / realized_vol, 0.25, 2.0)
    return base_fraction * scalar


def position_size_summary(
    capital: float,
    edge_bps: float,
    spread_vol: float,
    realized_vol: float,
    vix: float,
    regime_multiplier: float = 1.0,
) -> Dict:
    """
    Full position sizing calculation with all steps shown.
    Returns a detailed breakdown for transparency.
    """
    # Step 1: Convert edge to fraction
    edge_frac = edge_bps / 10000

    # Step 2: Full Kelly
    variance = spread_vol ** 2
    full_kelly_frac = fractional_kelly(edge_frac, variance, fraction=1.0)

    # Step 3: Fractional Kelly
    frac_kelly = full_kelly_frac * CFG.sizing.kelly_fraction

    # Step 4: Vol scaling
    vol_scaled = vol_target_size(frac_kelly, realized_vol)

    # Step 5: Regime adjustment
    regime_adjusted = vol_scaled * regime_multiplier

    # Step 6: VIX adjustment
    if vix > CFG.risk.vix_high:
        regime_adjusted *= CFG.risk.size_reduction_high_vol

    # Step 7: Hard cap
    final_frac = min(regime_adjusted, CFG.sizing.max_position_pct)
    notional = final_frac * capital

    return {
        "edge_bps": round(edge_bps, 2),
        "spread_vol": round(spread_vol, 4),
        "realized_vol_annual": round(realized_vol, 4),
        "vix": round(vix, 1),
        "full_kelly_fraction": round(full_kelly_frac, 4),
        "fractional_kelly_pct": round(CFG.sizing.kelly_fraction * 100),
        "after_kelly_frac": round(frac_kelly, 4),
        "after_vol_scale_frac": round(vol_scaled, 4),
        "regime_multiplier": round(regime_multiplier, 2),
        "after_regime_frac": round(regime_adjusted, 4),
        "final_fraction": round(final_frac, 4),
        "notional_usd": round(notional, 2),
        "pct_of_capital": round(final_frac * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_sharpe(
    returns: pd.Series,
    n_simulations: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap Monte Carlo: resample returns with replacement N times.
    Returns distribution of Sharpe ratios to quantify estimation uncertainty.
    """
    np.random.seed(seed)
    ret = returns.dropna().values
    sharpes = []
    for _ in range(n_simulations):
        sample = np.random.choice(ret, size=len(ret), replace=True)
        sr = sample.mean() / max(sample.std(), 1e-8) * np.sqrt(252)
        sharpes.append(sr)
    sharpes = np.array(sharpes)
    return {
        "mean_sharpe":   round(np.mean(sharpes), 3),
        "median_sharpe": round(np.median(sharpes), 3),
        "p5_sharpe":     round(np.percentile(sharpes, 5), 3),
        "p95_sharpe":    round(np.percentile(sharpes, 95), 3),
        "pct_positive":  round((sharpes > 0).mean() * 100, 1),
        "n_simulations": n_simulations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SENSITIVITY (for stress testing)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_grid(
    df_signals: pd.DataFrame,
    z_thresholds: list = None,
    lookbacks: list = None,
) -> pd.DataFrame:
    """
    Run backtester across a grid of (z_threshold, lookback) parameters.
    Returns DataFrame of Sharpe ratios for the heatmap.
    """
    from backtester.signals import compute_spread_zscore, generate_raw_signals, rolling_cointegration
    from backtester.engine import Backtester

    z_thresholds = z_thresholds or [1.5, 2.0, 2.5, 3.0]
    lookbacks    = lookbacks    or [20, 30, 45, 60]

    results = pd.DataFrame(index=z_thresholds, columns=lookbacks, dtype=float)

    for z in z_thresholds:
        for lb in lookbacks:
            try:
                # Temporarily override config
                df2 = compute_spread_zscore(df_signals.copy(), lookback=lb)
                df2["raw_signal"] = np.where(
                    df2["zscore"] < -z, 1,
                    np.where(df2["zscore"] > z, -1, 0)
                )
                df2["exit_flag"] = df2["zscore"].abs() < CFG.signal.zscore_exit

                bt = Backtester("sensitivity_test", capital=1_000_000)
                res = bt.run(df2)
                ret = res["returns"].dropna()
                sr = ret.mean() / max(ret.std(), 1e-8) * np.sqrt(252)
                results.loc[z, lb] = round(sr, 3)
            except Exception:
                results.loc[z, lb] = np.nan

    results.index.name = "z_threshold"
    results.columns.name = "lookback_days"
    return results
