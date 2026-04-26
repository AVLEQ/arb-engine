"""
backtester/metrics.py — Performance Metrics & Reporting
Computes all required + bonus metrics. Generates plots.
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("metrics")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, trade_log: pd.DataFrame = None, rf: float = 0.045) -> Dict:
    """
    Compute comprehensive performance metrics.
    rf: risk-free rate (~4.5%)
    """

    # 🔧 SAFETY FIX — ensure returns exist
    if "returns" not in results.columns:
        results["returns"] = results["equity"].pct_change()

    ret = results["returns"].dropna()
    eq = results["equity"].dropna()

    if len(ret) == 0 or len(eq) == 0:
        log.warning("No returns data — skipping metrics")
        return {}

    # Annualisation
    n_days = len(ret)
    ann_factor = 252 / max(n_days, 1)

    # Returns
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    ann_return = (1 + total_return) ** ann_factor - 1

    # Volatility
    ann_vol = ret.std() * np.sqrt(252)

    # ─────────────────────────────────────────────────────────────
    # Sharpe — Trade-Level (preferred)
    # ─────────────────────────────────────────────────────────────
    if trade_log is not None and len(trade_log) > 0:
        closed = trade_log[trade_log["exit_date"].notna()]

        if len(closed) > 1:
            trade_returns = closed["net_pnl"] / (closed["notional"] * 2)
            trade_returns = trade_returns.clip(-0.2, 0.2)

            avg_ret = trade_returns.mean()
            std_ret = trade_returns.std()

            trades_per_year = len(closed) / (n_days / 252)

            if std_ret > 1e-6:
                sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
    else:
        excess = ret.mean() * 252 - rf
        vol = max(ret.std() * np.sqrt(252), 1e-6)
        sharpe = excess / vol

    # ─────────────────────────────────────────────────────────────
    # Sortino
    # ─────────────────────────────────────────────────────────────
    excess_ann = ann_return - rf
    daily_rf = rf / 252

    downside = ret[ret < daily_rf] - daily_rf

    if len(downside) > 0:
        downside_ann_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    else:
        downside_ann_vol = ann_vol

    downside_ann_vol = max(downside_ann_vol, 1e-6)
    sortino = excess_ann / downside_ann_vol

    # ─────────────────────────────────────────────────────────────
    # Drawdown
    # ─────────────────────────────────────────────────────────────
    cum = (1 + ret).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "backtest_days": n_days,
    }

    # ─────────────────────────────────────────────────────────────
    # Trade stats
    # ─────────────────────────────────────────────────────────────
    if trade_log is not None and len(trade_log) > 0:
        closed = trade_log[trade_log["exit_date"].notna()]

        if len(closed) > 0:
            metrics["total_trades"] = len(closed)
            metrics["win_rate_pct"] = round((closed["net_pnl"] > 0).mean() * 100, 1)
            metrics["avg_hold_days"] = round(closed["hold_days"].mean(), 1)
            metrics["avg_net_pnl"] = round(closed["net_pnl"].mean(), 2)
            metrics["total_gross_pnl"] = round(closed["gross_pnl"].sum(), 2)
            metrics["total_costs"] = round(closed["total_costs"].sum(), 2)
            metrics["total_net_pnl"] = round(closed["net_pnl"].sum(), 2)

            metrics["cost_drag_pct"] = round(
                closed["total_costs"].sum()
                / max(abs(closed["gross_pnl"].sum()), 1)
                * 100,
                1,
            )

            avg_equity = eq.mean()
            total_notional = closed["notional"].sum() * 2

            metrics["annual_turnover_pct"] = round(
                total_notional / avg_equity / (n_days / 252) * 100,
                1,
            )

            metrics["exit_reasons"] = closed["exit_reason"].value_counts().to_dict()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PRINT METRICS
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict, pair_key: str = ""):
    print("\n" + "=" * 58)
    print(f"  BACKTEST RESULTS  {pair_key}")
    print("=" * 58)

    sections = {
        "Returns": ["total_return_pct", "ann_return_pct", "ann_volatility_pct"],
        "Risk-Adjusted": ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct"],
        "Trades": ["total_trades", "win_rate_pct", "avg_hold_days", "avg_net_pnl",
                   "annual_turnover_pct", "cost_drag_pct"],
        "P&L": ["total_gross_pnl", "total_costs", "total_net_pnl"],
    }

    for section, keys in sections.items():
        print(f"\n  {section}")
        print("  " + "-" * 40)
        for k in keys:
            if k in metrics:
                v = metrics[k]
                label = k.replace("_", " ").title()

                if isinstance(v, float):
                    if "pct" in k:
                        print(f"    {label:<28} {v:>8.2f}%")
                    else:
                        print(f"    {label:<28} {v:>8.3f}")
                elif isinstance(v, dict):
                    print(f"    {label:<28} {v}")
                else:
                    print(f"    {label:<28} {v:>8}")

    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results: pd.DataFrame,
    trade_log: pd.DataFrame,
    signals_df: pd.DataFrame,
    pair_key: str = "PAIR",
    save_path: str = None,
):
    """
    Simple equity + drawdown plot
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    eq = results["equity"]

    # Equity
    axes[0].plot(eq.index, eq.values, linewidth=1.5)
    axes[0].set_title(f"Equity Curve — {pair_key}")
    axes[0].grid(True)

    # Drawdown
    drawdown = (eq / eq.cummax() - 1) * 100
    axes[1].fill_between(drawdown.index, drawdown.values, 0)
    axes[1].set_title("Drawdown (%)")
    axes[1].grid(True)

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"backtest_{pair_key}.png"

    plt.savefig(save_path, dpi=120)
    plt.close()

    log.info(f"Plot saved -> {save_path}")
    return str(save_path)