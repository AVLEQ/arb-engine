"""
backtester/metrics.py — Performance Metrics & Reporting
Computes all required + bonus metrics. Generates plots.
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

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
    rf: risk-free rate (current Fed Funds ~4.5%)
    """
    ret = results["returns"].dropna()
    eq  = results["equity"].dropna()

    # Annualisation
    n_days = len(ret)
    ann_factor = 252 / n_days

    # Returns
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    ann_return   = (1 + total_return) ** ann_factor - 1

    # Volatility — annualised directly
    ann_vol = ret.std() * np.sqrt(252)

    # Sharpe — use annualised vol with a sensible floor (0.5% annualised)
    # This prevents inflation on ultra-tight ETF pairs where daily vol is near zero
    excess_ann = ann_return - rf
    ann_vol_floored = max(ann_vol, 0.005)   # floor at 0.5% annualised vol
    sharpe = excess_ann / ann_vol_floored

    # Sortino — use annualised downside vol
    daily_rf = rf / 252
    downside = ret[ret < daily_rf] - daily_rf
    if len(downside) > 0:
        downside_ann_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        downside_ann_vol = max(downside_ann_vol, 0.005)
    else:
        downside_ann_vol = ann_vol_floored
    sortino = excess_ann / downside_ann_vol

    # Max Drawdown
    cum = (1 + ret).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Calmar (annualised return / max drawdown)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "total_return_pct":     round(total_return * 100, 2),
        "ann_return_pct":       round(ann_return * 100, 2),
        "ann_volatility_pct":   round(ann_vol * 100, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "sortino_ratio":        round(sortino, 3),
        "calmar_ratio":         round(calmar, 3),
        "max_drawdown_pct":     round(max_dd * 100, 2),
        "backtest_days":        n_days,
    }

    if trade_log is not None and len(trade_log) > 0:
        closed = trade_log[trade_log["exit_date"].notna()]
        if len(closed) > 0:
            metrics["total_trades"]        = len(closed)
            metrics["win_rate_pct"]        = round((closed["net_pnl"] > 0).mean() * 100, 1)
            metrics["avg_hold_days"]       = round(closed["hold_days"].mean(), 1)
            metrics["avg_net_pnl"]         = round(closed["net_pnl"].mean(), 2)
            metrics["total_gross_pnl"]     = round(closed["gross_pnl"].sum(), 2)
            metrics["total_costs"]         = round(closed["total_costs"].sum(), 2)
            metrics["total_net_pnl"]       = round(closed["net_pnl"].sum(), 2)
            metrics["cost_drag_pct"]       = round(
                closed["total_costs"].sum() / max(abs(closed["gross_pnl"].sum()), 1) * 100, 1
            )

            avg_equity = eq.mean()
            total_notional = closed["notional"].sum() * 2   # round-trip
            metrics["annual_turnover_pct"] = round(
                total_notional / avg_equity / (n_days / 252) * 100, 1
            )

            reason_counts = closed["exit_reason"].value_counts().to_dict()
            metrics["exit_reasons"] = reason_counts

    return metrics


def print_metrics(metrics: Dict, pair_key: str = ""):
    """Pretty print performance metrics."""
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
# MONTHLY PnL HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def monthly_pnl_heatmap(results: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns table (years x months)."""
    monthly = results["returns"].resample("ME").sum()
    monthly.index = monthly.index.to_period("M")
    pivot = pd.DataFrame(index=range(1, 13))
    for period, val in monthly.items():
        y, m = period.year, period.month
        pivot.loc[m, y] = val
    pivot.index = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    return pivot * 100


# ─────────────────────────────────────────────────────────────────────────────
# EDGE DECAY CURVE
# ─────────────────────────────────────────────────────────────────────────────

def edge_decay_curve(base_edge_bps: float = 12.0, max_notional: float = 50_000_000):
    """
    Plot how edge decays with position size (Kyle's lambda model).
    edge(Q) = base_edge - lambda * sqrt(Q / ADV)
    """
    ADV = 1_000_000
    lambdas = CFG.costs.lambda_impact
    Q = np.linspace(1000, max_notional, 500)
    edge = base_edge_bps - lambdas * np.sqrt(Q / ADV)
    capacity = Q[edge > 0][-1] if any(edge > 0) else 0
    return Q, edge, capacity


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results: pd.DataFrame,
    trade_log: pd.DataFrame,
    signals_df: pd.DataFrame,
    pair_key: str = "ETF_PAIR",
    save_path: str = None,
):
    """
    Comprehensive 6-panel backtest report chart.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    dark_bg    = "#0d1117"
    panel_bg   = "#161b22"
    text_color = "#e6edf3"
    grid_color = "#30363d"
    accent1    = "#58a6ff"
    accent2    = "#3fb950"
    accent3    = "#f85149"
    accent4    = "#d29922"

    def style_ax(ax, title=""):
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(True, color=grid_color, linewidth=0.4, alpha=0.7)
        if title:
            ax.set_title(title, color=text_color, fontsize=10, fontweight="bold", pad=8)

    # Panel 1: Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :2])
    eq = results["equity"]
    norm_eq = eq / eq.iloc[0] * 100
    ax1.plot(norm_eq.index, norm_eq.values, color=accent2, linewidth=1.5, label="Strategy")
    ax1.axhline(100, color=grid_color, linewidth=0.7, linestyle="--")

    drawdown = results["drawdown"]
    ax1.fill_between(
        results.index,
        norm_eq,
        norm_eq.where(drawdown >= 0, norm_eq * (1 - drawdown)),
        alpha=0.15, color=accent3, label="Drawdown"
    )

    if trade_log is not None and len(trade_log) > 0:
        closed = trade_log[trade_log["exit_date"].notna()].copy()
        closed["entry_date"] = pd.to_datetime(closed["entry_date"])
        for _, t in closed.iterrows():
            if t["entry_date"] in norm_eq.index:
                color = accent2 if t["net_pnl"] > 0 else accent3
                ax1.axvline(t["entry_date"], color=color, alpha=0.3, linewidth=0.5)

    style_ax(ax1, "Cumulative Equity (Indexed to 100)")
    ax1.set_ylabel("Index Value", color=text_color, fontsize=9)
    ax1.legend(fontsize=8, facecolor=panel_bg, labelcolor=text_color, framealpha=0.8)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(results.index, results["drawdown"] * 100, 0, color=accent3, alpha=0.7)
    ax2.axhline(-CFG.risk.drawdown_pause * 100, color=accent4, linewidth=1, linestyle="--",
                label=f"Pause ({CFG.risk.drawdown_pause*100:.0f}%)")
    ax2.axhline(-CFG.risk.drawdown_stop * 100, color=accent3, linewidth=1, linestyle="--",
                label=f"Stop ({CFG.risk.drawdown_stop*100:.0f}%)")
    style_ax(ax2, "Drawdown %")
    ax2.set_ylabel("%", color=text_color, fontsize=9)
    ax2.legend(fontsize=7, facecolor=panel_bg, labelcolor=text_color)

    # Panel 3: Z-Score
    ax3 = fig.add_subplot(gs[1, :2])
    sig_df = signals_df.dropna(subset=["zscore"])
    ax3.plot(sig_df.index, sig_df["zscore"], color=accent1, linewidth=0.8, label="Z-Score")
    ax3.axhline( CFG.signal.zscore_entry, color=accent3, linewidth=1, linestyle="--",
                label=f"Entry +/-{CFG.signal.zscore_entry}")
    ax3.axhline(-CFG.signal.zscore_entry, color=accent3, linewidth=1, linestyle="--")
    ax3.axhline( CFG.signal.zscore_exit, color=accent2, linewidth=0.8, linestyle=":")
    ax3.axhline(-CFG.signal.zscore_exit, color=accent2, linewidth=0.8, linestyle=":")
    ax3.axhline(0, color=grid_color, linewidth=0.7)
    ax3.fill_between(sig_df.index, sig_df["zscore"], CFG.signal.zscore_entry,
                     where=sig_df["zscore"] > CFG.signal.zscore_entry, alpha=0.2, color=accent3)
    ax3.fill_between(sig_df.index, sig_df["zscore"], -CFG.signal.zscore_entry,
                     where=sig_df["zscore"] < -CFG.signal.zscore_entry, alpha=0.2, color=accent2)
    style_ax(ax3, "Spread Z-Score (Entry/Exit Thresholds)")
    ax3.set_ylabel("Z-Score", color=text_color, fontsize=9)
    ax3.legend(fontsize=8, facecolor=panel_bg, labelcolor=text_color)

    # Panel 4: Cointegration p-value
    ax4 = fig.add_subplot(gs[1, 2])
    coint_series = signals_df["coint_pval"].dropna()
    ax4.plot(coint_series.index, coint_series.values, color=accent4, linewidth=0.8)
    ax4.axhline(CFG.signal.cointegration_pval, color=accent3, linewidth=1,
                linestyle="--", label=f"Threshold {CFG.signal.cointegration_pval}")
    ax4.fill_between(coint_series.index, coint_series, CFG.signal.cointegration_pval,
                     where=coint_series > CFG.signal.cointegration_pval,
                     alpha=0.3, color=accent3, label="Coint. broken")
    style_ax(ax4, "Rolling Cointegration p-value")
    ax4.set_ylabel("p-value", color=text_color, fontsize=9)
    ax4.legend(fontsize=7, facecolor=panel_bg, labelcolor=text_color)

    # Panel 5: Monthly PnL heatmap
    ax5 = fig.add_subplot(gs[2, :2])
    try:
        mpnl = monthly_pnl_heatmap(results)
        cmap = LinearSegmentedColormap.from_list("rg", [accent3, "#0d1117", accent2], N=256)
        max_abs = max(abs(mpnl.values[~np.isnan(mpnl.values)]).max(), 0.001)
        im = ax5.imshow(mpnl.values.T, cmap=cmap, aspect="auto",
                        vmin=-max_abs, vmax=max_abs)
        ax5.set_xticks(range(len(mpnl.index)))
        ax5.set_xticklabels(mpnl.index, fontsize=7, color=text_color)
        ax5.set_yticks(range(len(mpnl.columns)))
        ax5.set_yticklabels([str(y) for y in mpnl.columns], fontsize=7, color=text_color)
        for (i, j), val in np.ndenumerate(mpnl.values):
            if not np.isnan(val):
                ax5.text(i, j, f"{val:.1f}", ha="center", va="center",
                         fontsize=6.5, color="white" if abs(val) > max_abs * 0.3 else text_color)
        fig.colorbar(im, ax=ax5, label="%", shrink=0.8)
        style_ax(ax5, "Monthly Returns Heatmap (%)")
    except Exception as e:
        ax5.text(0.5, 0.5, f"Heatmap error:\n{e}", transform=ax5.transAxes,
                 ha="center", va="center", color=text_color)
        style_ax(ax5, "Monthly Returns Heatmap")

    # Panel 6: Edge decay curve
    ax6 = fig.add_subplot(gs[2, 2])
    Q, edge, capacity = edge_decay_curve()
    Q_m = Q / 1e6
    ax6.plot(Q_m, edge, color=accent1, linewidth=1.5)
    ax6.axhline(0, color=accent3, linewidth=1, linestyle="--")
    ax6.fill_between(Q_m, edge, 0, where=edge > 0, alpha=0.2, color=accent2, label="Positive edge")
    ax6.fill_between(Q_m, edge, 0, where=edge <= 0, alpha=0.2, color=accent3, label="Edge destroyed")
    ax6.axvline(capacity / 1e6, color=accent4, linewidth=1, linestyle=":",
                label=f"Capacity ~${capacity/1e6:.1f}M")
    style_ax(ax6, "Edge Decay vs Position Size")
    ax6.set_xlabel("Notional ($M)", color=text_color, fontsize=9)
    ax6.set_ylabel("Net Edge (bps)", color=text_color, fontsize=9)
    ax6.legend(fontsize=7, facecolor=panel_bg, labelcolor=text_color)

    fig.suptitle(
        f"Global Arbitrage Signal Engine -- {pair_key} Backtest Report",
        color=text_color, fontsize=13, fontweight="bold", y=0.98
    )

    save_path = save_path or str(RESULTS_DIR / f"backtest_{pair_key}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=dark_bg)
    plt.close(fig)
    log.info(f"Chart saved -> {save_path}")
    return save_path