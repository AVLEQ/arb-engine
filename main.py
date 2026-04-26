"""
main.py — Global Arbitrage Signal Engine
Master runner: ingest → signals → backtest → ML filter → risk → report

Usage:
    python main.py                     # full pipeline, primary pair
    python main.py --pair GLD_IGLD     # specific pair
    python main.py --all-pairs         # run all configured pairs
    python main.py --skip-ingest       # use cached Parquet data
    python main.py --no-ml             # skip ML filter (pure z-score)
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.config import CFG
from data.ingest import ingest_all, load_pair, load_all_pairs
from backtester.signals import build_signals
from backtester.engine import Backtester
from backtester.metrics import compute_metrics, print_metrics, plot_results
from backtester.ml_filter import MLConfidenceModel
from risk.framework import RegimeMonitor, monte_carlo_sharpe, sensitivity_grid, position_size_summary

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "results" / "run.log", mode="w"),
    ],
)
log = logging.getLogger("main")

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PAIR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pair(
    key: str,
    raw_df: pd.DataFrame,
    use_ml: bool = True,
) -> dict:
    """
    Full analysis pipeline for one ETF pair.
    Returns summary dict with metrics for all pairs comparison.
    """
    t0 = time.time()
    log.info(f"\n{'═'*60}")
    log.info(f"  PAIR: {key}")
    log.info(f"{'═'*60}")

    # ── Step 1: Build signals ─────────────────────────────────────────────────
    log.info("[1/5] Building signals...")
    signals_df = build_signals(raw_df, verbose=True)

    # ── Step 2: ML confidence model ───────────────────────────────────────────
    ml_confidence = None
    ml_model = None

    if use_ml:
        log.info("[2/5] Training ML confidence model...")
        ml_model = MLConfidenceModel()
        ml_model.train(signals_df)

        if ml_model.is_trained:
            ml_confidence = ml_model.predict_confidence(signals_df)
            log.info(f"  ML model trained. Train AUC: {ml_model.train_auc:.3f} | Test AUC: {ml_model.test_auc:.3f}")
            log.info(f"  Top features:\n{ml_model.feature_importance().head(5).to_string()}")
        else:
            log.warning("  ML training skipped — insufficient data")
    else:
        log.info("[2/5] ML disabled — using pure z-score signals")

    # ── Step 3: Backtest (with ML filter) ─────────────────────────────────────
    log.info("[3/5] Running backtest...")
    bt = Backtester(key, capital=CFG.sizing.starting_capital)
    results = bt.run(signals_df)
    trade_log = bt.get_trade_log()

    # Also run without ML for ablation comparison
    bt_base = Backtester(f"{key}_base", capital=CFG.sizing.starting_capital)
    results_base = bt_base.run(signals_df, ml_confidence=None)
    trade_log_base = bt_base.get_trade_log()

    # ── Step 4: Metrics ───────────────────────────────────────────────────────
    log.info("[4/5] Computing metrics...")
    metrics     = compute_metrics(results, trade_log)
    metrics_base= compute_metrics(results_base, trade_log_base)

    print_metrics(metrics, pair_key=key)

    # Walk-forward Sharpe (sanity: not all in-sample)
    half = len(results) // 2
    wf_ret   = results["returns"].iloc[half:]
    wf_sharpe = wf_ret.mean() / max(wf_ret.std(), 1e-8) * np.sqrt(252)
    log.info(f"  Walk-forward Sharpe (2nd half): {wf_sharpe:.3f}")

    # Monte Carlo
    log.info("  Running Monte Carlo bootstrap (2000 sims)...")
    mc = monte_carlo_sharpe(results["returns"])
    log.info(
        f"  MC Sharpe: mean={mc['mean_sharpe']:.3f} | "
        f"p5={mc['p5_sharpe']:.3f} | p95={mc['p95_sharpe']:.3f} | "
        f"P(Sharpe>0)={mc['pct_positive']:.1f}%"
    )

    # ML ablation
    if ml_model and ml_model.is_trained:
        ablation = ml_model.ablation_study(results, results_base)
        log.info(
            f"  ML ablation — Sharpe with ML: {ablation['sharpe_with_ml']} | "
            f"without: {ablation['sharpe_without_ml']} | "
            f"improvement: {ablation['sharpe_improvement']}%"
        )

    # Position size example (for documentation)
    ps = position_size_summary(
        capital=CFG.sizing.starting_capital,
        edge_bps=12.0,
        spread_vol=0.015,
        realized_vol=0.12,
        vix=18.0,
        regime_multiplier=1.0,
    )
    log.info(f"  Example position size (12bps edge): ${ps['notional_usd']:,.0f} ({ps['pct_of_capital']:.1f}% of capital)")

    # Regime check
    regime = RegimeMonitor()
    regime_state = regime.update(results, signals_df)
    log.info(f"  Final regime state: {regime_state} | Flags: {regime.flags}")

    # ── Step 5: Plots & exports ───────────────────────────────────────────────
    log.info("[5/5] Generating outputs...")

    # Backtest chart
    chart_path = plot_results(
        results, trade_log, signals_df,
        pair_key=key,
        save_path=str(RESULTS_DIR / f"backtest_{key}.png"),
    )
    log.info(f"  Chart saved → {chart_path}")

    # Trade log CSV
    if len(trade_log) > 0:
        tl_path = RESULTS_DIR / f"trades_{key}.csv"
        trade_log.to_csv(tl_path, index=False)
        log.info(f"  Trade log → {tl_path} ({len(trade_log)} trades)")

    # Sensitivity grid (heatmap of Sharpe vs z_threshold × lookback)
    log.info("  Running sensitivity grid (z_threshold × lookback)...")
    try:
        sens = sensitivity_grid(signals_df)
        sens_path = RESULTS_DIR / f"sensitivity_{key}.csv"
        sens.to_csv(sens_path)
        log.info(f"  Sensitivity grid → {sens_path}")
    except Exception as e:
        log.warning(f"  Sensitivity grid failed: {e}")

    # Summary JSON
    summary = {
        "pair": key,
        "metrics_ml":   metrics,
        "metrics_base": metrics_base,
        "monte_carlo":  mc,
        "walk_forward_sharpe": round(wf_sharpe, 3),
        "regime_state": regime_state,
        "regime_flags": regime.flags,
        "position_size_example": ps,
        "ml_trained": ml_model.is_trained if ml_model else False,
        "runtime_sec": round(time.time() - t0, 1),
    }
    summary_path = RESULTS_DIR / f"summary_{key}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Summary JSON → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PAIR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_all_pairs(datasets: dict, use_ml: bool = True) -> pd.DataFrame:
    """Run pipeline for all pairs and produce a comparison table."""
    all_summaries = []
    for key, df in datasets.items():
        try:
            s = run_pair(key, df, use_ml=use_ml)
            all_summaries.append({
                "pair": key,
                "sharpe_ml":       s["metrics_ml"].get("sharpe_ratio"),
                "sharpe_base":     s["metrics_base"].get("sharpe_ratio"),
                "max_dd_pct":      s["metrics_ml"].get("max_drawdown_pct"),
                "ann_return_pct":  s["metrics_ml"].get("ann_return_pct"),
                "win_rate_pct":    s["metrics_ml"].get("win_rate_pct"),
                "total_trades":    s["metrics_ml"].get("total_trades"),
                "avg_hold_days":   s["metrics_ml"].get("avg_hold_days"),
                "mc_p5_sharpe":    s["monte_carlo"]["p5_sharpe"],
                "wf_sharpe":       s["walk_forward_sharpe"],
                "regime":          s["regime_state"],
            })
        except Exception as e:
            log.error(f"  Pair {key} failed: {e}", exc_info=True)

    comp = pd.DataFrame(all_summaries)
    comp_path = RESULTS_DIR / "all_pairs_comparison.csv"
    comp.to_csv(comp_path, index=False)

    log.info("\n" + "═" * 70)
    log.info("  ALL PAIRS COMPARISON")
    log.info("═" * 70)
    log.info(comp.to_string(index=False))
    log.info("═" * 70)
    log.info(f"  Saved → {comp_path}")

    return comp


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Global Arbitrage Signal Engine")
    p.add_argument("--pair",         default=None,  help="Specific pair key e.g. GLD_IGLD")
    p.add_argument("--all-pairs",    action="store_true", help="Run all configured pairs")
    p.add_argument("--skip-ingest",  action="store_true", help="Use cached Parquet data")
    p.add_argument("--no-ml",        action="store_true", help="Disable ML filter")
    p.add_argument("--start",        default=None,  help="Override start date YYYY-MM-DD")
    p.add_argument("--end",          default=None,  help="Override end date YYYY-MM-DD")
    return p.parse_args()


def main():
    args = parse_args()
    use_ml = not args.no_ml

    log.info("=" * 60)
    log.info("  GLOBAL ARBITRAGE SIGNAL ENGINE")
    log.info(f"  Period:  {CFG.data.start_date} → {CFG.data.end_date}")
    log.info(f"  Capital: ${CFG.sizing.starting_capital:,.0f}")
    log.info(f"  ML filter: {'ON' if use_ml else 'OFF'}")
    log.info("=" * 60)

    # ── Data ingestion ────────────────────────────────────────────────────────
    if args.skip_ingest:
        log.info("Loading cached data from Parquet...")
        try:
            datasets = load_all_pairs()
        except Exception as e:
            log.error(f"Cache load failed: {e}. Run without --skip-ingest first.")
            sys.exit(1)
    else:
        log.info("Downloading fresh data...")
        start = args.start or CFG.data.start_date
        end   = args.end   or CFG.data.end_date
        datasets = ingest_all(start=start, end=end, save=True)

    if not datasets:
        log.error("No data loaded. Check tickers and network connection.")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.all_pairs:
        run_all_pairs(datasets, use_ml=use_ml)

    elif args.pair:
        if args.pair not in datasets:
            log.error(f"Pair '{args.pair}' not in datasets. Available: {list(datasets.keys())}")
            sys.exit(1)
        run_pair(args.pair, datasets[args.pair], use_ml=use_ml)

    else:
        # Default: run primary pair
        primary_key = f"{CFG.pairs[0][0]}_{CFG.pairs[0][1]}"
        if primary_key not in datasets:
            primary_key = list(datasets.keys())[0]
        log.info(f"Running primary pair: {primary_key}")
        run_pair(primary_key, datasets[primary_key], use_ml=use_ml)

    log.info("\n✓ Pipeline complete. Results in: results/")


if __name__ == "__main__":
    main()
