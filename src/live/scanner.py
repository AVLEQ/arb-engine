"""
live/scanner.py — Live Signal Generator
Runs continuously, scans for arbitrage signals in real-time, and logs to JSONL.

Required deliverable: runs stably for 48 hours.
Each log entry: timestamp, instrument, direction, theoretical edge, confidence score.

Usage:
    python live/scanner.py                   # scan all configured pairs
    python live/scanner.py --pair GLD_IGLD   # specific pair
    python live/scanner.py --interval 60     # poll every 60 seconds (default)
    python live/scanner.py --dry-run         # print signals, don't write log
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config import CFG
from backtester.signals import compute_spread_zscore, generate_raw_signals, add_momentum_features
from backtester.metrics import edge_decay_curve

# ── Logging ───────────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH  = RESULTS_DIR / CFG.live.log_filename
STAT_PATH = RESULTS_DIR / "scanner_stats.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "scanner.log", mode="a"),
    ],
)
log = logging.getLogger("scanner")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent(ticker: str, days: int = 120) -> pd.DataFrame:
    """
    Fetch the last N calendar days of daily OHLCV.
    Uses yfinance. Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.warning(f"  yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_pair_live(ticker1: str, ticker2: str, days: int = 120) -> pd.DataFrame:
    """
    Build a live pair DataFrame from recent yfinance data.
    Same structure as the ingested training data.
    """
    df1 = fetch_recent(ticker1, days)
    df2 = fetch_recent(ticker2, days)
    fx  = fetch_recent("USDINR=X", days)
    vix = fetch_recent("^VIX", days)

    if df1.empty or df2.empty:
        return pd.DataFrame()

    common = df1.index.intersection(df2.index)
    df1 = df1.loc[common]
    df2 = df2.loc[common]

    fx_series  = fx["close"].reindex(common).ffill().bfill() if not fx.empty else pd.Series(83.0, index=common)
    vix_series = vix["close"].reindex(common).ffill().bfill() if not vix.empty else pd.Series(18.0, index=common)

    pair = pd.DataFrame({
        "close1":    df1["close"],
        "close2":    df2["close"],
        "close2_usd":df2["close"],
        "spread":    np.log(df1["close"]) - np.log(df2["close"]),
        "return1":   np.log(df1["close"] / df1["close"].shift(1)),
        "return2":   np.log(df2["close"] / df2["close"].shift(1)),
        "volume1":   df1["volume"],
        "volume2":   df2["volume"],
        "vol_ratio": df1["volume"] / (df2["volume"] + 1),
        "fx":        fx_series,
        "vix":       vix_series,
    }, index=common)

    # Volume rank (rolling)
    pair["vol_rank1"]   = pair["volume1"].rolling(60).rank(pct=True)
    pair["vol_rank2"]   = pair["volume2"].rolling(60).rank(pct=True)
    pair["vol_rank_min"]= pair[["vol_rank1", "vol_rank2"]].min(axis=1)

    return pair.dropna(subset=["close1", "close2"])


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE CONFIDENCE SCORE (live — no trained model required)
# ─────────────────────────────────────────────────────────────────────────────

def live_confidence_score(row: pd.Series, df: pd.DataFrame, i: int) -> float:
    """
    Heuristic confidence score for live signals (0–1).
    Used when the full ML model isn't available in the live context.

    Factors:
    - Magnitude of z-score (higher = more extreme = better)
    - Volume: is today's volume above median?
    - FX vol: low FX vol = better for cross-listed arb
    - Spread momentum: is spread accelerating back toward mean?
    - VIX: lower VIX = calmer market = better for mean-reversion
    """
    score = 0.5  # base

    # Factor 1: Z-score magnitude (normalize to 0–0.2 contribution)
    z = abs(row.get("zscore", 2.0))
    score += min((z - CFG.signal.zscore_entry) * 0.05, 0.15)

    # Factor 2: Volume above 60th percentile
    vr = row.get("vol_rank_min", 0.5)
    if vr > 0.6:
        score += 0.10
    elif vr < 0.3:
        score -= 0.10

    # Factor 3: FX vol (lower is better)
    fx_vol = row.get("fx_vol_20d", 0.06)
    if fx_vol < 0.04:
        score += 0.05
    elif fx_vol > 0.10:
        score -= 0.10

    # Factor 4: Spread momentum pointing toward mean reversion
    mom = row.get("spread_mom_5d", 0.0)
    if (row.get("zscore", 0) > 0 and mom < 0) or (row.get("zscore", 0) < 0 and mom > 0):
        score += 0.08  # momentum consistent with reversion
    else:
        score -= 0.05  # diverging

    # Factor 5: VIX regime
    vix = row.get("vix", 18.0)
    if vix < 15:
        score += 0.05
    elif vix > 25:
        score -= 0.15
    elif vix > 20:
        score -= 0.05

    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# THEORETICAL EDGE (live)
# ─────────────────────────────────────────────────────────────────────────────

def compute_live_edge(row: pd.Series) -> float:
    """
    Theoretical edge in basis points.
    = |zscore| × spread_std × 10000 − total_round_trip_cost_bps
    """
    z   = abs(row.get("zscore", 0.0))
    std = row.get("spread_std", 0.01)
    gross_bps = z * std * 10_000

    c = CFG.costs
    cost_bps = (
        (c.us_commission_pct + c.us_bid_ask_pct) * 2
        + (c.in_stt_pct + c.in_bid_ask_pct + c.in_brokerage_pct) * 2
        + c.fx_spread_pct * 2
    ) * 10_000

    net_bps = max(gross_bps - cost_bps, 0.0)
    return round(net_bps, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class SignalLogger:
    def __init__(self, log_path: Path, dry_run: bool = False):
        self.log_path = log_path
        self.dry_run  = dry_run
        self.total_signals = 0
        self.total_scans   = 0
        self.errors        = 0
        self.start_time    = datetime.now(timezone.utc)

    def write(self, entry: dict):
        """Append one JSONL record to the signal log."""
        self.total_signals += 1
        if self.dry_run:
            log.info(f"  [DRY RUN] Signal: {json.dumps(entry)}")
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_stats(self):
        stats = {
            "start_time":     self.start_time.isoformat(),
            "last_updated":   datetime.now(timezone.utc).isoformat(),
            "uptime_hours":   round((datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600, 2),
            "total_scans":    self.total_scans,
            "total_signals":  self.total_signals,
            "errors":         self.errors,
            "signals_per_hr": round(
                self.total_signals / max((datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600, 0.001), 2
            ),
        }
        with open(STAT_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# ONE SCAN CYCLE
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(ticker1: str, ticker2: str, desc: str, logger: SignalLogger) -> int:
    """
    Fetch latest data for one pair, compute signals, log any active signals.
    Returns number of signals found.
    """
    pair_key = f"{ticker1}_{ticker2}"
    signals_found = 0

    try:
        # Fetch recent data
        df = fetch_pair_live(ticker1, ticker2, days=120)
        if df.empty or len(df) < CFG.signal.zscore_lookback + 10:
            log.warning(f"  {pair_key}: insufficient data ({len(df)} rows)")
            return 0

        # Compute signals
        df = compute_spread_zscore(df)
        df = add_momentum_features(df)
        df = generate_raw_signals(df)

        # Latest row
        latest = df.iloc[-1]
        ts = datetime.now(timezone.utc).isoformat()

        # Always log current state (even if no signal)
        z = latest.get("zscore", np.nan)
        if pd.isna(z):
            log.info(f"  {pair_key}: z-score NaN — skipping")
            return 0

        edge_bps   = compute_live_edge(latest)
        confidence = live_confidence_score(latest, df, len(df) - 1)
        raw_signal = int(latest.get("raw_signal", 0))

        entry = {
            "timestamp":   ts,
            "pair":        pair_key,
            "description": desc,
            "ticker1":     ticker1,
            "ticker2":     ticker2,
            "price1":      round(float(latest["close1"]), 4),
            "price2":      round(float(latest["close2"]), 4),
            "zscore":      round(float(z), 4),
            "spread_ols":  round(float(latest.get("spread_ols", np.nan)), 6),
            "spread_std":  round(float(latest.get("spread_std", np.nan)), 6),
            "hedge_ratio": round(float(latest.get("hedge_ratio", 1.0)), 4),
            "edge_bps":    edge_bps,
            "confidence":  confidence,
            "signal":      raw_signal,
            "direction":   "LONG_LEG1" if raw_signal == 1 else ("SHORT_LEG1" if raw_signal == -1 else "FLAT"),
            "vix":         round(float(latest.get("vix", np.nan)), 1) if not pd.isna(latest.get("vix", np.nan)) else None,
            "fx_usdinr":   round(float(latest.get("fx", np.nan)), 4)  if not pd.isna(latest.get("fx", np.nan)) else None,
            "vol_rank_min":round(float(latest.get("vol_rank_min", np.nan)), 3) if not pd.isna(latest.get("vol_rank_min", np.nan)) else None,
            "is_cointegrated": bool(latest.get("is_cointegrated", False)),
            "coint_pval":  round(float(latest.get("coint_pval", 1.0)), 4) if not pd.isna(latest.get("coint_pval", np.nan)) else None,
            "signal_valid":bool(latest.get("signal_valid", False)),
            "data_rows":   len(df),
        }

        # Always write to log (allows reconstruction of full state timeline)
        logger.write(entry)

        if raw_signal != 0:
            signals_found += 1
            direction_str = "LONG LEG1 / SHORT LEG2" if raw_signal == 1 else "SHORT LEG1 / LONG LEG2"
            log.info(
                f"  *** SIGNAL *** {pair_key} | {direction_str} | "
                f"z={z:.3f} | edge={edge_bps:.1f}bps | conf={confidence:.3f}"
            )
        else:
            log.info(
                f"  {pair_key}: FLAT | z={z:.3f} | edge={edge_bps:.1f}bps | conf={confidence:.3f}"
            )

    except Exception as e:
        logger.errors += 1
        log.error(f"  {pair_key} scan error: {e}")
        log.debug(traceback.format_exc())

    return signals_found


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCAN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Arbitrage Signal Scanner")
    parser.add_argument("--pair",     default=None, help="Single pair key e.g. GLD_IGLD")
    parser.add_argument("--interval", type=int, default=CFG.live.poll_interval_sec,
                        help="Poll interval in seconds (default 60)")
    parser.add_argument("--dry-run",  action="store_true", help="Print signals, don't write log")
    parser.add_argument("--hours",    type=float, default=48.0, help="Run for N hours (default 48)")
    args = parser.parse_args()

    # ── Select pairs ──────────────────────────────────────────────────────────
    if args.pair:
        parts = args.pair.split("_")
        pairs = [(parts[0], parts[1], f"Live scan: {args.pair}")]
    else:
        pairs = [(t1, t2, desc) for t1, t2, desc in CFG.pairs]

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    running = [True]
    def _shutdown(sig, frame):
        log.info(f"\nShutdown signal received ({sig}). Stopping after current cycle...")
        running[0] = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Init logger ───────────────────────────────────────────────────────────
    logger = SignalLogger(LOG_PATH, dry_run=args.dry_run)
    max_runtime_sec = args.hours * 3600

    log.info("=" * 60)
    log.info("  LIVE ARBITRAGE SIGNAL SCANNER")
    log.info(f"  Pairs:    {[f'{t1}/{t2}' for t1, t2, _ in pairs]}")
    log.info(f"  Interval: {args.interval}s")
    log.info(f"  Duration: {args.hours}h")
    log.info(f"  Log:      {LOG_PATH}")
    log.info(f"  Dry run:  {args.dry_run}")
    log.info("=" * 60)

    scan_num = 0
    while running[0]:
        elapsed = (datetime.now(timezone.utc) - logger.start_time).total_seconds()

        if elapsed >= max_runtime_sec:
            log.info(f"Reached {args.hours}h runtime limit. Stopping.")
            break

        scan_num += 1
        logger.total_scans += 1
        hrs_remaining = (max_runtime_sec - elapsed) / 3600

        log.info(f"\n{'─'*50}")
        log.info(f"Scan #{scan_num} | Elapsed: {elapsed/3600:.2f}h | Remaining: {hrs_remaining:.2f}h")
        log.info(f"{'─'*50}")

        total_signals = 0
        for ticker1, ticker2, desc in pairs:
            if not running[0]:
                break
            n = scan_pair(ticker1, ticker2, desc, logger)
            total_signals += n

        # Stats update
        stats = logger.save_stats()
        log.info(
            f"Scan #{scan_num} complete | "
            f"Signals this scan: {total_signals} | "
            f"Total signals: {stats['total_signals']} | "
            f"Errors: {stats['errors']}"
        )

        # Watchdog: warn if no writes for a long time
        if scan_num > 1 and logger.total_signals == 0 and scan_num > 5:
            log.warning("No signals logged yet — check data connectivity")

        # Sleep until next cycle (interruptible)
        sleep_remaining = args.interval
        while sleep_remaining > 0 and running[0]:
            time.sleep(min(1.0, sleep_remaining))
            sleep_remaining -= 1.0

    # Final stats
    stats = logger.save_stats()
    log.info("\n" + "=" * 60)
    log.info("  SCANNER STOPPED")
    log.info(f"  Total scans:   {stats['total_scans']}")
    log.info(f"  Total signals: {stats['total_signals']}")
    log.info(f"  Errors:        {stats['errors']}")
    log.info(f"  Uptime:        {stats['uptime_hours']:.2f}h")
    log.info(f"  Signal log:    {LOG_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
