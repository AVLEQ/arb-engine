"""
backtester/engine.py — Backtesting Engine  [FIXED v2]
Key fixes:
  1. ml_confidence properly applied to filter signals
  2. max_hold_days enforced with time-stop exit
  3. Position sizing uses spread_std properly
  4. Only one open trade per pair at a time (no overlap)
"""

import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("backtester")

ADV_BY_TICKER = {
    "QQQ": 20_000_000_000, "QQQM": 500_000_000,
    "SPY": 40_000_000_000, "IVV": 5_000_000_000,
    "GLD": 1_500_000_000,  "IAU": 500_000_000,
    "IEF": 1_000_000_000,  "GOVT": 200_000_000,
    "EEM": 2_000_000_000,  "VWO": 500_000_000,
}
DEFAULT_ADV = 500_000_000


@dataclass
class Trade:
    pair: str
    direction: int
    entry_date: pd.Timestamp
    entry_price1: float
    entry_price2: float
    entry_zscore: float
    entry_fx: float
    notional: float
    units1: float
    units2: float
    hedge_ratio: float
    ticker1: str = ""
    ticker2: str = ""

    exit_date: Optional[pd.Timestamp] = None
    exit_price1: float = None
    exit_price2: float = None
    exit_zscore: float = None
    exit_reason: str = ""
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    hold_days: int = 0
    is_closed: bool = False

    def close(self, exit_date, exit_price1, exit_price2, exit_zscore, exit_reason,
              cost_fn, slippage_fn):
        self.exit_date = exit_date
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2
        self.exit_zscore = exit_zscore
        self.exit_reason = exit_reason
        self.hold_days = max(1, (exit_date - self.entry_date).days)
        self.is_closed = True

        pnl1 = self.direction * self.units1 * (exit_price1 - self.entry_price1)
        pnl2 = -self.direction * self.units2 * (exit_price2 - self.entry_price2)
        self.gross_pnl = pnl1 + pnl2

        entry_costs = cost_fn(self.notional) + slippage_fn(self.notional, self.ticker1, self.ticker2)
        exit_costs  = cost_fn(self.notional) + slippage_fn(self.notional, self.ticker1, self.ticker2)
        self.total_costs = entry_costs + exit_costs
        self.net_pnl = self.gross_pnl - self.total_costs

    def to_dict(self):
        return {
            "pair": self.pair,
            "direction": "LONG_L1" if self.direction == 1 else "SHORT_L1",
            "entry_date": str(self.entry_date.date()),
            "exit_date": str(self.exit_date.date()) if self.exit_date else None,
            "entry_price1": round(self.entry_price1, 4),
            "entry_price2": round(self.entry_price2, 4),
            "exit_price1": round(self.exit_price1, 4) if self.exit_price1 else None,
            "exit_price2": round(self.exit_price2, 4) if self.exit_price2 else None,
            "entry_zscore": round(self.entry_zscore, 4),
            "exit_zscore": round(self.exit_zscore, 4) if self.exit_zscore else None,
            "notional": self.notional,
            "hold_days": self.hold_days,
            "gross_pnl": round(self.gross_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "net_pnl": round(self.net_pnl, 2),
            "exit_reason": self.exit_reason,
        }


def total_cost_usd(notional: float) -> float:
    c = CFG.costs
    us_cost = notional * (c.us_commission_pct + c.us_bid_ask_pct + c.us_sec_fee_pct) * 2
    fx_cost = notional * c.fx_spread_pct
    return us_cost + fx_cost


def slippage_usd(notional: float, ticker1: str = "", ticker2: str = "") -> float:
    c = CFG.costs
    adv1 = ADV_BY_TICKER.get(ticker1, DEFAULT_ADV)
    adv2 = ADV_BY_TICKER.get(ticker2, DEFAULT_ADV)
    avg_adv = (adv1 + adv2) / 2
    fraction = notional / max(avg_adv, 1)
    impact_bps = c.lambda_impact * np.sqrt(fraction)
    return notional * impact_bps / 10000


class PositionSizer:
    def __init__(self, capital):
        self.capital = capital

    def compute(self, edge, spread_vol, realized_vol, vix):
        if spread_vol <= 0 or edge <= 0:
            return 0.0
        base = edge / (spread_vol ** 2 + 1e-6)
        size = base * CFG.sizing.kelly_fraction * self.capital
        if realized_vol > 0:
            size *= min(2.0, max(0.25, CFG.sizing.target_vol_annual / realized_vol))
        if vix > CFG.risk.vix_high:
            size *= CFG.risk.size_reduction_high_vol
        return min(size, CFG.sizing.max_position_pct * self.capital)


class Portfolio:
    def __init__(self, capital):
        self.capital = capital
        self.equity_curve = [(None, capital)]
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.peak = capital
        self.drawdown = 0

    @property
    def equity(self):
        return self.equity_curve[-1][1]

    def update(self, date, pnl):
        new_eq = self.equity + pnl
        self.equity_curve.append((date, new_eq))
        self.peak = max(self.peak, new_eq)
        self.drawdown = (self.peak - new_eq) / self.peak


class Backtester:
    def __init__(self, pair_key, capital=None):
        self.pair_key = pair_key
        self.portfolio = Portfolio(capital or CFG.sizing.starting_capital)
        self.sizer = PositionSizer(self.portfolio.capital)

        # Extract ticker names from pair key
        parts = pair_key.replace("_base", "").split("_")
        self.ticker1 = parts[0] if len(parts) >= 1 else ""
        self.ticker2 = parts[1] if len(parts) >= 2 else ""

    def run(self, signals_df: pd.DataFrame, ml_confidence: pd.Series = None):
        df = signals_df.copy().dropna(subset=["zscore"])
        port = self.portfolio
        records = []
        max_hold = CFG.signal.max_hold_days

        for date, row in df.iterrows():
            pnl = 0

            # ── Close trades ──────────────────────────────────────────────────
            for trade in port.open_trades[:]:
                hold = (date - trade.entry_date).days
                should_exit = row["exit_flag"] or (hold >= max_hold)
                exit_reason = "reversion" if row["exit_flag"] else "time_stop"

                if should_exit:
                    trade.close(
                        date, row["close1"], row["close2"], row["zscore"],
                        exit_reason, total_cost_usd, slippage_usd
                    )
                    pnl += trade.net_pnl
                    port.open_trades.remove(trade)
                    port.closed_trades.append(trade)

            port.update(date, pnl)

            signal = row.get("raw_signal", 0)

            # ── ML filter ─────────────────────────────────────────────────────
            if signal != 0 and ml_confidence is not None:
                conf = ml_confidence.get(date, 0.5) if hasattr(ml_confidence, 'get') else 0.5
                try:
                    conf = float(ml_confidence.loc[date])
                except Exception:
                    conf = 0.5
                if conf < CFG.ml.min_confidence:
                    signal = 0   # ML says skip

            # ── Only one position at a time per pair ──────────────────────────
            if signal != 0 and len(port.open_trades) == 0:
                size = self.sizer.compute(
                    edge=abs(row["zscore"]),
                    spread_vol=max(row.get("spread_std", 0.01), 1e-6),
                    realized_vol=0.15,
                    vix=row.get("vix", 15)
                )

                if size > 10_000:
                    units1 = size / row["close1"]
                    units2 = units1 * row.get("hedge_ratio", 1.0)

                    trade = Trade(
                        pair=self.pair_key,
                        direction=signal,
                        entry_date=date,
                        entry_price1=row["close1"],
                        entry_price2=row["close2"],
                        entry_zscore=row["zscore"],
                        entry_fx=row.get("fx", 83),
                        notional=size,
                        units1=units1,
                        units2=units2,
                        hedge_ratio=row.get("hedge_ratio", 1.0),
                        ticker1=self.ticker1,
                        ticker2=self.ticker2,
                    )
                    port.open_trades.append(trade)

            records.append({
                "date": date,
                "equity": port.equity,
                "drawdown": port.drawdown,
            })

        results = pd.DataFrame(records).set_index("date")
        results["returns"] = results["equity"].pct_change().fillna(0)
        log.info(f"Backtest complete: {len(port.closed_trades)} trades, equity {port.equity:.2f}")
        return results

    def get_trade_log(self):
        return pd.DataFrame([t.to_dict() for t in self.portfolio.closed_trades])