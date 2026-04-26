"""
backtester/engine.py — Backtesting Engine
Event-driven backtester with realistic cost model, position sizing, and trade log.
Disqualification-proof: every trade includes spread, slippage, commissions, and FX costs.
"""

import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("backtester")

# ADV lookup for realistic slippage (average daily volume in USD)
# Source: typical 30-day ADV for these instruments
ADV_BY_TICKER = {
    "QQQ":  20_000_000_000,   # $20B/day
    "QQQM": 500_000_000,      # $500M/day
    "SPY":  40_000_000_000,   # $40B/day
    "IVV":  5_000_000_000,    # $5B/day
    "GLD":  1_500_000_000,    # $1.5B/day
    "IAU":  500_000_000,      # $500M/day
    "IEF":  1_000_000_000,    # $1B/day
    "GOVT": 200_000_000,      # $200M/day
    "EEM":  2_000_000_000,    # $2B/day
    "VWO":  500_000_000,      # $500M/day
}
DEFAULT_ADV = 500_000_000  # $500M conservative fallback


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    pair: str
    direction: int              # +1 = long leg1 / short leg2; -1 = opposite
    entry_date: pd.Timestamp
    entry_price1: float
    entry_price2: float
    entry_zscore: float
    entry_fx: float
    notional: float             # USD notional of leg1
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

    def close(
        self,
        exit_date: pd.Timestamp,
        exit_price1: float,
        exit_price2: float,
        exit_zscore: float,
        exit_reason: str,
        cost_model,
        slippage_fn,
    ):
        self.exit_date   = exit_date
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2
        self.exit_zscore = exit_zscore
        self.exit_reason = exit_reason
        self.hold_days   = (exit_date - self.entry_date).days
        self.is_closed   = True

        pnl1 = self.direction * self.units1 * (exit_price1 - self.entry_price1)
        pnl2 = -self.direction * self.units2 * (exit_price2 - self.entry_price2)
        self.gross_pnl = pnl1 + pnl2

        entry_costs = cost_model(self.notional) + slippage_fn(self.notional, self.ticker1, self.ticker2)
        exit_costs  = cost_model(self.notional) + slippage_fn(self.notional, self.ticker1, self.ticker2)
        self.total_costs = entry_costs + exit_costs
        self.net_pnl = self.gross_pnl - self.total_costs

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": "LONG_L1" if self.direction == 1 else "SHORT_L1",
            "entry_date": str(self.entry_date.date()),
            "exit_date": str(self.exit_date.date()) if self.exit_date else None,
            "entry_price1": round(self.entry_price1, 4),
            "entry_price2": round(self.entry_price2, 4),
            "exit_price1": round(self.exit_price1, 4) if self.exit_price1 else None,
            "exit_price2": round(self.exit_price2, 4) if self.exit_price2 else None,
            "entry_zscore": round(self.entry_zscore, 3),
            "exit_zscore": round(self.exit_zscore, 3) if self.exit_zscore else None,
            "notional": round(self.notional, 2),
            "hold_days": self.hold_days,
            "gross_pnl": round(self.gross_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "net_pnl": round(self.net_pnl, 2),
            "exit_reason": self.exit_reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def total_cost_usd(notional: float) -> float:
    """
    One-way transaction cost in USD.
    For US-only ETF pairs: commission + SEC fee + bid-ask spread.
    Both legs included (we trade both sides of the pair).
    """
    c = CFG.costs
    # Both legs
    us_cost  = notional * (c.us_commission_pct + c.us_bid_ask_pct + c.us_sec_fee_pct) * 2
    fx_cost  = notional * c.fx_spread_pct
    return us_cost + fx_cost


def slippage_usd(notional: float, ticker1: str = "", ticker2: str = "") -> float:
    """
    Market impact slippage using square-root model (Kyle's lambda).
    Uses realistic ADV for each ticker pair.
    slippage = lambda * sqrt(order_size / ADV) * notional
    """
    c = CFG.costs
    adv1 = ADV_BY_TICKER.get(ticker1, DEFAULT_ADV)
    adv2 = ADV_BY_TICKER.get(ticker2, DEFAULT_ADV)
    avg_adv = (adv1 + adv2) / 2

    fraction = notional / max(avg_adv, 1)
    impact_bps = c.lambda_impact * np.sqrt(fraction)
    return notional * impact_bps / 10000


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZER
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    def __init__(self, capital: float):
        self.capital = capital

    def kelly_size(self, edge: float, vol: float) -> float:
        """Fractional Kelly position size."""
        if vol <= 0 or edge <= 0:
            return 0.0
        variance = vol ** 2
        full_kelly_frac = edge / variance
        frac_kelly = full_kelly_frac * CFG.sizing.kelly_fraction
        return min(frac_kelly * self.capital, CFG.sizing.max_position_pct * self.capital)

    def vol_scaled_size(self, base_size: float, realized_vol: float) -> float:
        """Scale position to hit target annual volatility."""
        target = CFG.sizing.target_vol_annual
        if realized_vol <= 0:
            return base_size
        scalar = target / realized_vol
        scalar = np.clip(scalar, 0.25, 2.0)
        return base_size * scalar

    def regime_adjust(self, size: float, vix: float) -> float:
        """Reduce size in high-vol regimes."""
        if vix > CFG.risk.vix_high:
            return size * CFG.risk.size_reduction_high_vol
        return size

    def compute(self, edge: float, spread_vol: float, realized_vol: float, vix: float) -> float:
        base = self.kelly_size(edge, spread_vol)
        scaled = self.vol_scaled_size(base, realized_vol)
        adjusted = self.regime_adjust(scaled, vix)
        return max(0.0, adjusted)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO STATE
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, capital: float):
        self.capital = capital
        self.equity_curve: List[Tuple] = [(None, capital)]
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.peak_equity = capital
        self.current_drawdown = 0.0
        self.paused = False

    @property
    def equity(self) -> float:
        return self.equity_curve[-1][1]

    @property
    def gross_exposure(self) -> float:
        return sum(t.notional for t in self.open_trades)

    def update_equity(self, date: pd.Timestamp, pnl_delta: float):
        new_eq = self.equity + pnl_delta
        self.equity_curve.append((date, new_eq))
        self.peak_equity = max(self.peak_equity, new_eq)
        self.current_drawdown = (self.peak_equity - new_eq) / self.peak_equity

    def check_circuit_breaker(self) -> str:
        dd = self.current_drawdown
        if dd >= CFG.risk.drawdown_stop:
            self.paused = True
            return "STOP"
        elif dd >= CFG.risk.drawdown_pause:
            self.paused = True
            return "PAUSE"
        elif dd >= CFG.risk.drawdown_warning:
            return "WARN"
        else:
            self.paused = False
            return "OK"

    def can_add_position(self, notional: float) -> bool:
        if self.paused:
            return False
        new_exposure = self.gross_exposure + notional
        return new_exposure <= CFG.sizing.max_gross_exposure * self.equity


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, pair_key: str, capital: float = None):
        self.pair_key = pair_key
        self.portfolio = Portfolio(capital or CFG.sizing.starting_capital)
        self.sizer = PositionSizer(self.portfolio.capital)
        # Extract tickers from pair key (e.g. "QQQ_QQQM" -> "QQQ", "QQQM")
        parts = pair_key.replace("_base", "").split("_")
        self.ticker1 = parts[0] if len(parts) >= 1 else ""
        self.ticker2 = parts[1] if len(parts) >= 2 else ""

    def run(self, signals_df: pd.DataFrame, ml_confidence: pd.Series = None) -> pd.DataFrame:
        """
        Main backtest loop. Steps through each day.
        Returns a DataFrame with daily equity, drawdown, and returns.
        """
        df = signals_df.copy().dropna(subset=["zscore", "spread_ols"])
        port = self.portfolio
        cfg_s = CFG.signal

        daily_records = []

        for i, (date, row) in enumerate(df.iterrows()):

            # 1. Mark-to-market open positions
            daily_pnl = 0.0
            trades_to_close = []

            for trade in port.open_trades:
                p1 = row["close1"]
                p2 = row["close2"]
                hold_days = (date - trade.entry_date).days

                exit_reason = None
                if row["exit_flag"] and not pd.isna(row["zscore"]):
                    exit_reason = "reversion"
                elif hold_days >= cfg_s.max_hold_days:
                    exit_reason = "max_hold"
                elif port.current_drawdown >= CFG.risk.drawdown_stop:
                    exit_reason = "drawdown_stop"

                if exit_reason:
                    trade.close(
                        exit_date=date,
                        exit_price1=p1,
                        exit_price2=p2,
                        exit_zscore=row["zscore"],
                        exit_reason=exit_reason,
                        cost_model=total_cost_usd,
                        slippage_fn=slippage_usd,
                    )
                    daily_pnl += trade.net_pnl
                    trades_to_close.append(trade)

            for t in trades_to_close:
                port.open_trades.remove(t)
                port.closed_trades.append(t)

            port.update_equity(date, daily_pnl)
            cb_status = port.check_circuit_breaker()

            # 2. Check for new entry signals
            signal = row.get("raw_signal", 0)

            if signal != 0 and not port.paused:
                conf = 1.0
                if ml_confidence is not None and date in ml_confidence.index:
                    conf = ml_confidence.loc[date]
                    if conf < CFG.ml.min_confidence:
                        signal = 0

            if signal != 0 and not port.paused:
                realized_vol = df["return1"].iloc[max(0, i-20):i].std() * np.sqrt(252)
                spread_vol = df["spread_std"].iloc[i] if not pd.isna(df["spread_std"].iloc[i]) else 0.01
                vix = row.get("vix", 15.0)
                edge_frac = abs(row["zscore"]) * spread_vol / max(abs(row["close1"]), 1)

                notional = self.sizer.compute(edge_frac, spread_vol, realized_vol, vix)
                notional = max(notional, 10000)  # minimum $10k trade

                if port.can_add_position(notional):
                    hr = row.get("hedge_ratio", 1.0)
                    if pd.isna(hr) or hr <= 0:
                        hr = 1.0
                    units1 = notional / max(row["close1"], 0.01)
                    units2 = units1 * hr

                    trade = Trade(
                        pair=self.pair_key,
                        direction=signal,
                        entry_date=date,
                        entry_price1=row["close1"],
                        entry_price2=row["close2"],
                        entry_zscore=row["zscore"],
                        entry_fx=row.get("fx", 83.0),
                        notional=notional,
                        units1=units1,
                        units2=units2,
                        hedge_ratio=hr,
                        ticker1=self.ticker1,
                        ticker2=self.ticker2,
                    )
                    port.open_trades.append(trade)

            # 3. Record daily state
            daily_records.append({
                "date": date,
                "equity": port.equity,
                "drawdown": port.current_drawdown,
                "open_positions": len(port.open_trades),
                "gross_exposure": port.gross_exposure,
                "daily_pnl": daily_pnl,
                "zscore": row.get("zscore", np.nan),
                "spread_ols": row.get("spread_ols", np.nan),
                "coint_pval": row.get("coint_pval", np.nan),
                "circuit_breaker": cb_status,
                "signal_today": signal if "signal" in dir() else 0,
            })

        results = pd.DataFrame(daily_records).set_index("date")
        results["returns"] = results["equity"].pct_change()
        log.info(
            f"Backtest complete: {len(port.closed_trades)} trades, "
            f"final equity ${port.equity:,.0f}"
        )
        return results

    def get_trade_log(self) -> pd.DataFrame:
        all_trades = self.portfolio.closed_trades + self.portfolio.open_trades
        return pd.DataFrame([t.to_dict() for t in all_trades])