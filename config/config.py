"""
config.py — Central configuration for the Global Arbitrage Signal Engine
FIXED: Better signal parameters for improved Sharpe
"""

from dataclasses import dataclass, field
from typing import List

PAIRS = [
    ("QQQ",  "QQQM", "Nasdaq-100 ETF arb: Invesco QQQ vs QQQM (same index, different share classes)"),
    ("SPY",  "IVV",  "S&P500 ETF arb: SPDR SPY vs iShares IVV (identical underlying)"),
    ("GLD",  "IAU",  "Gold ETF arb: SPDR GLD vs iShares IAU (same underlying gold holdings)"),
    ("IEF",  "GOVT", "Treasury ETF arb: iShares 7-10yr vs Vanguard Govt Bond"),
    ("EEM",  "VWO",  "Emerging Markets ETF arb: iShares EEM vs Vanguard VWO (~95% overlap)"),
]

PRIMARY_PAIR = ("QQQ", "QQQM", "Nasdaq-100 structural arb")

@dataclass
class DataConfig:
    start_date: str = "2021-01-01"
    end_date: str   = "2024-12-31"
    interval: str   = "1d"
    fx_ticker: str  = "USDINR=X"
    cache_dir: str  = "data/cache"
    parquet_dir: str= "data/parquet"

@dataclass
class SignalConfig:
    zscore_lookback: int   = 30        # longer lookback = more stable z-score
    zscore_entry: float    = 2.0       # back to 2.0 — enough signals
    zscore_exit: float     = 0.5       # exit at 0.5 — let winners run longer
    cointegration_pval: float = 0.10
    min_volume_pct: float  = 0.05
    max_hold_days: int     = 20        # enforce 20-day time stop
    lookback_coint: int    = 60
    # NEW: momentum filter — only enter when spread is REVERTING
    momentum_filter: bool  = True      # require spread momentum confirming reversion
    # NEW: minimum edge filter
    min_edge_bps: float    = 1.5       # require at least 1.5bps net edge

@dataclass
class CostConfig:
    us_commission_pct: float  = 0.00005
    us_sec_fee_pct: float     = 0.00002
    us_bid_ask_pct: float     = 0.00005   # slightly wider — more realistic
    in_stt_pct: float         = 0.0
    in_brokerage_pct: float   = 0.0
    in_stamp_pct: float       = 0.0
    in_bid_ask_pct: float     = 0.0
    fx_spread_pct: float      = 0.00005
    lambda_impact: float      = 0.15
    adv_fraction: float       = 0.0002

@dataclass
class SizingConfig:
    kelly_fraction: float     = 0.25       # slightly more aggressive
    target_vol_annual: float  = 0.12
    vol_lookback: int         = 20
    max_position_pct: float   = 0.15
    max_gross_exposure: float = 0.50
    starting_capital: float   = 1_000_000.0

@dataclass
class RiskConfig:
    drawdown_warning: float   = 0.03
    drawdown_pause: float     = 0.07
    drawdown_stop: float      = 0.10
    rolling_sharpe_min: float = 0.30
    rolling_sharpe_window: int= 30
    coint_pval_max: float     = 0.15
    fx_vol_sigma: float       = 2.5
    vix_high: float           = 30.0
    size_reduction_high_vol: float = 0.50

@dataclass
class MLConfig:
    reversion_horizon: int    = 5
    train_end: str            = "2023-06-30"
    test_start: str           = "2023-07-01"
    features: List[str] = field(default_factory=lambda: [
        "zscore", "spread_momentum_5d", "spread_momentum_10d",
        "vol_ratio", "volume_rank", "vix_level",
        "days_since_last_signal", "fx_vol_20d",
        "leg1_momentum_10d", "leg2_momentum_10d",
    ])
    min_confidence: float     = 0.55   # lowered — ML has low AUC, don't over-filter

@dataclass
class LiveConfig:
    poll_interval_sec: int    = 60
    log_dir: str              = "results"
    log_filename: str         = "signal_log.jsonl"
    dashboard_refresh_sec: int= 30
    watchdog_timeout_sec: int = 120

class Config:
    data    = DataConfig()
    signal  = SignalConfig()
    costs   = CostConfig()
    sizing  = SizingConfig()
    risk    = RiskConfig()
    ml      = MLConfig()
    live    = LiveConfig()
    pairs   = PAIRS

CFG = Config()