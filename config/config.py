"""
config.py — Central configuration for the Global Arbitrage Signal Engine
All parameters live here. No hardcoded values elsewhere.
"""

from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────────
# INSTRUMENT PAIRS
# ─────────────────────────────────────────────
# Each pair tracks the SAME underlying index — structural arbitrage
# These are not correlated pairs, they are near-identical instruments
PAIRS = [
    ("QQQ",  "QQQM", "Nasdaq-100 ETF arb: Invesco QQQ vs QQQM (same index, different share classes)"),
    ("SPY",  "IVV",  "S&P500 ETF arb: SPDR SPY vs iShares IVV (same index, ~$0 spread at scale)"),
    ("GLD",  "IAU",  "Gold ETF arb: SPDR GLD vs iShares IAU (same underlying gold holdings)"),
    ("IEF",  "GOVT", "Treasury ETF arb: iShares 7-10yr vs Vanguard Govt Bond"),
    ("EEM",  "VWO",  "EM ETF arb: iShares EEM vs Vanguard VWO (~95% overlap in holdings)"),
]

# Primary pair for deep analysis — QQQ/QQQM is the cleanest structural arb
PRIMARY_PAIR = ("QQQ", "QQQM", "Nasdaq-100 structural arb")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    start_date: str = "2021-01-01"
    end_date: str   = "2024-12-31"
    interval: str   = "1d"
    fx_ticker: str  = "USDINR=X"
    cache_dir: str  = "data/cache"
    parquet_dir: str= "data/parquet"

# ─────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────
@dataclass
class SignalConfig:
    zscore_lookback: int   = 20
    zscore_entry: float    = 1.5      # enter when |z| > 1.5
    zscore_exit: float     = 0.3      # exit when |z| < 0.3
    cointegration_pval: float = 0.10
    min_volume_pct: float  = 0.10
    max_hold_days: int     = 15
    lookback_coint: int    = 60

# ─────────────────────────────────────────────
# TRANSACTION COSTS
# ─────────────────────────────────────────────
@dataclass
class CostConfig:
    # US side — ultra-liquid ETFs
    # QQQ/SPY/GLD trade $30-50B/day — spreads are literally $0.01
    us_commission_pct: float  = 0.0001   # 0.01% — IB or zero-commission broker
    us_sec_fee_pct: float     = 0.00008  # SEC fee on sells only
    us_bid_ask_pct: float     = 0.00005  # sub-penny spread on mega-ETFs

    # India side — not applicable for US-only ETF pairs
    in_stt_pct: float         = 0.0
    in_brokerage_pct: float   = 0.0
    in_stamp_pct: float       = 0.0
    in_bid_ask_pct: float     = 0.0

    # FX — minimal for USD pairs
    fx_spread_pct: float      = 0.0001

    # Market impact: Kyle's lambda
    # For QQQ ($200B+ ADV), lambda is negligible — 0.5 bps per sqrt(Q/ADV)
    lambda_impact: float      = 0.5
    adv_fraction: float       = 0.0005   # 0.05% of ADV (tiny for these instruments)

# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
@dataclass
class SizingConfig:
    kelly_fraction: float     = 0.25
    target_vol_annual: float  = 0.15
    vol_lookback: int         = 20
    max_position_pct: float   = 0.20    # max 20% NAV per trade
    max_gross_exposure: float = 0.60    # max 60% NAV total open
    starting_capital: float   = 1_000_000.0

# ─────────────────────────────────────────────
# RISK / CIRCUIT BREAKERS
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    drawdown_warning: float   = 0.02
    drawdown_pause: float     = 0.05
    drawdown_stop: float      = 0.08
    rolling_sharpe_min: float = 0.50
    rolling_sharpe_window: int= 30
    coint_pval_max: float     = 0.10
    fx_vol_sigma: float       = 2.0
    vix_high: float           = 25.0
    size_reduction_high_vol: float = 0.50

# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────
@dataclass
class MLConfig:
    reversion_horizon: int    = 5
    train_end: str            = "2022-12-31"
    test_start: str           = "2023-01-01"
    features: List[str] = field(default_factory=lambda: [
        "zscore", "spread_momentum_5d", "spread_momentum_10d",
        "vol_ratio", "volume_rank", "vix_level",
        "days_since_last_signal", "fx_vol_20d",
        "leg1_momentum_10d", "leg2_momentum_10d",
    ])
    min_confidence: float     = 0.52    # slightly lower gate to pass more signals

# ─────────────────────────────────────────────
# LIVE SIGNAL LOGGER
# ─────────────────────────────────────────────
@dataclass
class LiveConfig:
    poll_interval_sec: int    = 60
    log_dir: str              = "results"
    log_filename: str         = "signal_log.jsonl"
    dashboard_refresh_sec: int= 30
    watchdog_timeout_sec: int = 120

# ─────────────────────────────────────────────
# MASTER CONFIG
# ─────────────────────────────────────────────
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