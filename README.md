# Global Arbitrage Signal Engine
**Saanvi TS | IIT Bombay IEOR | Aquifer Internship Submission**

---

## Project Structure

```
arb_engine/
├── config/                   ← All parameters (no hardcoded values elsewhere)
├── data/                     ← Data pipeline (yfinance → Parquet)
├── src/
│   ├── backtester/
│   │   ├── signals.py        ← Z-score, cointegration, ADF, momentum features
│   │   ├── engine.py         ← Event-driven backtester with full cost model
│   │   ├── metrics.py        ← Sharpe, Sortino, Calmar, drawdown, heatmap
│   │   └── ml_filter.py      ← XGBoost confidence model (P(reversion))
│   ├── live/
│   │   └── scanner.py        ← 48-hour continuous live signal logger
│   ├── risk/
│   │   └── framework.py      ← Kelly sizing, vol targeting, regime monitor, Monte Carlo
│   └── utils/
│       ├── generate_dashboard.py
│       └── Generate_report.py
├── reports/                  ← Submission PDF and strategy memo
├── results/                  ← All outputs land here
│   ├── signal_log.jsonl      ← Live signal log (48.48h, 21 signals)
│   ├── backtest_*.png        ← 6-panel backtest charts
│   ├── trades_*.csv          ← Trade-level log
│   ├── summary_*.json        ← Full metrics JSON
│   └── run.log               ← Execution log
├── main.py                   ← Master runner (full pipeline)
├── requirements.txt
└── __init__.py
```

---

## Backtest Results (2021–2024)

| Metric | Value |
|--------|-------|
| Total Net P&L | +1,786 USD |
| Win Rate | 44.3% |
| Annualised Daily Sharpe | 0.16 |
| Walk-Forward Sharpe (OOS) | 1.044 |
| Trade-Level Sharpe | 2.148 |
| Max Drawdown | -0.95% |
| Cost Drag | 52.9% |
| Total Trades | 97 |

> **Note:** Annualised daily Sharpe (0.16) is the standard full-period metric. Walk-forward Sharpe (1.044) reflects the out-of-sample regime-conditional edge. Transaction costs (52.9% drag) are the primary performance constraint, not signal quality.

---

## Live Signal Generator Results

- **Runtime:** 48.48 hours continuous, 0% downtime
- **Total Scans:** 2,910 cycles
- **Actionable Signals Logged:** 21
- **Pairs Monitored:** QQQ/QQQM, SPY/IVV, GLD/IAU, EEM/VWO, IEF/GOVT

Each log entry contains: `timestamp`, `instrument`, `direction`, `zscore`, `edge_bps`, `confidence`, `cointegrated`, `market_open`, `scanner_uptime_hrs`.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Full pipeline (primary pair, with ML):
```bash
python main.py
```

### All 5 pairs:
```bash
python main.py --all-pairs
```

### Skip re-downloading data (use cached Parquet):
```bash
python main.py --skip-ingest
```

### Pure z-score (no ML filter):
```bash
python main.py --no-ml
```

### Start the 48-hour live signal scanner:
```bash
python src/live/scanner.py
# specific pair:
python src/live/scanner.py --pair GLD_IAU
# dry run (print signals, don't write log):
python src/live/scanner.py --dry-run
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Spread model | Rolling OLS (not fixed 1:1) | Correct hedge ratio; avoids spurious signals |
| Cost model | Square-root market impact + full commissions | Required for non-disqualified results |
| Position sizing | Fractional Kelly (25%) + vol targeting | Balances edge exploitation vs parameter uncertainty |
| Signal gate | Cointegration + volume + ML confidence | Filters false positives; adds robustness |
| Risk | 4-factor regime monitor + circuit breakers | Detects edge degradation before large drawdown |
| Validation | Walk-forward + Monte Carlo + sensitivity grid | Distinguishes real edge from in-sample fitting |

---

## Deliverables Checklist

- [x] **Deliverable 1:** Strategy Design (`reports/strategy_memo.md`)
- [x] **Deliverable 2:** Backtesting Engine (`src/backtester/`, `main.py`) — 4 years of real data, full cost model, walk-forward validation
- [x] **Deliverable 3:** Live Signal Generator (`src/live/scanner.py`) — 48.48h runtime, 2,910 scans, 21 signals logged to JSONL
- [x] **Deliverable 4:** Risk Framework (`src/risk/framework.py`) — Fractional Kelly sizing, circuit breakers, regime detection

### Beyond Requirements

- XGBoost ML confidence layer with ablation study
- Monte Carlo bootstrap Sharpe uncertainty (2,000 simulations)
- Walk-forward validation (out-of-sample 2nd half) — Sharpe 1.044
- Parameter sensitivity heatmap (z-threshold × lookback)
- 6-panel backtest chart (equity, drawdown, z-score, cointegration, monthly PnL, edge decay)
- Edge capacity analysis (Kyle's lambda model)
- Regime monitor with 4 independent flags
