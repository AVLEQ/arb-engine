# Global Arbitrage Signal Engine

**Saanvi TS | IIT Bombay IEOR | Aquifer Round 2 Submission**

---

## Project Structure

```
arb_engine/
├── config/config.py          ← All parameters (no hardcoded values elsewhere)
├── data/ingest.py            ← Data pipeline (yfinance → Parquet)
├── backtester/
│   ├── signals.py            ← Z-score, cointegration, ADF, momentum features
│   ├── engine.py             ← Event-driven backtester with full cost model
│   ├── metrics.py            ← Sharpe, Sortino, Calmar, drawdown, heatmap
│   └── ml_filter.py          ← XGBoost confidence model (P(reversion))
├── risk/framework.py         ← Kelly sizing, vol targeting, regime monitor, Monte Carlo
├── live/scanner.py           ← 48-hour continuous live signal logger
├── main.py                   ← Master runner (full pipeline)
├── strategy_memo.md          ← Strategy design write-up (Deliverable 1)
├── requirements.txt
└── results/                  ← All outputs land here
    ├── signal_log.jsonl      ← Live signal log (48h)
    ├── backtest_*.png        ← 6-panel backtest charts
    ├── trades_*.csv          ← Trade-level log
    ├── summary_*.json        ← Full metrics JSON
    ├── sensitivity_*.csv     ← Parameter sensitivity grid
    └── run.log               ← Execution log
```

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
python live/scanner.py
# or specific pair:
python live/scanner.py --pair GLD_IGLD
# dry run (print signals, don't write log):
python live/scanner.py --dry-run
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

- [x] **Deliverable 1:** Strategy Design (`strategy_memo.md`)
- [x] **Deliverable 2:** Backtesting Engine (`backtester/`, `main.py`) — 4 years of data, full cost model
- [x] **Deliverable 3:** Live Signal Generator (`live/scanner.py`) — runs 48h, logs JSONL
- [x] **Deliverable 4:** Risk Framework (`risk/framework.py`) — Kelly sizing, circuit breakers, regime detection

### Beyond Requirements
- XGBoost ML confidence layer with ablation study
- Monte Carlo bootstrap Sharpe uncertainty (2,000 simulations)
- Walk-forward validation (out-of-sample 2nd half)
- Parameter sensitivity heatmap (z-threshold × lookback)
- 6-panel backtest chart (equity, drawdown, z-score, cointegration, monthly PnL, edge decay)
- Edge capacity analysis (Kyle's lambda model)
- Regime monitor with 4 independent flags
