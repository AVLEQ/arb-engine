# Strategy Design Memo
## Global Arbitrage Signal Engine
**Author:** Saanvi TS | IIT Bombay IEOR | Roll: 24b4501  
**Date:** April 2025  

---

## 1. What Inefficiency Are We Exploiting?

### The Core Idea: Cross-Listed ETF Dislocation

We exploit temporary price dislocations between economically equivalent assets that trade on different exchanges or in different wrappers. Specifically, we focus on **ETF pairs where both instruments have nearly identical underlying exposure** — for example, two gold ETFs, or a broad US ETF and its Indian market counterpart via a US-listed proxy.

When the same economic exposure can be accessed through two different vehicles, their prices should move in lockstep once converted to the same currency. When they don't — when the spread between them diverges beyond what transaction costs justify — that spread carries a strong statistical pull back toward zero.

### Why Does This Inefficiency Exist?

Several structural reasons prevent this spread from being instantly arbitraged away:

**1. Liquidity fragmentation.** Market makers on different exchanges quote independently. A large order on one leg moves that leg's price without moving the other's. The reconnection takes time — typically minutes to hours for liquid ETFs, sometimes days for less-liquid ones.

**2. Investor clientele effects.** Retail and institutional flows differ by venue. INDA (iShares MSCI India ETF on NYSE) and the underlying Indian equities it tracks are bought and sold by different sets of participants who respond to different news and sentiment.

**3. Capital controls and FX friction.** For pairs involving India, FX conversion cost, regulatory constraints on capital flows, and hedging cost create a persistent, if time-varying, basis that is not purely a free arbitrage. The basis can widen temporarily when FX moves sharply or when India-specific flows dominate.

**4. Creation/redemption delays.** ETF arbitrage via the creation/redemption mechanism is efficient at scale but subject to settlement timing (T+1 or T+2). This creates windows of several hours to several days where a significant ETF premium or discount persists.

**5. Attention and latency.** Most participants are not monitoring cross-ETF spreads continuously. Institutional desks that do monitor them require minimum profit thresholds (often 5–15 bps after costs) before acting, and they size positions conservatively to avoid moving the market. This creates a "dead zone" around fair value within which small dislocations are not worth trading for large funds.

### Why Hasn't It Been Fully Arbitraged Away?

The edge has *not* been fully eliminated because:
- It is too small for large funds to exploit without market impact erasing the profit
- It requires monitoring multiple pairs across multiple time zones continuously
- It requires careful FX hedging, which adds cost and operational complexity
- The edge is inconsistent — it requires statistical filters (cointegration) to distinguish real dislocations from structural regime changes

This is a strategy that sits in the "small fund" / "stat arb boutique" sweet spot: too small for a billion-dollar fund to exploit efficiently, too sophisticated and data-intensive for retail traders.

---

## 2. Entry and Exit Signals

### Entry Signal

We enter a position when **all four** of the following conditions are simultaneously true:

1. **Z-score trigger:** The rolling z-score of the spread (using a 30-day rolling OLS hedge ratio) exceeds ±2.0 standard deviations. This is computed on the residual spread from a rolling OLS regression, not a naive log-price difference.

2. **Cointegration gate:** A rolling Engle-Granger cointegration test (90-day lookback, recomputed every 5 days for efficiency) returns a p-value below 0.05. If the pair is not statistically cointegrated, the z-score is meaningless as a reversion predictor.

3. **Volume gate:** Both legs must be trading in the top 80th percentile of their 60-day rolling volume distribution. Trading in low-volume conditions amplifies slippage and increases the risk of stale prices.

4. **ML confidence gate:** An XGBoost classifier (trained on historical signal outcomes) must assign a probability ≥ 0.55 that the spread reverts within 5 days. This filters out signals that are statistically extreme but historically associated with poor outcomes (e.g., signals fired during FX vol spikes or VIX regime changes).

**Direction:**
- Spread too low (z < −2.0): Long leg1, short leg2 (expect spread to rise)
- Spread too high (z > +2.0): Short leg1, long leg2 (expect spread to fall)

### Exit Signal

We exit when **any one** of the following is true:

1. **Reversion:** |z-score| drops below 0.5 (spread has reverted toward its mean)
2. **Time stop:** Position has been held for 10 days (limits drawdown from non-reverting signals)
3. **Drawdown stop:** Portfolio drawdown exceeds 8% (circuit breaker)

---

## 3. Theoretical Edge and Edge Decay

### Expected Edge

The gross edge of a signal is:

```
gross_edge_bps = |z-score| × spread_std × 10,000
```

For a typical entry at |z| = 2.0 with spread_std = 0.015, this gives 30 bps gross.

Total round-trip transaction cost (US + India legs + FX):

| Cost component       | bps |
|----------------------|-----|
| US commission (both sides)     | 2.0 |
| US bid-ask spread (both sides) | 1.0 |
| India STT (both sides)         | 2.0 |
| India brokerage                | 0.6 |
| India bid-ask spread           | 1.6 |
| FX spread (both sides)         | 1.0 |
| **Total round-trip cost**      | **8.2 bps** |

Net expected edge at entry: ~21.8 bps. After adjusting for win rate (~60%), expected profit per trade: ~13 bps net.

### Edge Decay with Position Size

We use a square-root market impact model (Kyle's lambda framework):

```
slippage_bps = λ × sqrt(Q / ADV)
```

Where:
- λ = 5.0 bps per sqrt(fraction of ADV) — conservative for ETFs
- ADV = $1,000,000 (daily volume, conservative estimate)

At $50,000 notional (5% of $1M capital), slippage is minimal (~0.35 bps). The strategy's **capacity** — the point where market impact erases all edge — is approximately **$1.8M per trade** for liquid ETF pairs. This is computed and displayed in the backtest chart (edge decay panel).

This is why the strategy is not appropriate for large funds but viable for a fund under ~$10M in AUM targeting 5–10 trades simultaneously.

---

## 4. Backtesting Approach

### Data
- Source: yfinance (real historical OHLCV, auto-adjusted for splits/dividends)
- Period: January 2021 – December 2024 (4 years, >1,000 trading days)
- Pairs: 5 cross-listed ETF pairs (gold, large-cap, EM, rates, tech proxy)
- FX: USD/INR spot from yfinance (USDINR=X)
- VIX: CBOE VIX index for regime filtering

### Transaction Costs Modelled
Every trade includes:
- Commission (both legs), bid-ask spread, STT, FX conversion, stamp duty, slippage (square-root model)
- No results are reported without full cost deduction — disqualification-proof

### Validation
- Walk-forward Sharpe (out-of-sample 2nd half of period)
- Monte Carlo bootstrap (2,000 resamples) to quantify Sharpe uncertainty
- Sensitivity heatmap (z-threshold × lookback window) to confirm robustness
- ML ablation: compare ML-filtered vs pure z-score (quantifies ML contribution)

---

## 5. Risk Framework

### Position Sizing

Position sizing uses a three-layer approach:

**Layer 1: Fractional Kelly (25%)**
```
f* = (E[spread_return] / Var[spread_return]) × 0.25
```
Full Kelly is theoretically optimal but practically reckless (it assumes perfect knowledge of the true distribution). We use 25% of full Kelly to account for parameter uncertainty.

**Layer 2: Volatility targeting**
The fractional Kelly size is then scaled so that the position contributes a fixed annualised volatility to the portfolio (target: 10%). In high-volatility environments, this automatically reduces position size.

**Layer 3: Regime adjustment**
- VIX > 25: position size halved
- Circuit breaker triggered: position size = 0

Hard cap: no single position exceeds 5% of NAV. Total gross exposure capped at 20% of NAV.

### Circuit Breakers

| Drawdown | Action |
|----------|--------|
| 2% | Log warning |
| 5% | Pause new signals |
| 8% | Full stop, close all positions |

### Regime Monitor (4-factor)

The `RegimeMonitor` class tracks:
1. **Rolling Sharpe** (30-day): if below 0.50, flag degraded
2. **Cointegration p-value**: if above 0.10, relationship may have broken
3. **FX volatility z-score**: if INR/USD vol spikes >2σ above its mean, flag
4. **VIX**: if above 25, flag high-vol regime

If 2+ flags are active simultaneously, strategy is suspended until conditions normalise.

### When Does the Edge Disappear?

The strategy's edge degrades or disappears under these conditions:

1. **Cointegration breakdown**: If the two instruments diverge structurally (e.g., one ETF changes its benchmark, or India imposes capital controls), the spread may not revert. Detected by: rolling cointegration p-value rising above 0.10.

2. **VIX spikes above 30**: During market stress, ETF premiums/discounts widen for structural reasons (redemption mechanism stress, liquidity dry-up). Mean-reversion logic breaks down. Detected by: VIX level.

3. **FX regime change**: A large, fast USD/INR move creates a basis shift that looks like a signal but is not a reversion opportunity. Detected by: FX vol z-score.

4. **Regulatory change**: STT, capital controls, or ETF regulation changes can permanently alter transaction costs. Detected by: cost_drag_pct rising significantly vs historical baseline.

---

## 6. Honest Limitations

No strategy memo is complete without this section.

1. **Data quality**: yfinance data quality for Indian market proxies is imperfect. Pairs like GLD/IGLD involve a US-listed India gold ETF that may have low trading volume, widening bid-ask spreads in ways not fully captured by historical data.

2. **Execution assumption**: The backtest assumes we can trade at the prior day's close price. In reality, especially for less-liquid legs, we may face additional slippage or fail to fill at all. The square-root slippage model is an approximation.

3. **The strategy is not high-frequency**: Signals fire at daily close prices. Real cross-listed arbitrage at intraday resolution would require co-location, direct market access, and substantially more infrastructure.

4. **Regime sensitivity**: The strategy's Sharpe ratio varies significantly across market regimes. During low-VIX, stable FX periods (2021 much of 2023), performance is strong. During 2022 (rate shock) and early 2024 (India election uncertainty), edge compresses. Walk-forward validation partially captures this, but forward-looking regime prediction remains hard.

5. **Capacity**: The strategy is not scalable beyond approximately $5–10M AUM without fundamentally changing the execution approach (using limit orders, TWAP/VWAP algorithms, etc.).

These limitations are features, not failures — they define the operating envelope of the strategy and inform risk management decisions.
