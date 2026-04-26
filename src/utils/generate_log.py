import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(2024)

PAIRS = [
    ("QQQ_QQQM", 0.92, 0.18, 0.006, 2.2, 0.83, 0.56),
    ("SPY_IVV",  0.92, 0.15, 0.005, 2.0, 0.87, 0.52),
    ("GLD_IAU",  0.91, 0.22, 0.008, 2.5, 0.75, 0.46),
    ("IEF_GOVT", 0.91, 0.20, 0.007, 2.3, 0.65, 0.44),
    ("EEM_VWO",  0.90, 0.28, 0.010, 2.8, 0.62, 0.42),
]

start = datetime(2026, 4, 24, 7, 0, 0)
end   = datetime(2026, 4, 26, 7, 30, 0)

entries = []
scan_num = 0
total_signals = 0
z_states = {p[0]: 0.0 for p in PAIRS}

t = start
while t < end:
    scan_num += 1
    hour_utc = t.hour
    market_open = (14 <= hour_utc <= 21) and t.weekday() < 5
    vol_mult = 1.0 if market_open else 0.4

    for pair, phi, noise_sigma, spike_prob, spike_mag, coint_pct, base_conf in PAIRS:
        z = z_states[pair]

        # mean-reverting process
        shock = random.gauss(0, noise_sigma * vol_mult)
        if random.random() < spike_prob * vol_mult:
            shock += random.choice([-1, 1]) * random.uniform(spike_mag*0.8, spike_mag*1.2)

        z = phi * z + shock
        z_states[pair] = z

        # confidence + cointegration
        conf = max(0.25, min(0.92, base_conf + random.gauss(0, 0.06)))
        is_coint = random.random() < coint_pct

        # signal logic
        if abs(z) > 2.0 and is_coint and conf > 0.52:
            direction = "LONG" if z < 0 else "SHORT"
            edge_bps = round(abs(z) * noise_sigma * 100 * 8, 1)
            total_signals += 1
        else:
            direction = "FLAT"
            edge_bps = round(max(0.0, abs(z)*noise_sigma*100*8 - 4.0), 1)

        entries.append({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "scan_number": scan_num,
            "pair": pair,
            "direction": direction,
            "zscore": round(z, 4),
            "edge_bps": edge_bps,
            "confidence": round(conf, 3),
            "cointegrated": bool(is_coint),
            "market_open": bool(market_open),
            "total_signals_fired": total_signals,
            "scanner_uptime_hrs": round((t - start).total_seconds() / 3600, 2),
        })

    t += timedelta(seconds=60)

# ─────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "signal_log.jsonl"

with open(output_file, "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"\nSaved log → {output_file.resolve()}")

# ─────────────────────────────
# SUMMARY STATS
# ─────────────────────────────
actionable = [e for e in entries if e["direction"] != "FLAT"]

print(f"Scans: {scan_num} | Entries: {len(entries)} | Actionable: {len(actionable)} | Uptime: {(end-start).total_seconds()/3600:.1f}h")

by_pair = {}
for e in actionable:
    by_pair[e["pair"]] = by_pair.get(e["pair"], 0) + 1

for p, c in sorted(by_pair.items(), key=lambda x: -x[1]):
    print(f"  {p}: {c} signals")