import json
from pathlib import Path

# ── PATHS ─────────────────────
base = Path("results")
log_file = base / "signal_log.jsonl"
output_file = base / "dashboard.html"

# ── LOAD DATA ────────────────
with open(log_file, "r") as f:
    entries = [json.loads(l) for l in f]

# ── PROCESS DATA ─────────────
pairs = ["QQQ_QQQM","SPY_IVV","GLD_IAU","IEF_GOVT","EEM_VWO"]

sampled = [e for e in entries if e.get("scan_number", 0) % 10 == 0]

by_pair = {p: [] for p in pairs}

for e in sampled:
    pair = e.get("pair")
    if pair not in by_pair:
        continue

    by_pair[pair].append({
        "t": e["timestamp"][:16],
        "z": e["zscore"]
    })

actionable = [e for e in entries if e.get("direction") != "FLAT"]

data_js = json.dumps({
    "pairs": by_pair,
    "signals": actionable
})

# ── HTML ─────────────────────
html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Arbitrage Engine</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{
    background:#0f172a;
    color:#e2e8f0;
    font-family: system-ui;
    padding:20px;
}}
h1 {{ color:#38bdf8; }}
.card {{
    background:#1e293b;
    padding:16px;
    border-radius:10px;
    margin-bottom:16px;
}}
.grid {{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:16px;
}}
.big {{ font-size:28px; font-weight:bold; }}
.good {{ color:#22c55e; }}
.bad {{ color:#ef4444; }}
.neutral {{ color:#94a3b8; }}
table {{ width:100%; border-collapse:collapse; }}
td, th {{ padding:8px; border-bottom:1px solid #334155; }}
th {{ color:#94a3b8; text-align:left; }}
</style>
</head>

<body>

<h1>📊 ETF Arbitrage Engine</h1>
<p style="color:#94a3b8">Statistical Pair Trading Dashboard</p>

<div class="grid">

<div class="card">
<h3>📈 System Health</h3>
<div class="big good">ACTIVE</div>
<p>Engine running normally</p>
</div>

<div class="card">
<h3>💡 Best Opportunity</h3>
<div id="best"></div>
</div>

</div>

<div class="card">
<h3>📊 Z-Score (QQQ / QQQM)</h3>
<canvas id="chart"></canvas>
</div>

<div class="card">
<h3>⚡ Recent Signals</h3>
<table>
<tr><th>Time</th><th>Pair</th><th>Action</th><th>Z</th></tr>
<tbody id="signals"></tbody>
</table>
</div>

<script>
const DATA = {data_js};

// Best trade
let best = null;
DATA.signals.forEach(s => {{
    if(!best || Math.abs(s.zscore) > Math.abs(best.zscore)) best = s;
}});

document.getElementById("best").innerHTML = best
? `<div class="big">${{best.pair.replace('_','/')}}</div>
   <div class="${{best.direction==='LONG'?'good':'bad'}}">${{best.direction}}</div>
   <div>Z = ${{best.zscore.toFixed(2)}}</div>`
: `<div class="neutral">No strong signals</div>`;

// Chart
const pts = DATA.pairs["QQQ_QQQM"];

new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{
        labels: pts.map(p => p.t),
        datasets: [{{
            label: 'Z-score',
            data: pts.map(p => p.z),
            borderColor: '#38bdf8',
            tension:0.2
        }}]
    }}
}});

// Signals
const tbody = document.getElementById("signals");

DATA.signals.slice(-8).reverse().forEach(s => {{
    tbody.innerHTML += `
    <tr>
        <td>${{s.timestamp.slice(11,16)}}</td>
        <td>${{s.pair.replace('_','/')}}</td>
        <td class="${{s.direction==='LONG'?'good':'bad'}}">${{s.direction}}</td>
        <td>${{s.zscore.toFixed(2)}}</td>
    </tr>`;
}});
</script>

</body>
</html>
"""

# ── SAVE ─────────────────────
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Dashboard generated:", output_file.resolve())