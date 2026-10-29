# Dashboard — Streamlit Mission Control

This folder is reserved for `dashboard/app.py` (Streamlit), which visualises:

- Live readiness panel (kafka_connected, last_seq, is_ready)
- Per-day P&L curve
- Open positions + unrealised P&L
- Journal filter (ticker, action, status)
- Reject-code histogram
- Decision latency p50/p95

Per `docs/12 §1 Pillar 1`: the dashboard is a **jury artifact**, not on
the critical trading path. It runs as a separate process:

```bash
streamlit run dashboard/app.py
```

Sketch:
```python
# dashboard/app.py (sketch)
import streamlit as st
import json, pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Manthan 2026 — Mission Control")

journal = Path("runs")  # glob for journal.jsonl files
# ... read latest journal, plot equity curve, open positions, etc.
```

No code here yet — separate workstream.