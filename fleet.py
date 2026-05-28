"""
CleanPlanet DAPL Fleet Risk Monitor
Classic-firmware (AC001 / always_clean) fleet view with per-unit drill-down.
Run: python -m streamlit run fleet.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
from datetime import datetime, timezone
from io import StringIO

BASE_HOST = "https://recycling.cleanplanetchemical.com"

# ============================================================
# CONFIG
# ============================================================
KEY_VARS = ["Transducer04", "Temp08", "Energy", "DAC1", "Feedback06"]

VAR_AUC = {
    "Transducer04": 0.803,
    "Temp08":       0.789,
    "Energy":       0.789,
    "Feedback06":   0.781,
    "DAC1":         0.768,
}
_raw_weights = {v: VAR_AUC[v] - 0.5 for v in KEY_VARS}
_total = sum(_raw_weights.values())
VAR_WEIGHTS = {v: round(_raw_weights[v] / _total, 3) for v in KEY_VARS}

VAR_LABELS = {
    "Transducer04": {"label": "Pressure Sensor 4",    "unit": "bit", "icon": "⊙"},
    "Temp08":       {"label": "Temperature 8",        "unit": "°C",  "icon": "◐"},
    "Energy":       {"label": "Cumulative Energy",    "unit": "Wh",  "icon": "⚡"},
    "DAC1":         {"label": "Level Sensor (DAC1)",  "unit": "bit", "icon": "◇"},
    "Feedback06":   {"label": "Feedback Signal 6",    "unit": "—",   "icon": "◈"},
}
VAR_PRECISION = {"Transducer04": 1, "Temp08": 1, "Energy": 0, "DAC1": 1, "Feedback06": 1}

TIER_CRITICAL_MIN = 80
TIER_WARNING_MIN = 60
TIER_WATCH_MIN = 40

CSV_PATH = "danger_ranges.csv"


st.set_page_config(
    page_title="CleanPlanet Fleet Monitor",
    page_icon="⬢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING (same as before — abbreviated marker)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800;900&display=swap');

.stApp {
    background: #0a0e1a;
    background-image:
        radial-gradient(at 0% 0%, hsla(160, 70%, 25%, 0.25) 0px, transparent 50%),
        radial-gradient(at 100% 0%, hsla(190, 70%, 30%, 0.2) 0px, transparent 50%),
        radial-gradient(at 50% 100%, hsla(150, 80%, 20%, 0.15) 0px, transparent 50%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e6edf3;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1500px; }

.hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(6, 78, 59, 0.85) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 28px;
    padding: 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}
.hero::before {
    content: ""; position: absolute; top: -200px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
    animation: floatSlow 12s ease-in-out infinite;
}
.hero-content { position: relative; z-index: 1; }
.hero-logo {
    display: inline-flex; align-items: center; justify-content: center;
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    border-radius: 14px;
    font-size: 26px; color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
}
.hero-title { color: #f8fafc; font-size: 44px; font-weight: 800; letter-spacing: -2px; margin: 0; line-height: 1.05; }
.hero-gradient-text {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: rgba(226, 232, 240, 0.7); font-size: 16px; margin-top: 14px; max-width: 680px; line-height: 1.6; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(16, 185, 129, 0.12); color: #6ee7b7;
    padding: 8px 16px; border-radius: 100px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    margin-top: 20px;
    border: 1px solid rgba(16, 185, 129, 0.25);
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
}
.hero-badge-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulseLED 1.8s infinite; }

.stats-grid {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 32px;
}
.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px; padding: 20px;
    backdrop-filter: blur(15px);
}
.stat-card.critical { border-color: rgba(220, 38, 38, 0.5); background: linear-gradient(180deg, rgba(220, 38, 38, 0.12), rgba(15, 23, 42, 0.6)); }
.stat-card.warning { border-color: rgba(234, 88, 12, 0.5); background: linear-gradient(180deg, rgba(234, 88, 12, 0.12), rgba(15, 23, 42, 0.6)); }
.stat-card.watch { border-color: rgba(202, 138, 4, 0.4); background: linear-gradient(180deg, rgba(202, 138, 4, 0.10), rgba(15, 23, 42, 0.6)); }
.stat-card.normal { border-color: rgba(16, 185, 129, 0.4); background: linear-gradient(180deg, rgba(16, 185, 129, 0.10), rgba(15, 23, 42, 0.6)); }
.stat-label { color: rgba(148, 163, 184, 0.8); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.stat-value { color: #f1f5f9; font-size: 32px; font-weight: 900; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.stat-sub { color: rgba(148, 163, 184, 0.6); font-size: 11px; margin-top: 6px; }

.tier-section { margin: 36px 0 16px 0; }
.tier-header {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 20px; border-radius: 16px; margin-bottom: 16px;
}
.tier-header.critical { background: linear-gradient(90deg, rgba(220, 38, 38, 0.2), transparent); border-left: 4px solid #dc2626; }
.tier-header.warning  { background: linear-gradient(90deg, rgba(234, 88, 12, 0.18), transparent); border-left: 4px solid #ea580c; }
.tier-header.watch    { background: linear-gradient(90deg, rgba(202, 138, 4, 0.15), transparent); border-left: 4px solid #ca8a04; }
.tier-header.normal   { background: linear-gradient(90deg, rgba(16, 185, 129, 0.12), transparent); border-left: 4px solid #10b981; }
.tier-header.error    { background: linear-gradient(90deg, rgba(148, 163, 184, 0.10), transparent); border-left: 4px solid #94a3b8; }
.tier-title { color: #f1f5f9; font-size: 18px; font-weight: 700; margin: 0; flex: 1; }
.tier-count { color: rgba(226, 232, 240, 0.7); font-family: 'JetBrains Mono', monospace; font-size: 13px; }

.unit-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 18px; padding: 20px; height: 100%;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.unit-card.critical { border-color: rgba(220, 38, 38, 0.4); }
.unit-card.warning  { border-color: rgba(234, 88, 12, 0.4); }
.unit-card.watch    { border-color: rgba(202, 138, 4, 0.4); }
.unit-card.normal   { border-color: rgba(16, 185, 129, 0.3); }
.unit-card.error    { border-color: rgba(148, 163, 184, 0.2); opacity: 0.7; }

.unit-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; gap: 10px; }
.unit-name { color: #f1f5f9; font-size: 16px; font-weight: 700; line-height: 1.2; }
.unit-id { color: rgba(148, 163, 184, 0.7); font-size: 10px; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.unit-company { color: rgba(203, 213, 225, 0.85); font-size: 11px; margin-top: 4px; }

.risk-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 100px;
    font-size: 18px; font-weight: 900; font-family: 'JetBrains Mono', monospace;
    color: white;
    white-space: nowrap;
}
.risk-pill.critical { background: linear-gradient(135deg, #dc2626, #b91c1c); }
.risk-pill.warning  { background: linear-gradient(135deg, #ea580c, #c2410c); }
.risk-pill.watch    { background: linear-gradient(135deg, #ca8a04, #a16207); }
.risk-pill.normal   { background: linear-gradient(135deg, #10b981, #047857); }
.risk-pill.error    { background: rgba(148, 163, 184, 0.3); color: rgba(226, 232, 240, 0.7); font-size: 13px; }

.unit-divider { height: 1px; background: rgba(148, 163, 184, 0.1); margin: 12px 0; }
.unit-indicators { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }
.indicator-dot {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 7px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; font-weight: 700;
    vertical-align: middle;
}   
.indicator-dot.in-danger { background: rgba(220, 38, 38, 0.25); color: #fca5a5; border: 1px solid rgba(220, 38, 38, 0.4); }
.indicator-dot.safe { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.indicator-dot.no-data { background: rgba(148, 163, 184, 0.1); color: rgba(148, 163, 184, 0.5); border: 1px solid rgba(148, 163, 184, 0.15); }

.unit-meta { color: rgba(148, 163, 184, 0.7); font-size: 10px; font-family: 'JetBrains Mono', monospace; }
.unit-error-msg { color: rgba(252, 165, 165, 0.8); font-size: 11px; font-style: italic; margin-top: 6px; }

.section-header { display: flex; align-items: center; gap: 14px; margin: 36px 0 18px 0; }
.section-bar { width: 4px; height: 28px; background: linear-gradient(180deg, #10b981, #06b6d4); border-radius: 4px; }
.section-title { color: #f1f5f9; font-size: 22px; font-weight: 700; margin: 0; }

.info-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px; padding: 22px 24px;
}
.info-label { color: rgba(148, 163, 184, 0.7); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.info-value { color: #f1f5f9; font-size: 20px; font-weight: 700; }
.info-sub { color: rgba(148, 163, 184, 0.6); font-size: 12px; margin-top: 4px; }

.risk-container {
    border-radius: 28px; padding: 50px 40px; text-align: center;
    color: white; margin: 24px 0;
    position: relative; overflow: hidden;
    box-shadow: 0 30px 80px rgba(0,0,0,0.4);
}
.risk-container::before { content: ""; position: absolute; inset: 0; background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.18), transparent 40%); }
.risk-emoji-big { font-size: 60px; margin-bottom: 6px; position: relative; z-index: 1; }
.risk-percentage-big { font-size: 100px; font-weight: 900; letter-spacing: -5px; line-height: 1; color: white; font-family: 'JetBrains Mono', monospace; position: relative; z-index: 1; }
.risk-tier-pill { display: inline-block; font-size: 20px; font-weight: 700; padding: 8px 20px; background: rgba(255,255,255,0.18); border-radius: 100px; border: 1px solid rgba(255,255,255,0.3); margin-top: 12px; text-transform: uppercase; position: relative; z-index: 1; }
.risk-detail-big { font-size: 15px; opacity: 0.9; margin-top: 18px; position: relative; z-index: 1; }

.sensor-card-detail {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 22px; padding: 24px 16px; text-align: center;
    height: 100%;
}
.sensor-card-detail.safe { border-color: rgba(16, 185, 129, 0.4); }
.sensor-card-detail.danger { border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(180deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.6)); }
.sensor-card-detail.no-data { border-color: rgba(148, 163, 184, 0.15); opacity: 0.6; }
.sensor-icon-d { width: 48px; height: 48px; margin: 0 auto 14px auto; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 22px; }
.sensor-card-detail.safe .sensor-icon-d { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
.sensor-card-detail.danger .sensor-icon-d { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
.sensor-code-d { color: rgba(148, 163, 184, 0.7); font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; }
.sensor-label-d { color: #e2e8f0; font-size: 12px; font-weight: 600; margin: 6px 0 12px 0; min-height: 28px; }
.sensor-value-d { font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.sensor-card-detail.safe .sensor-value-d { color: #6ee7b7; }
.sensor-card-detail.danger .sensor-value-d { color: #fca5a5; }
.sensor-card-detail.no-data .sensor-value-d { color: rgba(148, 163, 184, 0.4); }
.sensor-zone-d { color: rgba(148, 163, 184, 0.6); font-size: 10px; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(148, 163, 184, 0.1); font-family: 'JetBrains Mono', monospace; }
.sensor-badge-d { display: inline-block; padding: 3px 10px; border-radius: 100px; font-size: 9px; font-weight: 700; margin-top: 8px; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'JetBrains Mono', monospace; }
.badge-safe-d { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
.badge-danger-d { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
.badge-no-data-d { background: rgba(148, 163, 184, 0.15); color: rgba(148, 163, 184, 0.7); }

.pareto-row { display: flex; align-items: center; gap: 16px; margin-bottom: 10px; padding: 12px 16px; background: rgba(30, 41, 59, 0.4); border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1); }
.pareto-icon { font-size: 22px; width: 32px; text-align: center; }
.pareto-name { width: 150px; }
.pareto-name-main { color: #e2e8f0; font-size: 13px; font-weight: 600; }
.pareto-name-sub { color: rgba(148, 163, 184, 0.6); font-size: 10px; font-family: 'JetBrains Mono', monospace; }
.pareto-bar-track { flex: 1; height: 22px; background: rgba(148, 163, 184, 0.1); border-radius: 12px; overflow: hidden; }
.pareto-bar-fill { height: 100%; border-radius: 12px; transition: width 1s ease; }
.pareto-bar-fill.contributing { background: linear-gradient(90deg, #dc2626, #ef4444); box-shadow: 0 0 12px rgba(239, 68, 68, 0.4); }
.pareto-bar-fill.not-contributing { background: linear-gradient(90deg, rgba(16, 185, 129, 0.3), rgba(16, 185, 129, 0.1)); }
.pareto-weight { width: 70px; text-align: right; color: #38bdf8; font-size: 12px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pareto-value { width: 70px; text-align: right; color: #e2e8f0; font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

section[data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid rgba(148, 163, 184, 0.08); }
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 13px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'JetBrains Mono', monospace; }
section[data-testid="stSidebar"] label { color: rgba(226, 232, 240, 0.85) !important; font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px; }
section[data-testid="stSidebar"] input { background: rgba(30, 41, 59, 0.6) !important; border: 1px solid rgba(148, 163, 184, 0.2) !important; color: #f1f5f9 !important; border-radius: 10px !important; padding: 12px 16px !important; }
section[data-testid="stSidebar"] input:focus { border-color: rgba(16, 185, 129, 0.6) !important; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important; }
section[data-testid="stSidebar"] .stButton button { background: linear-gradient(135deg, #10b981, #06b6d4) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 14px !important; width: 100% !important; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35) !important; margin-top: 8px !important; }
section[data-testid="stSidebar"] .stButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 14px 30px rgba(16, 185, 129, 0.5) !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: rgba(203, 213, 225, 0.75) !important; font-size: 13px !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.15) !important; }

.stProgress > div > div > div { background: linear-gradient(90deg, #10b981, #06b6d4) !important; }
.stButton button { border-radius: 10px !important; }
[data-testid="stAlert"] { background: rgba(30, 41, 59, 0.6) !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 12px !important; color: #e2e8f0 !important; }
.stSpinner > div { border-top-color: #10b981 !important; }
.stSpinner > div + div { color: #e2e8f0 !important; }

@keyframes pulseLED { 0%, 100% { opacity: 1; box-shadow: 0 0 12px #10b981; } 50% { opacity: 0.6; box-shadow: 0 0 6px #10b981; } }
@keyframes floatSlow { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(30px, -30px); } }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DANGER ZONES
# ============================================================
def parse_range_string(s):
    if pd.isna(s) or not isinstance(s, str):
        return None, None
    matches = re.findall(r"-?\d+\.?\d*", s)
    if len(matches) < 2:
        return None, None
    return float(matches[0]), float(matches[1])


@st.cache_data
def load_danger_zones(path=CSV_PATH):
    df = pd.read_csv(path)
    zones = {}
    for _, row in df.iterrows():
        var = row["Variable"]
        if var not in KEY_VARS:
            continue
        lo, hi = parse_range_string(row["Danger zone"])
        if lo is None or hi is None:
            zones[var] = None
            continue
        zones[var] = {"low": lo, "high": hi}
    return zones


# ============================================================
# API
# ============================================================
@st.cache_resource(ttl=3000)
def get_token(email, password):
    r = requests.post(
        f"{BASE_HOST}/api/v1/auth/login",
        json={"email": email, "password": password}, timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_unit_list(token):
    """
    Discover production classic-firmware always_clean units.
    Filters:
      - model == 'always_clean'
      - firmware_version starts with 'AC001'
      - Excludes dev/test/obsolete units by name pattern
      - Excludes units with no recent data
    """
    H = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_HOST}/api/v1/units", headers=H, timeout=60)
    r.raise_for_status()
    data = r.json()
    
    # Patterns that indicate non-production units
    EXCLUDE_NAME_PATTERNS = [
        "test", "testing", "dev_", "dev ", "simulator", "sim_",
        "obsolete", "bluetooth", "engineering", "not working",
        "electronic", "spare", "demo", "scrap", "decommiss",
        "broken", "trash", "junk", "removed",
    ]
    
    seen = {}
    for key in ("units", "attention_units"):
        items = data.get(key, [])
        if not isinstance(items, list):
            continue
        for u in items:
            if not isinstance(u, dict):
                continue
            uid = u.get("id")
            if uid is None:
                continue
            if uid not in seen:
                seen[uid] = u
    
    classic_units = []
    for uid, u in seen.items():
        model = u.get("model", "") or ""
        fw = u.get("firmware_version") or ""
        name = (u.get("name") or "").lower()
        
        # Filter 1: Must be always_clean production model
        if model != "always_clean":
            continue
        
        # Filter 2: Must be classic AC001 firmware
        if not isinstance(fw, str) or not fw.startswith("AC001"):
            continue
        
        # Filter 3: Skip dev/test/obsolete by name
        if any(pat in name for pat in EXCLUDE_NAME_PATTERNS):
            continue
        
        # Filter 4: Skip units with weird firmware versions (non-prod builds)
        # Production firmware is AC001V-Rxx.Fx.xx; dev firmware is AC001V-Txx
        if fw.startswith("AC001V-T"):  # T = test build
            continue
        
        classic_units.append({
            "id": uid,
            "name": u.get("name") or f"Unit_{uid}",
            "company": u.get("c_name") or "Unknown",
            "facility": u.get("f_name") or "",
            "firmware_version": fw,
            "model": model,
            "material": u.get("material") or "Unknown",
            "comm_status": u.get("comm_status") or "unknown",
            "last_data_timestamp": u.get("last_data_timestamp") or "",
            "is_idle": u.get("is_idle", 0),
            "color": u.get("color") or "",
        })
    
    return classic_units


def fetch_latest_sensor_values(unit_id, token):
    H = {"Authorization": f"Bearer {token}"}
    now_unix = int(datetime.now(timezone.utc).timestamp())
    yesterday_unix = now_unix - 24 * 3600
    url = f"{BASE_HOST}/api/v1/units/{unit_id}/diagnostics/main/main/export-csv"
    params = {"x_min": yesterday_unix, "zoom": 3, "high_resolution": "true"}
    r = requests.get(url, headers=H, params=params, timeout=60)
    r.raise_for_status()
    
    if len(r.text.strip()) == 0:
        raise ValueError("No data in last 24h")
    
    try:
        df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty data")
    
    if len(df) == 0 or "cctimestamp" not in df.columns:
        raise ValueError("No valid rows")
    
    df = df.sort_values("cctimestamp")
    latest = df.iloc[-1]
    latest_ts = pd.to_datetime(latest["cctimestamp"], unit="s", utc=True)
    
    recent_cutoff = latest["cctimestamp"] - 30 * 60
    recent = df[df["cctimestamp"] >= recent_cutoff]
    if "LoadAmp" in recent.columns:
        recent = recent[recent["LoadAmp"] > 0.1]
    
    values = {}
    for var in KEY_VARS:
        if var in df.columns:
            if len(recent) >= 3:
                values[var] = float(recent[var].mean())
            else:
                v = latest.get(var, np.nan)
                values[var] = float(v) if pd.notna(v) else None
        else:
            values[var] = None
    
    return {
        "latest_timestamp": latest_ts,
        "values": values,
        "n_recent_rows": len(recent),
        "n_total_rows_24h": len(df),
    }


def fetch_unit_extra(unit_id, token):
    """Get galmax and material for detail view."""
    H = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_HOST}/api/v1/units/{unit_id}", headers=H, timeout=30)
    r.raise_for_status()
    data = r.json()
    configs = data.get("configurations") or []
    config = configs[0] if configs else {}
    return {"galmax": config.get("galmax"), "material": config.get("material", "Unknown")}


# ============================================================
# RISK SCORING
# ============================================================
def score_risk_weighted(values, zones):
    per_var = {}
    weighted_score = 0.0
    max_possible_score = 0.0
    
    for var in KEY_VARS:
        z = zones.get(var)
        v = values.get(var)
        weight = VAR_WEIGHTS.get(var, 0)
        max_possible_score += weight
        
        if z is None or v is None or (isinstance(v, float) and pd.isna(v)):
            per_var[var] = {"status": "no_data", "value": v, "zone": z, "weight": weight, "contribution": 0}
            continue
        
        is_in = z["low"] <= v <= z["high"]
        contribution = weight if is_in else 0
        per_var[var] = {
            "status": "in_danger" if is_in else "safe",
            "value": v, "zone": z, "weight": weight, "contribution": contribution,
        }
        weighted_score += contribution
    
    normalized = (weighted_score / max_possible_score) if max_possible_score > 0 else 0
    pct = normalized * 100
    
    if pct >= TIER_CRITICAL_MIN:
        tier, tier_label = "critical", "Critical"
        gradient = "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 50%, #dc2626 100%)"
        emoji, desc = "⛔", "Multiple strong predictors elevated — consider immediate DC drain"
    elif pct >= TIER_WARNING_MIN:
        tier, tier_label = "warning", "Warning"
        gradient = "linear-gradient(135deg, #7c2d12 0%, #ea580c 100%)"
        emoji, desc = "⚠", "Strong predictors triggering — operator attention recommended"
    elif pct >= TIER_WATCH_MIN:
        tier, tier_label = "watch", "Watch"
        gradient = "linear-gradient(135deg, #713f12 0%, #ca8a04 100%)"
        emoji, desc = "◑", "Mild indicators present — monitor closely"
    else:
        tier, tier_label = "normal", "Normal"
        gradient = "linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%)"
        emoji, desc = "✓", "All systems normal — no action needed"
    
    n_in_danger = sum(1 for v in per_var.values() if v["status"] == "in_danger")
    n_total = sum(1 for v in per_var.values() if v["status"] != "no_data")
    
    return {
        "per_var": per_var, "weighted_score": weighted_score,
        "max_possible_score": max_possible_score, "percentage": pct,
        "tier": tier, "tier_label": tier_label, "gradient": gradient,
        "emoji": emoji, "description": desc,
        "n_in_danger": n_in_danger, "n_total": n_total,
    }


def fmt(val, var):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:.{VAR_PRECISION.get(var, 1)}f}"


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "fleet"
if "selected_unit_id" not in st.session_state:
    st.session_state.selected_unit_id = None
if "fleet_data" not in st.session_state:
    st.session_state.fleet_data = None
if "fleet_loaded_at" not in st.session_state:
    st.session_state.fleet_loaded_at = None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🔐 ACCESS")
    email = st.text_input("Email", value="pranav.vilas.chavare.th@dartmouth.edu")
    password = st.text_input("Password", type="password")
    
    st.markdown("---")
    st.markdown("### 🛰 FLEET")
    refresh_button = st.button("⟳ REFRESH FLEET")
    
    if st.session_state.fleet_loaded_at:
        age_sec = (datetime.now(timezone.utc) - st.session_state.fleet_loaded_at).total_seconds()
        st.caption(f"Last refresh: {int(age_sec // 60)} min ago")
    
    st.markdown("---")
    st.markdown("### 📊 SCOPE")
    st.markdown("""
**Classic firmware only**
- always_clean model
- AC001 firmware series

**Excluded:** Newer firmware (AC350), a_series, dev/test units
    """)
    
    st.markdown("---")
    st.caption("Risk tiers: Normal <40%, Watch 40-60%, Warning 60-80%, Critical ≥80%")


# Load danger zones
try:
    zones = load_danger_zones()
except FileNotFoundError:
    st.error(f"Couldn't find `{CSV_PATH}`.")
    st.stop()


# ============================================================
# FLEET LOADER
# ============================================================
def load_fleet(token):
    """Discover and risk-score the classic-firmware always_clean fleet."""
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-logo">⬢</div>
            <h1 class="hero-title">Loading fleet…</h1>
            <p class="hero-subtitle">Discovering classic-firmware always_clean units and fetching live sensor readings.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Step 1/2: Discovering classic-firmware units…")
    try:
        classic_units = fetch_unit_list(token)
    except Exception as e:
        st.error(f"Failed to list units: {e}")
        st.stop()
    
    if not classic_units:
        st.error("No classic-firmware always_clean units found.")
        st.stop()
    
    progress_text.text(f"Step 2/2: Fetching live data for {len(classic_units)} units…")
    
    fleet_results = []
    for i, meta in enumerate(classic_units):
        result = dict(meta)
        result["unit_id"] = meta["id"]
        
        try:
            sensor_data = fetch_latest_sensor_values(meta["id"], token)
            risk = score_risk_weighted(sensor_data["values"], zones)
            result["status"] = "ok"
            result.update({
                "percentage": risk["percentage"],
                "tier": risk["tier"],
                "tier_label": risk["tier_label"],
                "n_in_danger": risk["n_in_danger"],
                "n_total": risk["n_total"],
                "per_var": risk["per_var"],
                "sensor_values": sensor_data["values"],
                "latest_timestamp": sensor_data["latest_timestamp"],
            })
        except Exception as e:
            result["status"] = "error"
            result["error_msg"] = str(e)[:80]
            result["percentage"] = None
            result["tier"] = "error"
        
        fleet_results.append(result)
        progress_bar.progress((i + 1) / len(classic_units))
        time.sleep(0.25)
    
    progress_bar.empty()
    progress_text.empty()
    return fleet_results


# ============================================================
# PAGE: FLEET VIEW
# ============================================================
def render_fleet_view(fleet_data):
    total = len(fleet_data)
    by_tier = {"critical": 0, "warning": 0, "watch": 0, "normal": 0, "error": 0}
    for r in fleet_data:
        by_tier[r.get("tier", "error")] = by_tier.get(r.get("tier", "error"), 0) + 1
    
    st.markdown(f"""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-logo">⬢</div>
            <h1 class="hero-title">CleanPlanet<br><span class="hero-gradient-text">Fleet Monitor</span></h1>
            <p class="hero-subtitle">
                Live DC drain risk monitoring across the classic-firmware always_clean fleet.
                Risk computed from the validated 5-sensor weighted model.
            </p>
            <span class="hero-badge"><span class="hero-badge-dot"></span>CLASSIC FIRMWARE · {total} UNITS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card critical">
            <div class="stat-label">Critical</div>
            <div class="stat-value">{by_tier['critical']}</div>
            <div class="stat-sub">≥ 80% risk</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-label">Warning</div>
            <div class="stat-value">{by_tier['warning']}</div>
            <div class="stat-sub">60-80% risk</div>
        </div>
        <div class="stat-card watch">
            <div class="stat-label">Watch</div>
            <div class="stat-value">{by_tier['watch']}</div>
            <div class="stat-sub">40-60% risk</div>
        </div>
        <div class="stat-card normal">
            <div class="stat-label">Normal</div>
            <div class="stat-value">{by_tier['normal']}</div>
            <div class="stat-sub">&lt; 40% risk</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Offline</div>
            <div class="stat-value">{by_tier['error']}</div>
            <div class="stat-sub">No data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sorted_fleet = sorted(fleet_data, key=lambda r: -(r.get("percentage") if r.get("percentage") is not None else -1))
    
    tier_order = ["critical", "warning", "watch", "normal", "error"]
    tier_names = {
        "critical": "🚨 Critical Risk",
        "warning":  "⚠ Warning",
        "watch":    "◑ Watch",
        "normal":   "✓ Normal",
        "error":    "⊘ Offline / Error",
    }
    
    by_tier_lists = {t: [] for t in tier_order}
    for r in sorted_fleet:
        by_tier_lists.get(r.get("tier", "error"), by_tier_lists["error"]).append(r)
    
    for tier in tier_order:
        units = by_tier_lists[tier]
        if not units:
            continue
        
        st.markdown(f"""
        <div class="tier-section">
            <div class="tier-header {tier}">
                <div class="tier-title">{tier_names[tier]}</div>
                <div class="tier-count">{len(units)} unit{'s' if len(units) != 1 else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        cols_per_row = 4
        for row_start in range(0, len(units), cols_per_row):
            cols = st.columns(cols_per_row)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx >= len(units):
                    break
                with col:
                    render_unit_card(units[idx])


def render_unit_card(unit):
    tier = unit.get("tier", "error")
    pct = unit.get("percentage")
    pct_str = f"{pct:.0f}%" if pct is not None else "OFF"
    
    # Build indicator dots HTML
    dots_html = ""
    if unit.get("per_var"):
        dot_pieces = []
        for var in KEY_VARS:
            info = unit["per_var"].get(var, {})
            status = info.get("status", "no_data")
            meta = VAR_LABELS[var]
            cls = "in-danger" if status == "in_danger" else ("safe" if status == "safe" else "no-data")
            dot_pieces.append(f'<span class="indicator-dot {cls}">{meta["icon"]}</span>')
        dots_html = f'<div class="unit-indicators">{"".join(dot_pieces)}</div>'
    
    # Build body
    if unit["status"] == "error":
        body_html = f'<div class="unit-error-msg">⚠ {unit.get("error_msg", "Unknown")}</div>'
    else:
        meta_text = f'{unit.get("n_in_danger", 0)}/{unit.get("n_total", 0)} in danger'
        body_html = f'{dots_html}<div class="unit-meta">{meta_text}</div>'
    
    # Truncate company name
    company = unit.get("company", "") or "Unknown"
    if len(company) > 28:
        company = company[:25] + "..."
    
    # Build full card as ONE string, no nesting
    card_html = (
        f'<div class="unit-card {tier}">'
        f'<div class="unit-header">'
        f'<div style="flex: 1; min-width: 0;">'
        f'<div class="unit-name">{unit["name"]}</div>'
        f'<div class="unit-id">ID {unit["unit_id"]}</div>'
        f'<div class="unit-company">{company}</div>'
        f'</div>'
        f'<div class="risk-pill {tier}">{pct_str}</div>'
        f'</div>'
        f'<div class="unit-divider"></div>'
        f'{body_html}'
        f'</div>'
    )
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    if st.button(f"Open →", key=f"open_{unit['unit_id']}", use_container_width=True):
        st.session_state.page = "detail"
        st.session_state.selected_unit_id = unit["unit_id"]
        st.rerun()


# ============================================================
# PAGE: UNIT DETAIL
# ============================================================
def render_unit_detail(unit_id, token):
    if st.button("← Back to Fleet"):
        st.session_state.page = "fleet"
        st.session_state.selected_unit_id = None
        st.rerun()
    
    unit = next((u for u in (st.session_state.fleet_data or []) if u["unit_id"] == unit_id), None)
    if not unit:
        st.error(f"Unit {unit_id} not found in fleet data. Try refreshing.")
        return
    
    # Refresh sensor data live
    if unit["status"] != "error":
        try:
            sensor_data = fetch_latest_sensor_values(unit_id, token)
            risk = score_risk_weighted(sensor_data["values"], zones)
            unit.update({
                "percentage": risk["percentage"], "tier": risk["tier"],
                "tier_label": risk["tier_label"], "gradient": risk["gradient"],
                "emoji": risk["emoji"], "description": risk["description"],
                "n_in_danger": risk["n_in_danger"], "n_total": risk["n_total"],
                "per_var": risk["per_var"], "sensor_values": sensor_data["values"],
                "latest_timestamp": sensor_data["latest_timestamp"],
            })
        except Exception as e:
            st.warning(f"Couldn't refresh: {e}")
    
    if unit["status"] == "error":
        st.error(f"Unit {unit_id} is offline: {unit.get('error_msg', 'unknown')}")
        return
    
    # Get extra metadata (galmax)
    try:
        extra = fetch_unit_extra(unit_id, token)
        unit.update(extra)
    except:
        pass
    
    st.markdown(f"""
    <div class="hero" style="padding: 36px;">
        <div class="hero-content">
            <div class="hero-logo">⬢</div>
            <h1 class="hero-title" style="font-size: 36px;">{unit['name']}</h1>
            <p class="hero-subtitle">{unit.get('company', '')} · {unit.get('facility', '')} · Unit ID {unit_id}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    info_cols = st.columns(4)
    info_cols[0].markdown(f"""
    <div class="info-card">
        <div class="info-label">Material</div>
        <div class="info-value" style="font-size:15px;">{unit.get('material', 'Unknown')}</div>
    </div>
    """, unsafe_allow_html=True)
    info_cols[1].markdown(f"""
    <div class="info-card">
        <div class="info-label">Firmware</div>
        <div class="info-value" style="font-size:13px;">{unit.get('firmware_version', 'Unknown')[:22]}</div>
    </div>
    """, unsafe_allow_html=True)
    info_cols[2].markdown(f"""
    <div class="info-card">
        <div class="info-label">GalMax</div>
        <div class="info-value">{unit.get('galmax') or '—'}</div>
    </div>
    """, unsafe_allow_html=True)
    
    age_min = "—"
    if unit.get("latest_timestamp"):
        ts = unit["latest_timestamp"]
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, utc=True)
        age_min = f"{int((datetime.now(timezone.utc) - ts).total_seconds() // 60)} min ago"
    info_cols[3].markdown(f"""
    <div class="info-card">
        <div class="info-label">Last Reading</div>
        <div class="info-value" style="font-size:16px;">{age_min}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><div class="section-bar"></div><h2 class="section-title">Risk Assessment</h2></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="risk-container" style="background: {unit['gradient']};">
        <div class="risk-emoji-big">{unit['emoji']}</div>
        <div><span class="risk-percentage-big">{unit['percentage']:.1f}%</span></div>
        <div><span class="risk-tier-pill">{unit['tier_label']}</span></div>
        <div class="risk-detail-big">{unit['n_in_danger']} of {unit['n_total']} indicators in danger zone</div>
        <div class="risk-detail-big">{unit['description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><div class="section-bar"></div><h2 class="section-title">Risk Contribution (Pareto)</h2></div>', unsafe_allow_html=True)
    
    sorted_vars = sorted(KEY_VARS, key=lambda v: VAR_WEIGHTS.get(v, 0), reverse=True)
    max_w = max(VAR_WEIGHTS.values())
    
    for var in sorted_vars:
        info = unit["per_var"][var]
        meta = VAR_LABELS[var]
        weight = VAR_WEIGHTS.get(var, 0)
        contribution = info["contribution"]
        is_contributing = contribution > 0
        
        bar_pct = (weight / max_w) * 100
        if not is_contributing:
            bar_pct = bar_pct * 0.25
        
        bar_class = "contributing" if is_contributing else "not-contributing"
        contribution_text = f"+{contribution:.2f}" if is_contributing else "0.00"
        
        st.markdown(f"""
        <div class="pareto-row">
            <div class="pareto-icon">{meta['icon']}</div>
            <div class="pareto-name">
                <div class="pareto-name-main">{meta['label']}</div>
                <div class="pareto-name-sub">{var}</div>
            </div>
            <div class="pareto-bar-track">
                <div class="pareto-bar-fill {bar_class}" style="width: {min(bar_pct, 100)}%;"></div>
            </div>
            <div class="pareto-weight">w={weight:.2f}</div>
            <div class="pareto-value">{contribution_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><div class="section-bar"></div><h2 class="section-title">Sensor Breakdown</h2></div>', unsafe_allow_html=True)
    
    var_cols = st.columns(len(KEY_VARS))
    for i, var in enumerate(KEY_VARS):
        info = unit["per_var"][var]
        val = info["value"]
        z = info["zone"]
        meta = VAR_LABELS[var]
        
        with var_cols[i]:
            if info["status"] == "no_data":
                st.markdown(f"""
                <div class="sensor-card-detail no-data">
                    <div class="sensor-icon-d">{meta['icon']}</div>
                    <div class="sensor-code-d">{var}</div>
                    <div class="sensor-label-d">{meta['label']}</div>
                    <div class="sensor-value-d">—</div>
                    <span class="sensor-badge-d badge-no-data-d">No Data</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                val_str = fmt(val, var)
                lo_str = fmt(z["low"], var) if z else "?"
                hi_str = fmt(z["high"], var) if z else "?"
                
                if info["status"] == "in_danger":
                    st.markdown(f"""
                    <div class="sensor-card-detail danger">
                        <div class="sensor-icon-d">{meta['icon']}</div>
                        <div class="sensor-code-d">{var}</div>
                        <div class="sensor-label-d">{meta['label']}</div>
                        <div class="sensor-value-d">{val_str}</div>
                        <div class="sensor-zone-d">⚠ [{lo_str}, {hi_str}]</div>
                        <span class="sensor-badge-d badge-danger-d">In Danger</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sensor-card-detail safe">
                        <div class="sensor-icon-d">{meta['icon']}</div>
                        <div class="sensor-code-d">{var}</div>
                        <div class="sensor-label-d">{meta['label']}</div>
                        <div class="sensor-value-d">{val_str}</div>
                        <div class="sensor-zone-d">✓ Outside [{lo_str}, {hi_str}]</div>
                        <span class="sensor-badge-d badge-safe-d">Safe</span>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
if not password:
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-logo">⬢</div>
            <h1 class="hero-title">CleanPlanet<br><span class="hero-gradient-text">Fleet Monitor</span></h1>
            <p class="hero-subtitle">Enter your credentials in the sidebar to begin.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    token = get_token(email, password)
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()

if refresh_button or st.session_state.fleet_data is None:
    with st.spinner("Loading fleet…"):
        st.session_state.fleet_data = load_fleet(token)
        st.session_state.fleet_loaded_at = datetime.now(timezone.utc)
    st.rerun()

if st.session_state.page == "fleet":
    render_fleet_view(st.session_state.fleet_data)
elif st.session_state.page == "detail":
    render_unit_detail(st.session_state.selected_unit_id, token)