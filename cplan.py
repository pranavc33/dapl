"""
CleanPlanet DC Risk Monitor — with Pareto Bar Chart + Variable Strength Explainer
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import traceback
import plotly.graph_objects as go
from datetime import datetime, timezone
from io import StringIO

BASE_HOST = "https://recycling.cleanplanetchemical.com"

KEY_VARS = ["Transducer04", "Temp08", "Energy", "DAC1", "Feedback06"]

# AUC and other discrimination metrics from our analysis
VAR_METRICS = {
    "Transducer04": {"auc": 0.803, "cohens_d": 0.681, "mw_p": 0.0162, "mi": 0.181,
                     "bad_mean": 1756.10, "clean_mean": 1937.97, "direction": "lower in bad"},
    "Temp08":       {"auc": 0.789, "cohens_d": 0.764, "mw_p": 0.0222, "mi": 0.073,
                     "bad_mean": 38.82,   "clean_mean": 44.73,    "direction": "lower in bad"},
    "Energy":       {"auc": 0.789, "cohens_d": 0.631, "mw_p": 0.0222, "mi": 0.092,
                     "bad_mean": 243660.97, "clean_mean": 107422.70, "direction": "higher in bad"},
    "Feedback06":   {"auc": 0.781, "cohens_d": 0.610, "mw_p": 0.0271, "mi": 0.146,
                     "bad_mean": 56.05,   "clean_mean": 68.22,    "direction": "lower in bad"},
    "DAC1":         {"auc": 0.768, "cohens_d": 0.597, "mw_p": 0.0385, "mi": 0.146,
                     "bad_mean": 2378.89, "clean_mean": 3412.06,  "direction": "lower in bad"},
}

VAR_AUC = {v: VAR_METRICS[v]["auc"] for v in KEY_VARS}
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
CSV_PATH = "danger_ranges.csv"

st.set_page_config(
    page_title="CleanPlanet Risk Monitor",
    page_icon="⬢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
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
    border-radius: 28px; padding: 56px; margin-bottom: 40px;
    position: relative; overflow: hidden;
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
    border-radius: 14px; font-size: 26px; color: white; margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
}
.hero-title { color: #f8fafc; font-size: 52px; font-weight: 800; letter-spacing: -2px; margin: 0; line-height: 1.05; }
.hero-gradient-text {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: rgba(226, 232, 240, 0.7); font-size: 18px; margin-top: 16px; max-width: 680px; line-height: 1.6; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(16, 185, 129, 0.12); color: #6ee7b7;
    padding: 9px 18px; border-radius: 100px;
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    margin-top: 24px; border: 1px solid rgba(16, 185, 129, 0.25);
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
}
.hero-badge-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulseLED 1.8s infinite; }

.section-header { display: flex; align-items: center; gap: 14px; margin: 48px 0 24px 0; }
.section-bar { width: 4px; height: 28px; background: linear-gradient(180deg, #10b981, #06b6d4); border-radius: 4px; }
.section-title { color: #f1f5f9; font-size: 22px; font-weight: 700; margin: 0; }
.section-caption { color: rgba(148, 163, 184, 0.8); font-size: 14px; margin: -12px 0 24px 18px; }

/* EXPLAINER BOX */
.explainer-box {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px; padding: 24px 28px;
    margin: 20px 0;
}
.explainer-step {
    background: rgba(15, 23, 42, 0.5);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
}
.explainer-step-num {
    color: #38bdf8; font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;
}
.explainer-step-text { color: #e2e8f0; font-size: 14px; line-height: 1.6; }
.explainer-step-text strong { color: #6ee7b7; }

/* ZONE CARDS */
.zone-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 20px; padding: 28px 18px; text-align: center;
    transition: all 0.4s ease; height: 100%; backdrop-filter: blur(20px);
}
.zone-card:hover { transform: translateY(-6px); border-color: rgba(16, 185, 129, 0.4); }
.zone-icon {
    width: 56px; height: 56px; margin: 0 auto 18px auto;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(6, 182, 212, 0.15));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 16px; display: flex; align-items: center; justify-content: center;
    font-size: 26px; color: #6ee7b7;
}
.zone-code { color: rgba(148, 163, 184, 0.7); font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
.zone-label { color: #e2e8f0; font-size: 14px; font-weight: 600; margin-bottom: 14px; }
.zone-range { color: #f8fafc; font-size: 18px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.zone-weight { color: #6ee7b7; font-size: 11px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; }

.info-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px; padding: 22px 24px;
    backdrop-filter: blur(15px);
}
.info-label { color: rgba(148, 163, 184, 0.7); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.info-value { color: #f1f5f9; font-size: 20px; font-weight: 700; }
.info-sub { color: rgba(148, 163, 184, 0.6); font-size: 12px; margin-top: 4px; }

/* RISK HERO */
.risk-container {
    border-radius: 28px; padding: 60px 40px; text-align: center;
    color: white; margin: 28px 0;
    position: relative; overflow: hidden;
    box-shadow: 0 30px 80px rgba(0,0,0,0.4);
    animation: fadeUp 0.6s ease-out;
}
.risk-container::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.18), transparent 40%);
}
.risk-emoji { font-size: 72px; margin-bottom: 8px; animation: bounce 3s infinite ease-in-out; filter: drop-shadow(0 8px 24px rgba(0,0,0,0.3)); position: relative; z-index: 1; }
.risk-percentage {
    font-size: 110px; font-weight: 900; letter-spacing: -5px; line-height: 1;
    color: white; text-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    font-family: 'JetBrains Mono', monospace; position: relative; z-index: 1; display: inline-block;
}
.risk-percent-sign { font-size: 60px; font-weight: 700; opacity: 0.85; margin-left: 8px; }
.risk-tier-label {
    font-size: 22px; font-weight: 700; color: white;
    letter-spacing: 0.5px; padding: 10px 22px;
    background: rgba(255, 255, 255, 0.18); border-radius: 100px;
    border: 1px solid rgba(255, 255, 255, 0.3); backdrop-filter: blur(10px);
    position: relative; z-index: 1; text-transform: uppercase;
    display: inline-block; margin-top: 12px;
}
.risk-detail { font-size: 16px; opacity: 0.9; margin-top: 20px; position: relative; z-index: 1; }
.risk-progress { margin: 28px auto 0 auto; height: 10px; background: rgba(255,255,255,0.2); border-radius: 100px; overflow: hidden; width: 60%; position: relative; z-index: 1; }
.risk-progress-bar { height: 100%; background: white; border-radius: 100px; transition: width 1s ease; box-shadow: 0 0 16px rgba(255,255,255,0.6); }
.risk-description { margin-top: 24px; font-size: 14px; opacity: 0.85; font-style: italic; position: relative; z-index: 1; }

/* SENSOR CARDS */
.sensor-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 22px; padding: 28px 18px; text-align: center;
    transition: all 0.4s ease; height: 100%;
}
.sensor-card.safe { border-color: rgba(16, 185, 129, 0.4); background: linear-gradient(180deg, rgba(16, 185, 129, 0.08), rgba(15, 23, 42, 0.6)); }
.sensor-card.danger { border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(180deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.6)); animation: dangerGlow 2s infinite; }
.sensor-card.no-data { border-color: rgba(148, 163, 184, 0.15); opacity: 0.6; }
.sensor-card:hover { transform: translateY(-6px); }
.sensor-icon { width: 56px; height: 56px; margin: 0 auto 18px auto; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 26px; }
.sensor-card.safe .sensor-icon { background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 182, 212, 0.2)); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; }
.sensor-card.danger .sensor-icon { background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(220, 38, 38, 0.2)); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; animation: shake 2.5s infinite; }
.sensor-card.no-data .sensor-icon { background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); color: rgba(148, 163, 184, 0.5); }
.sensor-code { color: rgba(148, 163, 184, 0.7); font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
.sensor-label { color: #e2e8f0; font-size: 13px; font-weight: 600; margin-bottom: 16px; min-height: 32px; }
.sensor-value { font-size: 32px; font-weight: 800; letter-spacing: -1px; display: block; font-family: 'JetBrains Mono', monospace; }
.sensor-card.safe .sensor-value { color: #6ee7b7; }
.sensor-card.danger .sensor-value { color: #fca5a5; }
.sensor-card.no-data .sensor-value { color: rgba(148, 163, 184, 0.4); }
.sensor-unit { color: rgba(148, 163, 184, 0.7); font-size: 12px; margin-top: 6px; }
.sensor-weight-badge { display: inline-block; padding: 4px 10px; border-radius: 100px; font-size: 10px; font-weight: 700; margin-top: 10px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); font-family: 'JetBrains Mono', monospace; }
.sensor-zone { color: rgba(148, 163, 184, 0.6); font-size: 10px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(148, 163, 184, 0.1); font-family: 'JetBrains Mono', monospace; }
.sensor-badge { display: inline-block; padding: 5px 12px; border-radius: 100px; font-size: 10px; font-weight: 700; margin-top: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'JetBrains Mono', monospace; }
.badge-safe { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); animation: blink 1.8s infinite; }
.badge-no-data { background: rgba(148, 163, 184, 0.15); color: rgba(148, 163, 184, 0.7); }

/* SIDEBAR */
section[data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid rgba(148, 163, 184, 0.08); }
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 13px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'JetBrains Mono', monospace; }
section[data-testid="stSidebar"] label { color: rgba(226, 232, 240, 0.85) !important; font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px; }
section[data-testid="stSidebar"] input { background: rgba(30, 41, 59, 0.6) !important; border: 1px solid rgba(148, 163, 184, 0.2) !important; color: #f1f5f9 !important; border-radius: 10px !important; padding: 12px 16px !important; }
section[data-testid="stSidebar"] input:focus { border-color: rgba(16, 185, 129, 0.6) !important; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important; }
section[data-testid="stSidebar"] .stButton button { background: linear-gradient(135deg, #10b981, #06b6d4) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 14px !important; width: 100% !important; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35) !important; margin-top: 8px !important; }
section[data-testid="stSidebar"] .stButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 14px 30px rgba(16, 185, 129, 0.5) !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: rgba(203, 213, 225, 0.75) !important; font-size: 13px !important; }
section[data-testid="stSidebar"] code { background: rgba(16, 185, 129, 0.15) !important; color: #6ee7b7 !important; padding: 2px 6px !important; border-radius: 4px !important; }
section[data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.15) !important; }

[data-testid="stAlert"] { background: rgba(30, 41, 59, 0.6) !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 12px !important; color: #e2e8f0 !important; }
.stSpinner > div { border-top-color: #10b981 !important; }
.stSpinner > div + div { color: #e2e8f0 !important; }

@keyframes pulseLED { 0%, 100% { opacity: 1; box-shadow: 0 0 12px #10b981; } 50% { opacity: 0.6; box-shadow: 0 0 6px #10b981; } }
@keyframes floatSlow { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(30px, -30px); } }
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes dangerGlow { 0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } 50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.25); } }
@keyframes shake { 0%, 100% { transform: rotate(0); } 25% { transform: rotate(-4deg); } 75% { transform: rotate(4deg); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.65; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING + API
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


@st.cache_resource(ttl=3000)
def get_token(email, password):
    r = requests.post(
        f"{BASE_HOST}/api/v1/auth/login",
        json={"email": email, "password": password}, timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_latest_sensor_values(unit_id, token):
    H = {"Authorization": f"Bearer {token}"}
    now_unix = int(datetime.now(timezone.utc).timestamp())
    yesterday_unix = now_unix - 24 * 3600
    url = f"{BASE_HOST}/api/v1/units/{unit_id}/diagnostics/main/main/export-csv"
    params = {"x_min": yesterday_unix, "zoom": 3, "high_resolution": "true"}
    r = requests.get(url, headers=H, params=params, timeout=60)
    r.raise_for_status()
    
    if len(r.text.strip()) == 0:
        raise ValueError(f"Unit {unit_id} returned empty data — likely offline. Try 129, 60, 201, 132.")
    
    try:
        df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Unit {unit_id} has no data in the last 24h. Try 129, 60, 201, 132.")
    
    if len(df) == 0:
        raise ValueError(f"Unit {unit_id} returned no rows.")
    
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
        "unit_id": unit_id, "latest_timestamp": latest_ts,
        "values": values,
        "n_recent_rows": len(recent), "n_total_rows_24h": len(df),
    }


def fetch_unit_info(unit_id, token):
    H = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_HOST}/api/v1/units/{unit_id}", headers=H, timeout=30)
    r.raise_for_status()
    data = r.json()
    unit = data.get("unit") or {}
    configs = data.get("configurations") or []
    config = configs[0] if configs else {}
    return {
        "name": unit.get("name", "Unknown"),
        "company": unit.get("c_name", "Unknown"),
        "material": config.get("material", "Unknown"),
        "galmax": config.get("galmax"),
    }


def fmt(val, var):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:.{VAR_PRECISION.get(var, 1)}f}"


# ============================================================
# WEIGHTED RISK SCORING
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
    
    normalized_score = weighted_score / max_possible_score if max_possible_score > 0 else 0
    pct = normalized_score * 100
    
    if pct < 50:
        name, gradient, emoji, desc = "Medium Risk", "linear-gradient(135deg, #713f12 0%, #ca8a04 100%)", "◑", "Mild indicators present — monitor closely"
    elif pct < 70:
        name, gradient, emoji, desc = "High Risk", "linear-gradient(135deg, #7c2d12 0%, #ea580c 100%)", "⚠", "Strong predictors triggering — operator attention recommended"
    elif pct < 85:
        name, gradient, emoji, desc = "Very High Risk", "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%)", "⚠", "Multiple strong predictors elevated — early DC drain advised"
    else:
        name, gradient, emoji, desc = "Critical Risk", "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 50%, #dc2626 100%)", "⛔", "All key predictors elevated — consider immediate DC drain"
    
    n_in_danger = sum(1 for v in per_var.values() if v["status"] == "in_danger")
    n_total = sum(1 for v in per_var.values() if v["status"] != "no_data")
    
    return {
        "per_var": per_var,
        "weighted_score": weighted_score, "max_possible_score": max_possible_score,
        "normalized_score": normalized_score, "percentage": pct,
        "n_in_danger": n_in_danger, "n_total": n_total,
        "risk_level": name, "risk_gradient": gradient,
        "risk_emoji": emoji, "risk_description": desc,
    }


# ============================================================
# PARETO BAR CHART (Plotly)
# ============================================================
def build_pareto_chart(risk):
    """Build a proper Pareto bar chart with cumulative line, sorted by importance."""
    # Sort by weight (descending)
    sorted_vars = sorted(KEY_VARS, key=lambda v: VAR_WEIGHTS.get(v, 0), reverse=True)
    
    var_codes = sorted_vars
    var_labels = [VAR_LABELS[v]["label"] for v in sorted_vars]
    weights_pct = [VAR_WEIGHTS[v] * 100 for v in sorted_vars]
    cumulative = np.cumsum(weights_pct).tolist()
    
    # Colors — bars red if currently contributing to risk, otherwise green
    colors = []
    for v in sorted_vars:
        info = risk["per_var"].get(v, {})
        if info.get("status") == "in_danger":
            colors.append("#ef4444")
        elif info.get("status") == "safe":
            colors.append("#10b981")
        else:
            colors.append("#64748b")
    
    fig = go.Figure()
    
    # Bars
    fig.add_trace(go.Bar(
        x=var_codes,
        y=weights_pct,
        text=[f"{w:.1f}%" for w in weights_pct],
        textposition="outside",
        textfont=dict(color="#f1f5f9", size=14, family="Inter"),
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.2)", width=2),
        ),
        hovertemplate="<b>%{customdata[0]}</b><br>Variable: %{x}<br>Weight: %{y:.1f}%<br>Status: %{customdata[1]}<extra></extra>",
        customdata=[[lbl, "IN DANGER" if c == "#ef4444" else ("SAFE" if c == "#10b981" else "NO DATA")] 
                    for lbl, c in zip(var_labels, colors)],
        name="Individual Weight",
        showlegend=True,
    ))
    
    # Cumulative line
    fig.add_trace(go.Scatter(
        x=var_codes,
        y=cumulative,
        mode="lines+markers+text",
        text=[f"{c:.0f}%" for c in cumulative],
        textposition="top center",
        textfont=dict(color="#06b6d4", size=13, family="Inter"),
        line=dict(color="#06b6d4", width=3),
        marker=dict(size=12, color="#0e7490", line=dict(color="white", width=2)),
        yaxis="y2",
        name="Cumulative %",
        hovertemplate="<b>Cumulative through %{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    
    # 80% reference line
    fig.add_hline(
        y=80, yref="y2",
        line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5, dash="dash"),
        annotation_text="80% threshold",
        annotation_position="right",
        annotation_font=dict(color="#94a3b8", size=11),
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(t=40, b=80, l=60, r=60),
        font=dict(family="Inter", color="#e6edf3"),
        xaxis=dict(
            title=dict(text="Sensor Variable (ranked by predictive importance)", font=dict(color="#cbd5e1", size=13)),
            tickfont=dict(color="#cbd5e1", size=12),
            showgrid=False,
            linecolor="rgba(148, 163, 184, 0.3)",
        ),
        yaxis=dict(
            title=dict(text="Individual Weight (%)", font=dict(color="#cbd5e1", size=13)),
            tickfont=dict(color="#cbd5e1", size=11),
            gridcolor="rgba(148, 163, 184, 0.1)",
            range=[0, max(weights_pct) * 1.4],
        ),
        yaxis2=dict(
            title=dict(text="Cumulative Weight (%)", font=dict(color="#06b6d4", size=13)),
            tickfont=dict(color="#06b6d4", size=11),
            overlaying="y",
            side="right",
            range=[0, 110],
            showgrid=False,
        ),
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top",
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.2)", borderwidth=1,
            font=dict(color="#e6edf3", size=12),
        ),
        bargap=0.3,
    )
    
    return fig


# ============================================================
# UI
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-logo">⬢</div>
        <h1 class="hero-title">CleanPlanet<br><span class="hero-gradient-text">Risk Monitor</span></h1>
        <p class="hero-subtitle">
            Weighted multi-signal monitoring of DC drain health. Risk is calculated based on each variable's
            empirical predictive strength (AUC), not raw counts.
        </p>
        <span class="hero-badge"><span class="hero-badge-dot"></span>LIVE · WEIGHTED 5-SIGNAL MODEL</span>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    zones = load_danger_zones()
except FileNotFoundError:
    st.error(f"Couldn't find `{CSV_PATH}`.")
    st.stop()

if not zones:
    st.error("danger_ranges.csv loaded but empty.")
    st.stop()

with st.sidebar:
    st.markdown("### 🔐 ACCESS")
    email = st.text_input("Email", value="pranav.vilas.chavare.th@dartmouth.edu")
    password = st.text_input("Password", type="password")
    st.markdown("---")
    st.markdown("### 🎯 TARGET UNIT")
    unit_id_input = st.number_input("Unit ID", min_value=1, max_value=999, value=129, step=1, label_visibility="collapsed")
    check_button = st.button("CHECK RISK")
    st.markdown("---")
    st.markdown("### 💡 SUGGESTED")
    st.markdown("""
**Active offenders:**
- `129` — St. Johns
- `201` — RPM Wood Finishes
- `60` — PPC Rome

**Healthy controls:**
- `132` — PPC
- `194` — PPC
    """)

st.markdown("""
<div class="section-header">
    <div class="section-bar"></div>
    <h2 class="section-title">Calibrated Danger Zones</h2>
</div>
""", unsafe_allow_html=True)
st.markdown('<p class="section-caption">Each variable carries a weight based on its predictive strength (AUC) for code 215.</p>', unsafe_allow_html=True)

cols = st.columns(len(KEY_VARS))
for i, var in enumerate(KEY_VARS):
    z = zones.get(var)
    meta = VAR_LABELS[var]
    weight = VAR_WEIGHTS.get(var, 0)
    with cols[i]:
        if z is None:
            st.markdown(f"""
            <div class="zone-card">
                <div class="zone-icon">{meta['icon']}</div>
                <div class="zone-code">{var}</div>
                <div class="zone-label">{meta['label']}</div>
                <div class="zone-range" style="color: rgba(148,163,184,0.5);">N/A</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="zone-card">
                <div class="zone-icon">{meta['icon']}</div>
                <div class="zone-code">{var}</div>
                <div class="zone-label">{meta['label']}</div>
                <div class="zone-range">{fmt(z['low'], var)} → {fmt(z['high'], var)}</div>
                <div class="zone-weight">weight: {weight:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

if not password:
    st.info("👈  Enter your password in the sidebar to begin.")
    st.stop()

if check_button:
    try:
        with st.spinner(f"Fetching live telemetry for unit {unit_id_input}..."):
            token = get_token(email, password)
            unit_info = fetch_unit_info(unit_id_input, token)
            sensor_data = fetch_latest_sensor_values(unit_id_input, token)
    except Exception as e:
        st.error(f"⚠ {type(e).__name__}: {e}")
        st.stop()
    
    risk = score_risk_weighted(sensor_data["values"], zones)
    
    # UNIT INFO
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Unit Profile</h2>
    </div>
    """, unsafe_allow_html=True)
    
    info_cols = st.columns(4)
    info_cols[0].markdown(f"""<div class="info-card"><div class="info-label">Unit</div><div class="info-value">{unit_info['name']}</div><div class="info-sub">ID {unit_id_input}</div></div>""", unsafe_allow_html=True)
    info_cols[1].markdown(f"""<div class="info-card"><div class="info-label">Company</div><div class="info-value" style="font-size:15px;">{unit_info['company']}</div></div>""", unsafe_allow_html=True)
    info_cols[2].markdown(f"""<div class="info-card"><div class="info-label">Material</div><div class="info-value" style="font-size:15px;">{unit_info['material']}</div></div>""", unsafe_allow_html=True)
    
    age_min = (datetime.now(timezone.utc) - sensor_data["latest_timestamp"]).total_seconds() / 60
    info_cols[3].markdown(f"""<div class="info-card"><div class="info-label">Last Reading</div><div class="info-value" style="font-size:18px;">{age_min:.0f} min ago</div><div class="info-sub">{sensor_data['latest_timestamp'].strftime('%H:%M UTC')}</div></div>""", unsafe_allow_html=True)
    
    # RISK HERO
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="risk-container" style="background: {risk['risk_gradient']};">
        <div class="risk-emoji">{risk['risk_emoji']}</div>
        <div style="position: relative; z-index: 1;">
            <span class="risk-percentage">{risk['percentage']:.1f}<span class="risk-percent-sign">%</span></span>
        </div>
        <div style="margin-top: 8px; position: relative; z-index: 1;">
            <span class="risk-tier-label">{risk['risk_level']}</span>
        </div>
        <div class="risk-detail">{risk['n_in_danger']} of {risk['n_total']} indicators in danger zone</div>
        <div class="risk-progress"><div class="risk-progress-bar" style="width: {risk['percentage']}%;"></div></div>
        <div class="risk-description">{risk['risk_description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SECTION 1: HOW WE DETERMINED VARIABLE STRENGTH
    # ============================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">How Variable Strength Was Determined</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="section-caption">A walk-through of how we identified the 5 strongest predictors of code 215 and assigned each a weight.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explainer-box">
        <div class="explainer-step">
            <div class="explainer-step-num">STEP 1 — DATA COLLECTION</div>
            <div class="explainer-step-text">
            We collected sensor data from the <strong>4 always_clean units with the most code 215 events</strong>:
            Unit 129 (St. Johns), Unit 44 (ChemChamp), Unit 201 (RPM Wood Finishes), and Unit 199 (ChemChamp).
            For each batch, we labeled it as either <strong>"bad"</strong> (drain failed → code 215 fired) or <strong>"clean"</strong> (drain worked normally).
            </div>
        </div>
        <div class="explainer-step">
            <div class="explainer-step-num">STEP 2 — COMPUTE FEATURES</div>
            <div class="explainer-step-text">
            For each batch, we computed the <strong>average value of every numeric sensor</strong> across that batch
            (while the scraper was active). This gave us ~50 candidate variables per batch.
            </div>
        </div>
        <div class="explainer-step">
            <div class="explainer-step-num">STEP 3 — RUN 4 STATISTICAL TESTS</div>
            <div class="explainer-step-text">
            We tested every variable against four independent statistical metrics:
            <br><br>
            <strong>• AUC (Area Under ROC Curve):</strong> How well can this variable alone predict bad vs clean? (0.5 = random, 1.0 = perfect)<br>
            <strong>• Cohen's d:</strong> Effect size — how many standard deviations apart the means are<br>
            <strong>• Mann-Whitney p-value:</strong> Statistical significance of the difference<br>
            <strong>• Mutual Information:</strong> How much info the variable carries about the bad/clean label
            </div>
        </div>
        <div class="explainer-step">
            <div class="explainer-step-num">STEP 4 — SCORE EACH VARIABLE 0-4</div>
            <div class="explainer-step-text">
            Each variable scored <strong>0 to 4</strong> based on how many tests it passed:
            AUC ≥ 0.75, Cohen's d ≥ 0.8, MW p &lt; 0.05, MI ≥ 0.05. Variables passing 3+ tests were kept.
            </div>
        </div>
        <div class="explainer-step">
            <div class="explainer-step-num">STEP 5 — ASSIGN WEIGHTS FROM AUC</div>
            <div class="explainer-step-text">
            For the 5 surviving variables, we used <strong>AUC as the weight basis</strong>:
            <br><br>
            <code style="color:#6ee7b7;">weight = (AUC - 0.5) / total_above_random</code>
            <br><br>
            This normalizes weights so they sum to 1.0, with each variable's weight proportional to how much above random chance it predicts code 215.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Table of variable scores
    st.markdown("**Detailed scores for the 5 selected variables:**")
    score_table = []
    for var in KEY_VARS:
        m = VAR_METRICS[var]
        score_table.append({
            "Variable": var,
            "Description": VAR_LABELS[var]["label"],
            "AUC": f"{m['auc']:.3f}",
            "Cohen's d": f"{m['cohens_d']:.3f}",
            "MW p-value": f"{m['mw_p']:.4f}",
            "Mutual Info": f"{m['mi']:.3f}",
            "Direction": m['direction'],
            "Bad Mean": f"{m['bad_mean']:.1f}",
            "Clean Mean": f"{m['clean_mean']:.1f}",
            "Weight": f"{VAR_WEIGHTS[var]:.3f}",
        })
    score_df = pd.DataFrame(score_table)
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    
    # ============================================================
    # SECTION 2: PARETO ANALYSIS
    # ============================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Pareto Analysis — Risk Contribution Breakdown</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explainer-box">
        <div class="explainer-step">
            <div class="explainer-step-num">WHAT IS PARETO ANALYSIS?</div>
            <div class="explainer-step-text">
            Pareto analysis is based on the <strong>80/20 principle</strong>: typically 80% of effects come from 20% of causes.
            We use it here to show <strong>which variables contribute the most to code 215 prediction</strong>, so you can focus
            attention on the strongest signals.
            <br><br>
            The chart below ranks variables by predictive importance. The <strong>bars</strong> show each variable's individual contribution,
            while the <strong>cyan line</strong> shows the cumulative total as you add each variable from left to right.
            </div>
        </div>
        <div class="explainer-step">
            <div class="explainer-step-num">HOW TO READ THIS CHART</div>
            <div class="explainer-step-text">
            <strong>• Bar color:</strong> <span style="color:#ef4444;">Red</span> = currently in danger zone (contributing to risk now). <span style="color:#10b981;">Green</span> = safe.<br>
            <strong>• Bar height:</strong> The variable's intrinsic predictive weight (independent of current state)<br>
            <strong>• Cumulative line:</strong> If you only monitored the top N variables, you'd capture this % of predictive power<br>
            <strong>• 80% dashed line:</strong> The classic Pareto threshold — the point at which most signal is captured
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # The Pareto chart
    fig = build_pareto_chart(risk)
    st.plotly_chart(fig, use_container_width=True)
    
    # Pareto interpretation for current unit
    sorted_vars_by_weight = sorted(KEY_VARS, key=lambda v: VAR_WEIGHTS.get(v, 0), reverse=True)
    cumulative_at_2 = sum(VAR_WEIGHTS[v] for v in sorted_vars_by_weight[:2]) * 100
    cumulative_at_3 = sum(VAR_WEIGHTS[v] for v in sorted_vars_by_weight[:3]) * 100
    
    top_in_danger = [v for v in sorted_vars_by_weight if risk["per_var"][v]["status"] == "in_danger"]
    
    interp_text = f"""
**Pareto interpretation for this analysis:**

- **Top 2 variables** ({sorted_vars_by_weight[0]} + {sorted_vars_by_weight[1]}) account for **{cumulative_at_2:.1f}%** of predictive power
- **Top 3 variables** capture **{cumulative_at_3:.1f}%** of predictive power
- All 5 variables together = 100%
"""
    if top_in_danger:
        interp_text += f"\n- **Currently contributing to risk:** {', '.join(top_in_danger)} ({len(top_in_danger)} variable{'s' if len(top_in_danger) != 1 else ''} in danger zone)"
    else:
        interp_text += "\n- **No variables currently in danger zone** for this unit"
    
    st.markdown(interp_text)
    
    # ============================================================
    # SENSOR BREAKDOWN
    # ============================================================
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Sensor Breakdown</h2>
    </div>
    """, unsafe_allow_html=True)
    
    var_cols = st.columns(len(KEY_VARS))
    for i, var in enumerate(KEY_VARS):
        info = risk["per_var"][var]
        val = info["value"]
        z = info["zone"]
        meta = VAR_LABELS[var]
        weight = VAR_WEIGHTS.get(var, 0)
        
        with var_cols[i]:
            if info["status"] == "no_data":
                st.markdown(f"""<div class="sensor-card no-data"><div class="sensor-icon">{meta['icon']}</div><div class="sensor-code">{var}</div><div class="sensor-label">{meta['label']}</div><span class="sensor-value">—</span><span class="sensor-weight-badge">w={weight:.2f}</span><span class="sensor-badge badge-no-data">No Data</span></div>""", unsafe_allow_html=True)
            else:
                val_str = fmt(val, var)
                lo_str = fmt(z["low"], var)
                hi_str = fmt(z["high"], var)
                if info["status"] == "in_danger":
                    st.markdown(f"""<div class="sensor-card danger"><div class="sensor-icon">{meta['icon']}</div><div class="sensor-code">{var}</div><div class="sensor-label">{meta['label']}</div><span class="sensor-value">{val_str}</span><div class="sensor-unit">{meta['unit']}</div><span class="sensor-weight-badge">w={weight:.2f} · +{info['contribution']:.2f}</span><div class="sensor-zone">⚠ [{lo_str}, {hi_str}]</div><span class="sensor-badge badge-danger">In Danger</span></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="sensor-card safe"><div class="sensor-icon">{meta['icon']}</div><div class="sensor-code">{var}</div><div class="sensor-label">{meta['label']}</div><span class="sensor-value">{val_str}</span><div class="sensor-unit">{meta['unit']}</div><span class="sensor-weight-badge">w={weight:.2f}</span><div class="sensor-zone">✓ Outside [{lo_str}, {hi_str}]</div><span class="sensor-badge badge-safe">Safe</span></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    foot_cols = st.columns(3)
    foot_cols[0].caption(f"📊 Rows in last 24h: **{sensor_data['n_total_rows_24h']:,}**")
    foot_cols[1].caption(f"⏱ Window: **{sensor_data['n_recent_rows']} rows (last 30 min)**")
    foot_cols[2].caption(f"🕐 Latest: **{sensor_data['latest_timestamp'].strftime('%Y-%m-%d %H:%M UTC')}**")

else:
    st.info("👈 Enter a unit ID and click **CHECK RISK** to begin.")