"""
CleanPlanet DC Risk Monitor — Apple-grade UI
Run with: python -m streamlit run app.py

Requires danger_ranges.csv in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import traceback
from datetime import datetime, timezone
from io import StringIO

BASE_HOST = "https://recycling.cleanplanetchemical.com"

# === UPDATED: 5 data-validated predictors ===
KEY_VARS = ["Transducer04", "Temp08", "Energy", "DAC1", "Feedback06"]

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
# STYLING — dark, premium, high contrast
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp {
    background: #0a0e1a;
    background-image:
        radial-gradient(at 0% 0%, hsla(160, 70%, 25%, 0.25) 0px, transparent 50%),
        radial-gradient(at 100% 0%, hsla(190, 70%, 30%, 0.2) 0px, transparent 50%),
        radial-gradient(at 50% 100%, hsla(150, 80%, 20%, 0.15) 0px, transparent 50%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e6edf3;
}
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1500px; }

/* HERO */
.hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(6, 78, 59, 0.85) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 28px;
    padding: 56px 56px 52px 56px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
}
.hero::before {
    content: ""; position: absolute; top: -200px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
    animation: floatSlow 12s ease-in-out infinite;
}
.hero::after {
    content: ""; position: absolute; bottom: -150px; left: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(8, 145, 178, 0.1) 0%, transparent 70%);
    animation: floatSlow 15s ease-in-out infinite reverse;
}
.hero-content { position: relative; z-index: 1; }
.hero-logo {
    display: inline-flex; align-items: center; justify-content: center;
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    border-radius: 14px;
    font-size: 26px; color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
}
.hero-title {
    color: #f8fafc; font-size: 52px; font-weight: 800;
    margin: 0; letter-spacing: -2px; line-height: 1.05;
}
.hero-gradient-text {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    color: rgba(226, 232, 240, 0.7); font-size: 18px;
    margin-top: 16px; font-weight: 400;
    max-width: 680px; line-height: 1.6;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(16, 185, 129, 0.12); color: #6ee7b7;
    padding: 9px 18px; border-radius: 100px;
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    margin-top: 24px;
    border: 1px solid rgba(16, 185, 129, 0.25);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.hero-badge-dot {
    width: 8px; height: 8px;
    background: #10b981; border-radius: 50%;
    box-shadow: 0 0 12px #10b981;
    animation: pulseLED 1.8s infinite;
}

/* SECTION HEADERS */
.section-header { display: flex; align-items: center; gap: 14px; margin: 48px 0 24px 0; }
.section-bar {
    width: 4px; height: 28px;
    background: linear-gradient(180deg, #10b981, #06b6d4);
    border-radius: 4px;
}
.section-title { color: #f1f5f9; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
.section-caption { color: rgba(148, 163, 184, 0.8); font-size: 14px; margin: -12px 0 24px 18px; }

/* ZONE CARDS */
.zone-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 20px;
    padding: 32px 18px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.zone-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #10b981, #06b6d4, transparent);
    opacity: 0; transition: opacity 0.4s ease;
}
.zone-card:hover {
    transform: translateY(-6px);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(16, 185, 129, 0.2);
}
.zone-card:hover::before { opacity: 1; }
.zone-icon {
    width: 56px; height: 56px;
    margin: 0 auto 18px auto;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 182, 212, 0.15) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; color: #6ee7b7;
    transition: all 0.4s ease;
}
.zone-card:hover .zone-icon {
    transform: scale(1.1);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(6, 182, 212, 0.25) 100%);
}
.zone-code {
    color: rgba(148, 163, 184, 0.7);
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.zone-label { color: #e2e8f0; font-size: 14px; font-weight: 600; margin-bottom: 16px; }
.zone-range {
    color: #f8fafc; font-size: 18px; font-weight: 700;
    letter-spacing: -0.5px;
    font-family: 'JetBrains Mono', monospace;
}
.zone-arrow { color: rgba(16, 185, 129, 0.7); margin: 0 4px; }
.zone-unit {
    color: rgba(148, 163, 184, 0.6);
    font-size: 11px; margin-top: 8px; font-weight: 500;
    letter-spacing: 1px; text-transform: uppercase;
}

/* INFO CARDS */
.info-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 22px 24px;
    transition: all 0.3s ease;
    backdrop-filter: blur(15px);
}
.info-card:hover { border-color: rgba(16, 185, 129, 0.3); transform: translateY(-2px); }
.info-label {
    color: rgba(148, 163, 184, 0.7);
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.info-value { color: #f1f5f9; font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
.info-sub { color: rgba(148, 163, 184, 0.6); font-size: 12px; margin-top: 4px; }

/* RISK HERO */
.risk-container {
    border-radius: 28px;
    padding: 60px 40px;
    text-align: center;
    color: white;
    margin: 28px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 30px 80px rgba(0,0,0,0.4);
    animation: fadeUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.risk-container::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.18), transparent 40%),
                radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1), transparent 40%);
}
.risk-emoji {
    font-size: 96px; line-height: 1; margin-bottom: 16px;
    animation: bounce 3s infinite ease-in-out;
    filter: drop-shadow(0 8px 24px rgba(0,0,0,0.3));
    position: relative; z-index: 1;
}
.risk-level {
    font-size: 52px; font-weight: 900; letter-spacing: -2px;
    margin: 8px 0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.25);
    position: relative; z-index: 1; line-height: 1;
}
.risk-detail {
    font-size: 18px; opacity: 0.95; font-weight: 500;
    position: relative; z-index: 1; margin-top: 12px;
}
.risk-progress {
    margin: 28px auto 0 auto;
    height: 8px;
    background: rgba(255,255,255,0.15);
    border-radius: 100px; overflow: hidden;
    width: 55%; position: relative; z-index: 1;
}
.risk-progress-bar {
    height: 100%; background: white;
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 16px rgba(255,255,255,0.6);
}
.risk-description {
    margin-top: 24px; font-size: 14px; opacity: 0.85; font-weight: 500;
    position: relative; z-index: 1; letter-spacing: 0.3px;
}

/* SENSOR CARDS */
.sensor-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 22px;
    padding: 28px 18px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative; overflow: hidden;
    backdrop-filter: blur(20px);
}
.sensor-card.safe {
    border-color: rgba(16, 185, 129, 0.4);
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.6) 100%);
}
.sensor-card.danger {
    border-color: rgba(239, 68, 68, 0.5);
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
    animation: dangerGlow 2s infinite;
}
.sensor-card.no-data { border-color: rgba(148, 163, 184, 0.15); opacity: 0.6; }
.sensor-card:hover { transform: translateY(-8px); }
.sensor-card.safe:hover { box-shadow: 0 20px 50px rgba(16, 185, 129, 0.2); }
.sensor-card.danger:hover { box-shadow: 0 20px 50px rgba(239, 68, 68, 0.3); }

.sensor-icon {
    width: 60px; height: 60px;
    margin: 0 auto 18px auto;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    transition: all 0.4s ease;
}
.sensor-card.safe .sensor-icon {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.2) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
}
.sensor-card.danger .sensor-icon {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.2) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    animation: shake 2.5s infinite;
}
.sensor-card.no-data .sensor-icon {
    background: rgba(148, 163, 184, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: rgba(148, 163, 184, 0.5);
}
.sensor-card:hover .sensor-icon { transform: scale(1.1); }
.sensor-code {
    color: rgba(148, 163, 184, 0.7);
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.sensor-label { color: #e2e8f0; font-size: 13px; font-weight: 600; margin-bottom: 18px; min-height: 32px; }
.sensor-value {
    font-size: 32px; font-weight: 800; letter-spacing: -1px;
    line-height: 1; display: block;
    font-family: 'JetBrains Mono', monospace;
}
.sensor-card.safe .sensor-value { color: #6ee7b7; }
.sensor-card.danger .sensor-value { color: #fca5a5; }
.sensor-card.no-data .sensor-value { color: rgba(148, 163, 184, 0.4); }
.sensor-unit { color: rgba(148, 163, 184, 0.7); font-size: 12px; font-weight: 500; margin-top: 6px; letter-spacing: 1px; }
.sensor-zone {
    color: rgba(148, 163, 184, 0.6);
    font-size: 10px; margin-top: 14px; padding-top: 12px;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    font-family: 'JetBrains Mono', monospace;
}
.sensor-badge {
    display: inline-block;
    padding: 5px 12px; border-radius: 100px;
    font-size: 10px; font-weight: 700;
    margin-top: 12px;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}
.badge-safe { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); animation: blink 1.8s infinite; }
.badge-no-data { background: rgba(148, 163, 184, 0.15); color: rgba(148, 163, 184, 0.7); }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid rgba(148, 163, 184, 0.08);
}
section[data-testid="stSidebar"] > div { padding-top: 32px; }
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-size: 13px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 12px !important;
    font-family: 'JetBrains Mono', monospace;
}
section[data-testid="stSidebar"] label {
    color: rgba(226, 232, 240, 0.85) !important;
    font-size: 13px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 8px !important;
}
section[data-testid="stSidebar"] input {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: rgba(16, 185, 129, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
    background: rgba(30, 41, 59, 0.8) !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: transparent !important; border: none !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    padding: 14px 24px !important; font-size: 15px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
    width: 100% !important; margin-top: 8px !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px rgba(16, 185, 129, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.15) inset !important;
}
section[data-testid="stSidebar"] .stButton button:active { transform: translateY(0) !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: rgba(203, 213, 225, 0.75) !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] code {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #6ee7b7 !important;
    padding: 2px 6px !important; border-radius: 4px !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.15) !important; margin: 24px 0 !important; }

/* STREAMLIT WIDGETS */
[data-testid="stAlert"] {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
}
[data-testid="stAlert"][kind="info"] { background: rgba(56, 189, 248, 0.08) !important; }
[data-testid="stAlert"][kind="error"] { background: rgba(239, 68, 68, 0.1) !important; border-color: rgba(239, 68, 68, 0.4) !important; }
.stSpinner > div { border-top-color: #10b981 !important; border-right-color: rgba(16, 185, 129, 0.3) !important; }
.stSpinner > div + div { color: #e2e8f0 !important; font-weight: 500 !important; }

/* ANIMATIONS */
@keyframes pulseLED {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 12px #10b981; }
    50% { opacity: 0.6; transform: scale(0.85); box-shadow: 0 0 6px #10b981; }
}
@keyframes floatSlow { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(30px, -30px); } }
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes dangerGlow { 0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } 50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.25); } }
@keyframes shake { 0%, 100% { transform: rotate(0); } 25% { transform: rotate(-4deg); } 75% { transform: rotate(4deg); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.65; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

#MainMenu, footer, header { visibility: hidden; }
.stCaption, [data-testid="stCaptionContainer"] { color: rgba(148, 163, 184, 0.7) !important; }
hr { border-color: rgba(148, 163, 184, 0.1) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DANGER ZONES
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
        json={"email": email, "password": password},
        timeout=60,
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
        raise ValueError(
            f"Unit {unit_id} returned empty data — likely offline or retired. "
            f"Try an active unit: 129, 60, 201, 132."
        )
    
    try:
        df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Unit {unit_id} has no sensor data in the last 24h. "
            f"Try an active unit: 129, 60, 201, 132."
        )
    
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
        "unit_id": unit_id,
        "latest_timestamp": latest_ts,
        "values": values,
        "n_recent_rows": len(recent),
        "n_total_rows_24h": len(df),
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
    }


def fmt(val, var):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:.{VAR_PRECISION[var]}f}"


# ============================================================
# RISK SCORING — now 6 tiers (0-5 vars)
# ============================================================
def score_risk(values, zones):
    per_var = {}
    in_danger = 0
    for var in KEY_VARS:
        z = zones.get(var)
        v = values.get(var)
        if z is None or v is None or (isinstance(v, float) and pd.isna(v)):
            per_var[var] = {"status": "no_data", "value": v, "zone": z}
            continue
        is_in = z["low"] <= v <= z["high"]
        per_var[var] = {"status": "in_danger" if is_in else "safe", "value": v, "zone": z}
        if is_in:
            in_danger += 1
    
    total = sum(1 for v in per_var.values() if v["status"] != "no_data")
    
    levels = {
        0: ("Safe",            "linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%)", "✓",  "All systems normal — no action needed"),
        1: ("Low Risk",        "linear-gradient(135deg, #365314 0%, #65a30d 100%)",              "◐",  "One indicator elevated — monitor"),
        2: ("Medium Risk",     "linear-gradient(135deg, #713f12 0%, #ca8a04 100%)",              "◑",  "Two indicators elevated — review soon"),
        3: ("Moderate Risk",   "linear-gradient(135deg, #7c2d12 0%, #ea580c 100%)",              "⚠",  "Three indicators elevated — operator attention recommended"),
        4: ("High Risk",       "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%)",              "⚠",  "Four indicators elevated — early DC drain advised"),
        5: ("Critical Risk",   "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 50%, #dc2626 100%)", "⛔", "All indicators elevated — consider immediate DC drain"),
    }
    name, gradient, emoji, desc = levels.get(in_danger, ("Unknown", "#1e293b", "?", ""))
    
    return {
        "per_var": per_var,
        "in_danger_count": in_danger,
        "total_vars": total,
        "risk_level": name,
        "risk_gradient": gradient,
        "risk_emoji": emoji,
        "risk_description": desc,
    }


# ============================================================
# UI
# ============================================================

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-logo">⬢</div>
        <h1 class="hero-title">
            CleanPlanet<br>
            <span class="hero-gradient-text">Risk Monitor</span>
        </h1>
        <p class="hero-subtitle">
            Real-time monitoring of DC drain health across the always_clean fleet.
            Detects bad-drain conditions before code 215 fires.
        </p>
        <span class="hero-badge">
            <span class="hero-badge-dot"></span>
            LIVE · DC-FILLED-TOO-QUICKLY PREDICTOR · 5-SIGNAL MODEL
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load zones
try:
    zones = load_danger_zones()
except FileNotFoundError:
    st.error(f"Couldn't find `{CSV_PATH}` in this folder.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.code(traceback.format_exc())
    st.stop()

if not zones:
    st.error(f"{CSV_PATH} loaded but empty.")
    st.stop()

# SIDEBAR
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
    
    st.markdown("---")
    st.caption("Data-validated 5-signal model calibrated on units 129, 44, 201, 199")

# DANGER ZONES
st.markdown("""
<div class="section-header">
    <div class="section-bar"></div>
    <h2 class="section-title">Calibrated Danger Zones</h2>
</div>
""", unsafe_allow_html=True)
st.markdown('<p class="section-caption">Value ranges where bad-drain batches concentrate beyond clean operation.</p>', unsafe_allow_html=True)

cols = st.columns(len(KEY_VARS))
for i, var in enumerate(KEY_VARS):
    z = zones.get(var)
    meta = VAR_LABELS[var]
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
                <div class="zone-range">{fmt(z['low'], var)}<span class="zone-arrow">→</span>{fmt(z['high'], var)}</div>
                <div class="zone-unit">{meta['unit']}</div>
            </div>
            """, unsafe_allow_html=True)

if not password:
    st.info("👈  Enter your password in the sidebar to begin checking units.")
    st.stop()

# RISK CHECK
if check_button:
    try:
        with st.spinner(f"Fetching live telemetry for unit {unit_id_input}..."):
            token = get_token(email, password)
            unit_info = fetch_unit_info(unit_id_input, token)
            sensor_data = fetch_latest_sensor_values(unit_id_input, token)
    except Exception as e:
        st.error(f"⚠ {type(e).__name__}: {e}")
        st.stop()
    
    risk = score_risk(sensor_data["values"], zones)
    
    # UNIT INFO
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Unit Profile</h2>
    </div>
    """, unsafe_allow_html=True)
    
    info_cols = st.columns(4)
    info_cols[0].markdown(f"""
    <div class="info-card">
        <div class="info-label">Unit</div>
        <div class="info-value">{unit_info['name']}</div>
        <div class="info-sub">ID {unit_id_input}</div>
    </div>
    """, unsafe_allow_html=True)
    info_cols[1].markdown(f"""
    <div class="info-card">
        <div class="info-label">Company</div>
        <div class="info-value" style="font-size:15px;">{unit_info['company']}</div>
    </div>
    """, unsafe_allow_html=True)
    info_cols[2].markdown(f"""
    <div class="info-card">
        <div class="info-label">Material</div>
        <div class="info-value" style="font-size:15px;">{unit_info['material']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    age_min = (datetime.now(timezone.utc) - sensor_data["latest_timestamp"]).total_seconds() / 60
    info_cols[3].markdown(f"""
    <div class="info-card">
        <div class="info-label">Last Reading</div>
        <div class="info-value" style="font-size:18px;">{age_min:.0f} min ago</div>
        <div class="info-sub">{sensor_data['latest_timestamp'].strftime('%H:%M UTC')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # RISK HERO
    st.markdown("""
    <div class="section-header">
        <div class="section-bar"></div>
        <h2 class="section-title">Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    progress_pct = (risk['in_danger_count'] / max(risk['total_vars'], 1)) * 100
    
    st.markdown(f"""
    <div class="risk-container" style="background: {risk['risk_gradient']};">
        <div class="risk-emoji">{risk['risk_emoji']}</div>
        <div class="risk-level">{risk['risk_level']}</div>
        <div class="risk-detail">
            {risk['in_danger_count']} of {risk['total_vars']} indicators in danger zone
        </div>
        <div class="risk-progress">
            <div class="risk-progress-bar" style="width: {progress_pct}%;"></div>
        </div>
        <div class="risk-description">{risk['risk_description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # SENSORS
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
        
        with var_cols[i]:
            if info["status"] == "no_data":
                st.markdown(f"""
                <div class="sensor-card no-data">
                    <div class="sensor-icon">{meta['icon']}</div>
                    <div class="sensor-code">{var}</div>
                    <div class="sensor-label">{meta['label']}</div>
                    <span class="sensor-value">—</span>
                    <span class="sensor-badge badge-no-data">No Data</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                val_str = fmt(val, var)
                lo_str = fmt(z['low'], var)
                hi_str = fmt(z['high'], var)
                
                if info["status"] == "in_danger":
                    st.markdown(f"""
                    <div class="sensor-card danger">
                        <div class="sensor-icon">{meta['icon']}</div>
                        <div class="sensor-code">{var}</div>
                        <div class="sensor-label">{meta['label']}</div>
                        <span class="sensor-value">{val_str}</span>
                        <div class="sensor-unit">{meta['unit']}</div>
                        <div class="sensor-zone">⚠ Danger: [{lo_str}, {hi_str}]</div>
                        <span class="sensor-badge badge-danger">In Danger</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sensor-card safe">
                        <div class="sensor-icon">{meta['icon']}</div>
                        <div class="sensor-code">{var}</div>
                        <div class="sensor-label">{meta['label']}</div>
                        <span class="sensor-value">{val_str}</span>
                        <div class="sensor-unit">{meta['unit']}</div>
                        <div class="sensor-zone">✓ Outside [{lo_str}, {hi_str}]</div>
                        <span class="sensor-badge badge-safe">Safe</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    foot_cols = st.columns(3)
    foot_cols[0].caption(f"📊 Rows in last 24h: **{sensor_data['n_total_rows_24h']:,}**")
    foot_cols[1].caption(f"⏱ Averaging window: **{sensor_data['n_recent_rows']} rows**")
    foot_cols[2].caption(f"🕐 Latest: **{sensor_data['latest_timestamp'].strftime('%Y-%m-%d %H:%M UTC')}**")

else:
    st.info("👈 Enter a unit ID and click **CHECK RISK** to begin.")