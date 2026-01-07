import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Dashboard")

# =====================
# HELPERS
# =====================
def z_label(z):
    if z >= 2:
        return "🟢 Exceptional"
    elif z >= 1:
        return "🟢 Strong"
    elif z > -1:
        return "⚪ Normal"
    elif z > -1.5:
        return "🟡 Weak"
    elif z > -2:
        return "🟠 Poor"
    else:
        return "🔴 Critical"

def z_color(z):
    if z <= -2:
        return "#ff4d4f"
    elif z <= -1.5:
        return "#ffa940"
    elif z <= -1:
        return "#ffec3d"
    elif z <= 1:
        return "#f0f0f0"
    else:
        return "#95de64"

def confidence_from_z(z):
    return min(0.99, abs(z) / 3)

def probability_text(z):
    if abs(z) < 1:
        return "≈ 68% of outcomes fall here (normal)"
    elif abs(z) < 2:
        return "≈ 95% confidence (unlikely variation)"
    else:
        return "≈ 99.7% confidence (abnormal)"

# =====================
# LOAD DATA
# =====================
uploaded_file = st.file_uploader("Upload daily metrics CSV", type=["csv"])

if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =====================
# DATE SELECTION
# =====================
selected_date = st.date_input(
    "Select date to analyze",
    max_value=df["date"].max().date(),
    value=df["date"].max().date()
)

selected_date = pd.to_datetime(selected_date)

today_df = df[df["date"] == selected_date]
yesterday_df = df[df["date"] == selected_date - timedelta(days=1)]

baseline_df = df[
    (df["date"] < selected_date) &
    (df["date"] >= selected_date - timedelta(days=7))
]

if today_df.empty or baseline_df.empty:
    st.error("Not enough data for selected date or baseline.")
    st.stop()

# =====================
# METRICS
# =====================
METRICS = [
    "revenue", "ecpm", "impressions", "requests",
    "ctr", "match_rate", "delivery_rate", "viewability"
]

signals = {}

for m in METRICS:
    if m not in df.columns:
        continue

    today = today_df[m].values[0]
    yday = yesterday_df[m].values[0] if not yesterday_df.empty else np.nan
    mean = baseline_df[m].mean()
    std = baseline_df[m].std() or 1

    z = (today - mean) / std
    signals[m] = {
        "today": today,
        "yesterday": yday,
        "baseline": mean,
        "z": z,
        "label": z_label(z),
        "confidence": confidence_from_z(z),
        "prob": probability_text(z)
    }

# =====================
# KPI VIEW
# =====================
st.subheader("📌 Executive KPIs")

kpi_cols = st.columns(4)
for i, metric in enumerate(["revenue", "ecpm", "impressions", "ctr"]):
    s = signals[metric]
    with kpi_cols[i]:
        st.markdown(
            f"""
            <div style="
                padding:16px;
                border-radius:12px;
                background-color:{z_color(s['z'])};
                text-align:center">
                <h3>{metric.upper()}</h3>
                <h2>{s['today']:.2f}</h2>
                <p>Z: {s['z']:.2f} — {s['label']}</p>
                <p>Confidence: {int(s['confidence']*100)}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================
# SUMMARY
# =====================
st.subheader("🧠 Auto-Generated Explanation")

st.markdown(f"""
**Revenue status:** {signals['revenue']['label']}  
Z-Score **{signals['revenue']['z']:.2f}** → {signals['revenue']['prob']}

- Revenue today: **{signals['revenue']['today']:.2f}**
- Yesterday: **{signals['revenue']['yesterday']:.2f}**
- 7-day baseline: **{signals['revenue']['baseline']:.2f}**

This indicates revenue performance is
**{signals['revenue']['label'].lower()} relative to recent normal behavior**.
""")

# =====================
# STORY CHARTS
# =====================
st.subheader("📈 What Changed")

chart_df = df[df["date"] >= selected_date - timedelta(days=14)]

rev_chart = alt.Chart(chart_df).mark_line().encode(
    x="date:T",
    y="revenue:Q",
    tooltip=["date", "revenue"]
).properties(title="Revenue Trend")

ecpm_chart = alt.Chart(chart_df).mark_line(color="orange").encode(
    x="date:T",
    y="ecpm:Q"
)

st.altair_chart(rev_chart + ecpm_chart, use_container_width=True)

# =====================
# ALERTING LOGIC
# =====================
st.subheader("🚨 Alerts")

alerts = []

if signals["revenue"]["z"] <= -2:
    alerts.append("🔴 Revenue is critically below baseline")

if signals["ecpm"]["z"] <= -1.5:
    alerts.append("🟠 eCPM degradation detected")

if signals["impressions"]["z"] <= -1.5:
    alerts.append("🟠 Impression delivery issue")

if not alerts:
    st.success("✅ No critical anomalies detected")
else:
    for a in alerts:
        st.error(a)

# =====================
# Z-SCORE GUIDE
# =====================
with st.expander("📘 How to Read Z-Scores"):
    st.markdown("""
    **Z-Score = distance from recent normal behavior**

    - 🟢 ≥ +2 → Exceptional
    - 🟢 +1 → +2 → Strong
    - ⚪ −1 → +1 → Normal
    - 🟡 −1 → −1.5 → Weak
    - 🟠 −1.5 → −2 → Poor
    - 🔴 ≤ −2 → Critical

    Baseline = **previous 7 days (excluding selected day)**.
    """)

st.caption("Revenue Intelligence Dashboard • Dynamic • Baseline-aware • Alert-driven")
