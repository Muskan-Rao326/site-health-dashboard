import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Performance Diagnosis")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload daily metrics CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =========================
# DATE RANGE SELECTION
# =========================
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.date_input(
    "Select baseline date range",
    [min_date, max_date - timedelta(days=1)],
    min_value=min_date,
    max_value=max_date
)

baseline_df = df[(df["date"] >= pd.to_datetime(date_range[0])) &
                 (df["date"] <= pd.to_datetime(date_range[1]))]

latest_day = df["date"].max()
today_df = df[df["date"] == latest_day]

# =========================
# METRIC DEFINITIONS
# =========================
METRICS = {
    "traffic": ["sessions", "users", "pageviews"],
    "quality": ["ctr", "engagement_rate", "viewability"],
    "monetization": ["ecpm", "fill_rate", "revenue"]
}

# =========================
# BASELINE STATS
# =========================
baseline_stats = {}
for group in METRICS.values():
    for m in group:
        if m in baseline_df.columns:
            baseline_stats[m] = {
                "mean": baseline_df[m].mean(),
                "std": baseline_df[m].std() or 1
            }

# =========================
# TODAY VS BASELINE (Z-SCORE)
# =========================
signals = {}
for m, stats in baseline_stats.items():
    today_val = today_df[m].values[0]
    z = (today_val - stats["mean"]) / stats["std"]
    signals[m] = {
        "today": today_val,
        "z": z,
        "pct_change": (today_val - stats["mean"]) / stats["mean"] * 100
    }

# =========================
# REVENUE ROOT CAUSE LOGIC
# =========================
ecpm_drop = signals.get("ecpm", {}).get("z", 0) < -1.2
traffic_stable = abs(signals.get("sessions", {}).get("z", 0)) < 0.8
quality_good = (
    signals.get("ctr", {}).get("z", 0) >= -0.8 and
    signals.get("viewability", {}).get("z", 0) >= -0.8
)

if ecpm_drop and traffic_stable and quality_good:
    primary_cause = "🔴 Monetization Pressure"
    explanation = (
        "Revenue declined mainly due to a significant drop in eCPM while traffic "
        "and engagement quality remained stable or improved. This indicates "
        "advertiser-side pricing pressure or demand softening."
    )
    confidence = 0.85

elif signals.get("sessions", {}).get("z", 0) < -1:
    primary_cause = "🔴 Traffic Loss"
    explanation = (
        "Revenue declined primarily due to a significant drop in traffic volume. "
        "Monetization efficiency remains normal."
    )
    confidence = 0.8

elif signals.get("fill_rate", {}).get("z", 0) < -1:
    primary_cause = "🔴 Demand / Fill Issue"
    explanation = (
        "Revenue declined due to reduced fill rate, indicating weakened advertiser "
        "demand or inventory blocking."
    )
    confidence = 0.75

else:
    primary_cause = "🟡 Normal Market Movement"
    explanation = (
        "Observed changes fall within normal historical variation. No structural "
        "issues detected. Continued monitoring recommended."
    )
    confidence = 0.6

# =========================
# KPI ROW (EXECUTIVE VIEW)
# =========================
st.subheader("📌 Executive Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue",
    f"${signals['revenue']['today']:.2f}",
    f"{signals['revenue']['pct_change']:.1f}%"
)

col2.metric(
    "eCPM",
    f"${signals['ecpm']['today']:.2f}",
    f"{signals['ecpm']['pct_change']:.1f}%"
)

col3.metric(
    "Sessions",
    int(signals.get("sessions", {}).get("today", 0)),
    f"{signals.get('sessions', {}).get('pct_change', 0):.1f}%"
)

col4.metric(
    "Confidence",
    f"{int(confidence * 100)}%"
)

st.markdown(f"### **Primary Cause:** {primary_cause}")
st.info(explanation)

# =========================
# STORYTELLING CHARTS
# =========================
st.subheader("📈 What Changed & Why")

# Revenue vs eCPM
chart1 = alt.Chart(df).mark_line().encode(
    x="date:T",
    y="revenue:Q",
    tooltip=["date", "revenue"]
).properties(title="Revenue Trend")

chart2 = alt.Chart(df).mark_line(color="orange").encode(
    x="date:T",
    y="ecpm:Q",
    tooltip=["date", "ecpm"]
)

st.altair_chart(chart1 + chart2, use_container_width=True)

# Sessions vs Revenue (Decoupling)
chart3 = alt.Chart(df).mark_line().encode(
    x="date:T",
    y="sessions:Q",
    tooltip=["date", "sessions"]
).properties(title="Traffic vs Revenue Decoupling")

st.altair_chart(chart3, use_container_width=True)

# Quality Metrics
quality_cols = [c for c in ["ctr", "viewability", "engagement_rate"] if c in df.columns]
quality_df = df.melt(id_vars="date", value_vars=quality_cols)

chart4 = alt.Chart(quality_df).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="variable:N",
    tooltip=["date", "variable", "value"]
).properties(title="Traffic Quality Health")

st.altair_chart(chart4, use_container_width=True)

# =========================
# ACTION CHECKLIST
# =========================
st.subheader("🛠 Recommended Actions")

if "Monetization" in primary_cause:
    st.markdown("""
    - Check advertiser demand by GEO & device
    - Review floor price changes
    - Compare GAM vs exchange bid density
    - Validate auction pressure logs
    """)

elif "Traffic" in primary_cause:
    st.markdown("""
    - Investigate traffic source drops
    - Check SEO / Discover / Social referrals
    - Validate GA4 tagging & consent
    """)

elif "Demand" in primary_cause:
    st.markdown("""
    - Inspect blocked categories / ads.txt
    - Review brand safety restrictions
    - Check ad server errors
    """)

else:
    st.markdown("""
    - No immediate action required
    - Monitor next 3–5 days
    - Compare with market benchmarks
    """)

st.caption("Revenue Intelligence Dashboard • Baseline-aware • False-positive protected")
