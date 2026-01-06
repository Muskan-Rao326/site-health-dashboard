import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Dashboard")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Daily Metrics CSV",
    type=["csv"]
)

if not uploaded_file:
    st.info("Please upload a valid daily metrics CSV file.")
    st.stop()

df = pd.read_csv(uploaded_file)

# =====================================================
# DATA CLEANING & VALIDATION
# =====================================================
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

REQUIRED_COLS = [
    "date", "site_name", "sessions", "revenue", "ecpm",
    "ctr", "fill_rate", "viewability", "engagement_rate",
    "pageviews_per_session", "impressions_per_session",
    "clicks_per_session"
]

missing = set(REQUIRED_COLS) - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🔍 Filters")

site = st.sidebar.selectbox(
    "Select Site",
    sorted(df["site_name"].unique())
)

df = df[df["site_name"] == site]

baseline_days = st.sidebar.slider(
    "Baseline window (days)",
    min_value=7,
    max_value=30,
    value=14
)

# =====================================================
# DAY DEFINITIONS
# =====================================================
latest_day = df["date"].max()
yesterday = latest_day - timedelta(days=1)

today_df = df[df["date"] == latest_day]
yesterday_df = df[df["date"] == yesterday]
baseline_df = df[df["date"] < latest_day].tail(baseline_days)

if today_df.empty or yesterday_df.empty:
    st.error("Not enough data to compare today vs yesterday.")
    st.stop()

# =====================================================
# METRIC LIST
# =====================================================
METRICS = [
    "revenue", "ecpm", "sessions",
    "ctr", "fill_rate", "viewability",
    "engagement_rate",
    "pageviews_per_session",
    "impressions_per_session",
    "clicks_per_session"
]

# =====================================================
# METRIC COMPARISON FUNCTION
# =====================================================
def compare(metric):
    today = today_df[metric].values[0]
    yesterday_val = yesterday_df[metric].values[0]
    baseline_mean = baseline_df[metric].mean()

    return {
        "today": today,
        "vs_yesterday_pct": ((today - yesterday_val) / yesterday_val) * 100,
        "vs_baseline_pct": ((today - baseline_mean) / baseline_mean) * 100
    }

signals = {m: compare(m) for m in METRICS}

# =====================================================
# KPI ROW (EXECUTIVE VIEW)
# =====================================================
st.subheader("📌 Executive Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue",
    f"${signals['revenue']['today']:.2f}",
    f"{signals['revenue']['vs_yesterday_pct']:.1f}% vs Yesterday"
)

col2.metric(
    "eCPM",
    f"${signals['ecpm']['today']:.2f}",
    f"{signals['ecpm']['vs_yesterday_pct']:.1f}% vs Yesterday"
)

col3.metric(
    "Sessions",
    int(signals['sessions']['today']),
    f"{signals['sessions']['vs_yesterday_pct']:.1f}% vs Yesterday"
)

col4.metric(
    "Viewability",
    f"{signals['viewability']['today']:.1f}%",
    f"{signals['viewability']['vs_yesterday_pct']:.1f}% vs Yesterday"
)

# =====================================================
# ROOT CAUSE LOGIC (RULE-BASED, STABLE)
# =====================================================
ecpm_down = signals["ecpm"]["vs_baseline_pct"] < -10
traffic_flat = abs(signals["sessions"]["vs_baseline_pct"]) < 5
quality_ok = (
    signals["ctr"]["vs_baseline_pct"] > -5 and
    signals["viewability"]["vs_baseline_pct"] > -5
)

if ecpm_down and traffic_flat and quality_ok:
    cause = "🔴 Monetization Pressure"
    reason = "eCPM dropped while traffic & quality stayed stable."
elif signals["sessions"]["vs_baseline_pct"] < -10:
    cause = "🔴 Traffic Loss"
    reason = "Revenue decline driven by lower traffic."
elif signals["fill_rate"]["vs_baseline_pct"] < -10:
    cause = "🔴 Demand / Fill Issue"
    reason = "Lower fill rate reduced monetization."
else:
    cause = "🟡 Normal Market Movement"
    reason = "All changes are within expected variance."

st.markdown(f"### **Primary Cause: {cause}**")
st.info(reason)

# =====================================================
# STORYTELLING CHARTS
# =====================================================
st.subheader("📈 What Changed & Why")

def line_chart(metric, title):
    return alt.Chart(df).mark_line(point=True).encode(
        x="date:T",
        y=metric,
        tooltip=["date", metric]
    ).properties(title=title)

st.altair_chart(
    line_chart("revenue", "Revenue Trend") +
    line_chart("ecpm", "eCPM Trend"),
    use_container_width=True
)

st.altair_chart(
    line_chart("sessions", "Traffic Trend"),
    use_container_width=True
)

quality_df = df.melt(
    id_vars="date",
    value_vars=["ctr", "viewability", "engagement_rate"],
    var_name="metric",
    value_name="value"
)

st.altair_chart(
    alt.Chart(quality_df).mark_line().encode(
        x="date:T",
        y="value:Q",
        color="metric:N",
        tooltip=["date", "metric", "value"]
    ).properties(title="Traffic Quality Health"),
    use_container_width=True
)

# =====================================================
# DELIVERY INTELLIGENCE
# =====================================================
st.subheader("📦 Delivery Efficiency")

st.altair_chart(
    alt.Chart(df).mark_line(point=True).encode(
        x="date:T",
        y="impressions_per_session:Q",
        tooltip=["date", "impressions_per_session"]
    ).properties(title="Impressions per Session"),
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================
st.caption(
    "Baseline-aware • Yesterday comparison • Root-cause driven • No Z-score abuse"
)
