import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from math import erf, sqrt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Dashboard")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload combined GA4 + Ad metrics CSV",
    type=["csv"]
)

if not uploaded_file:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =========================
# DATE SELECTION
# =========================
min_date = df["date"].min()
max_date = df["date"].max()

selected_date = st.date_input(
    "📅 Select analysis date (Today)",
    max_date,
    min_value=min_date,
    max_value=max_date
)

baseline_range = st.date_input(
    "📊 Select baseline range (historical normal)",
    [max_date - timedelta(days=7), max_date - timedelta(days=1)],
    min_value=min_date,
    max_value=max_date - timedelta(days=1)
)

today = pd.to_datetime(selected_date)
yesterday = today - timedelta(days=1)

today_df = df[df["date"] == today]
yesterday_df = df[df["date"] == yesterday]
baseline_df = df[
    (df["date"] >= pd.to_datetime(baseline_range[0])) &
    (df["date"] <= pd.to_datetime(baseline_range[1]))
]

if today_df.empty or baseline_df.empty:
    st.error("Selected date or baseline has no data.")
    st.stop()

# =========================
# METRICS USED
# =========================
METRICS = [
    "revenue",
    "ecpm",
    "sessions",
    "impressions",
    "ctr",
    "fill_rate",
    "viewability",
    "engagement_rate"
]

# =========================
# BASELINE STATS
# =========================
baseline_stats = {}
for m in METRICS:
    if m in baseline_df.columns:
        baseline_stats[m] = {
            "mean": baseline_df[m].mean(),
            "std": baseline_df[m].std() if baseline_df[m].std() > 0 else 1
        }

# =========================
# SIGNAL CALCULATION
# =========================
signals = {}

for m in baseline_stats:
    today_val = today_df[m].values[0]
    base_mean = baseline_stats[m]["mean"]
    base_std = baseline_stats[m]["std"]

    z = (today_val - base_mean) / base_std
    pct_vs_base = ((today_val - base_mean) / base_mean) * 100 if base_mean != 0 else 0

    if not yesterday_df.empty:
        y_val = yesterday_df[m].values[0]
        pct_vs_yday = ((today_val - y_val) / y_val) * 100 if y_val != 0 else 0
    else:
        pct_vs_yday = 0

    signals[m] = {
        "today": today_val,
        "z": z,
        "pct_vs_baseline": pct_vs_base,
        "pct_vs_yesterday": pct_vs_yday
    }

# =========================
# CONFIDENCE FROM Z-SCORE (NO LOGIC CHANGE)
# =========================
def confidence_from_z(z):
    return erf(abs(z) / sqrt(2))

def confidence_label(conf):
    if conf >= 0.8:
        return "High Confidence 🔴"
    elif conf >= 0.5:
        return "Medium Confidence 🟠"
    else:
        return "Low Confidence 🟢"

# =========================
# ROOT CAUSE LOGIC (REVENUE-FIRST)
# =========================
ecpm_drop = signals["ecpm"]["z"] < -1.2
traffic_stable = abs(signals["sessions"]["z"]) < 0.8
quality_ok = (
    signals["ctr"]["z"] >= -0.8 and
    signals["viewability"]["z"] >= -0.8
)

if ecpm_drop and traffic_stable and quality_ok:
    cause = "🔴 Monetization Pressure"
    explanation = (
        "Revenue dropped primarily due to lower advertiser pricing (eCPM), "
        "while traffic and engagement remained stable."
    )
elif signals["sessions"]["z"] < -1:
    cause = "🔴 Traffic Decline"
    explanation = (
        "Revenue drop is driven by fewer users/sessions reaching the site."
    )
elif signals["fill_rate"]["z"] < -1:
    cause = "🔴 Demand / Fill Issue"
    explanation = (
        "Inventory is not being filled consistently, reducing monetization."
    )
else:
    cause = "🟡 Normal Market Movement"
    explanation = (
        "All changes are within normal historical variation."
    )

confidence = confidence_from_z(signals["revenue"]["z"])

# =========================
# EXECUTIVE KPI ROW
# =========================
st.subheader("📌 Executive Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Revenue",
    f"${signals['revenue']['today']:.2f}",
    f"{signals['revenue']['pct_vs_yesterday']:.1f}% vs Yesterday"
)

c2.metric(
    "eCPM",
    f"${signals['ecpm']['today']:.2f}",
    f"{signals['ecpm']['pct_vs_yesterday']:.1f}% vs Yesterday"
)

c3.metric(
    "Sessions",
    int(signals["sessions"]["today"]),
    f"{signals['sessions']['pct_vs_yesterday']:.1f}% vs Yesterday"
)

c4.metric(
    "Confidence",
    f"{int(confidence * 100)}%",
    confidence_label(confidence)
)

st.markdown(f"### **Primary Cause:** {cause}")
st.info(explanation)

# =========================
# CONFIDENCE EXPLANATION
# =========================
st.caption(
    f"🧠 Confidence reflects how far today’s value deviates from baseline using "
    f"standard deviation (Z-score). Higher confidence = less likely random noise."
)

# =========================
# STORYTELLING CHARTS
# =========================
st.subheader("📈 What Changed & Why")

# Revenue & eCPM
rev_ecpm = df.melt(
    id_vars="date",
    value_vars=["revenue", "ecpm"]
)

st.altair_chart(
    alt.Chart(rev_ecpm).mark_line().encode(
        x="date:T",
        y="value:Q",
        color="variable:N",
        tooltip=["date", "value"]
    ).properties(title="Revenue vs eCPM"),
    use_container_width=True
)

# Traffic vs Revenue
st.altair_chart(
    alt.Chart(df).mark_line().encode(
        x="date:T",
        y="sessions:Q",
        tooltip=["date", "sessions"]
    ).properties(title="Traffic Trend"),
    use_container_width=True
)

# Quality Metrics
quality_cols = ["ctr", "viewability", "engagement_rate"]
quality_df = df.melt(id_vars="date", value_vars=quality_cols)

st.altair_chart(
    alt.Chart(quality_df).mark_line().encode(
        x="date:T",
        y="value:Q",
        color="variable:N",
        tooltip=["date", "value"]
    ).properties(title="Traffic Quality Health"),
    use_container_width=True
)

# =========================
# ACTION CHECKLIST
# =========================
st.subheader("🛠 Recommended Actions")

if "Monetization" in cause:
    st.markdown("""
    - Review floor prices & demand sources  
    - Check GEO/device bid pressure  
    - Compare GAM vs exchange demand  
    - Inspect auction competitiveness  
    """)
elif "Traffic" in cause:
    st.markdown("""
    - Investigate traffic source drops  
    - SEO / Discover / Social checks  
    - GA4 tagging validation  
    """)
elif "Demand" in cause:
    st.markdown("""
    - Review ads.txt / blocking  
    - Brand safety & category filters  
    - Ad server errors  
    """)
else:
    st.markdown("""
    - No immediate action  
    - Monitor next few days  
    - Compare with market benchmarks  
    """)

# =========================
# Z-SCORE GUIDE
# =========================
st.sidebar.markdown("### 📘 Z-Score Guide")
st.sidebar.markdown("""
- |Z| < 1 → Normal  
- 1–2 → Mild anomaly  
- 2–3 → Strong signal  
- >3 → Extreme issue  

Confidence is derived directly from Z-score.
""")

st.caption("Revenue Intelligence Dashboard • Dynamic • Baseline-aware • Error-safe")
