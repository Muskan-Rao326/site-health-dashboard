import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Monitor")

# =============================
# DATA UPLOAD
# =============================
uploaded_file = st.file_uploader("📂 Upload daily metrics CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to start analysis")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =============================
# DATE LOGIC
# =============================
today = df["date"].max()
yesterday = today - timedelta(days=1)

today_df = df[df["date"] == today]
yesterday_df = df[df["date"] == yesterday]

if today_df.empty or yesterday_df.empty:
    st.error("Need at least two consecutive days of data")
    st.stop()

# =============================
# BASELINE RANGE
# =============================
st.sidebar.header("📅 Baseline (What is normal?)")

baseline_range = st.sidebar.date_input(
    "Select baseline date range",
    [df["date"].min(), today - timedelta(days=1)]
)

baseline_df = df[
    (df["date"] >= pd.to_datetime(baseline_range[0])) &
    (df["date"] <= pd.to_datetime(baseline_range[1]))
]

# =============================
# METRICS
# =============================
METRICS = [
    "revenue", "ecpm", "sessions", "users",
    "pageviews", "ctr", "fill_rate", "viewability"
]

# =============================
# BASELINE STATS
# =============================
baseline = {}
for m in METRICS:
    if m in df.columns:
        baseline[m] = {
            "mean": baseline_df[m].mean(),
            "std": baseline_df[m].std() or 1
        }

# =============================
# SIGNAL ENGINE
# =============================
def pct_change(curr, prev):
    return ((curr - prev) / prev * 100) if prev else 0

signals = {}

for m in baseline:
    today_val = today_df[m].values[0]
    y_val = yesterday_df[m].values[0]

    signals[m] = {
        "today": today_val,
        "vs_yesterday": pct_change(today_val, y_val),
        "vs_baseline": pct_change(today_val, baseline[m]["mean"]),
        "z": (today_val - baseline[m]["mean"]) / baseline[m]["std"]
    }

# =============================
# ROOT CAUSE ENGINE
# =============================
revenue_drop = signals["revenue"]["vs_yesterday"] < -5
ecpm_drop = signals["ecpm"]["z"] < -1.2
traffic_stable = abs(signals["sessions"]["z"]) < 0.8
fill_drop = signals.get("fill_rate", {}).get("z", 0) < -1
quality_ok = (
    signals.get("ctr", {}).get("z", 0) > -0.8 and
    signals.get("viewability", {}).get("z", 0) > -0.8
)

if revenue_drop and ecpm_drop and traffic_stable:
    cause = "🔴 Monetization Pressure"
    explanation = (
        "Revenue dropped mainly because advertiser pricing (eCPM) declined "
        "abnormally while traffic volume stayed stable. This points to demand-side "
        "pressure, floor price changes, or auction competition loss."
    )
    confidence = 0.90

elif revenue_drop and signals["sessions"]["z"] < -1:
    cause = "🔴 Traffic Loss"
    explanation = (
        "Revenue declined due to a significant drop in traffic volume. "
        "Monetization efficiency remains normal."
    )
    confidence = 0.85

elif revenue_drop and fill_drop:
    cause = "🔴 Delivery / Fill Issue"
    explanation = (
        "Revenue dropped due to reduced fill rate, indicating ads not being served "
        "consistently. Possible causes include blocking, latency, or demand gaps."
    )
    confidence = 0.80

elif revenue_drop:
    cause = "🟡 Mixed Signals"
    explanation = (
        "Revenue declined, but no single dominant factor was detected. "
        "Multiple minor movements likely combined."
    )
    confidence = 0.65

else:
    cause = "🟢 Normal Market Movement"
    explanation = (
        "Revenue changes fall within normal historical variation. "
        "No structural issues detected."
    )
    confidence = 0.60

# =============================
# EXECUTIVE KPI ROW
# =============================
st.subheader("📌 Executive Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Revenue",
    f"${signals['revenue']['today']:.2f}",
    f"YoD {signals['revenue']['vs_yesterday']:.1f}% | Base {signals['revenue']['vs_baseline']:.1f}%"
)

c2.metric(
    "eCPM",
    f"${signals['ecpm']['today']:.2f}",
    f"YoD {signals['ecpm']['vs_yesterday']:.1f}% | Base {signals['ecpm']['vs_baseline']:.1f}%"
)

c3.metric(
    "Sessions",
    int(signals['sessions']['today']),
    f"YoD {signals['sessions']['vs_yesterday']:.1f}%"
)

c4.metric("Confidence", f"{int(confidence*100)}%")

st.markdown(f"## **Primary Cause: {cause}**")
st.info(explanation)

# =============================
# STORY CHARTS (EASY)
# =============================
st.subheader("📉 What Changed (Focused Comparison)")

compare_df = pd.DataFrame({
    "Period": ["Baseline Avg", "Yesterday", "Today"],
    "Revenue": [
        baseline["revenue"]["mean"],
        yesterday_df["revenue"].values[0],
        today_df["revenue"].values[0]
    ],
    "eCPM": [
        baseline["ecpm"]["mean"],
        yesterday_df["ecpm"].values[0],
        today_df["ecpm"].values[0]
    ]
})

rev_chart = alt.Chart(compare_df).mark_bar().encode(
    x="Period",
    y="Revenue",
    color="Period"
).properties(title="Revenue Comparison")

ecpm_chart = alt.Chart(compare_df).mark_bar().encode(
    x="Period",
    y="eCPM",
    color="Period"
).properties(title="eCPM Comparison")

st.altair_chart(rev_chart, use_container_width=True)
st.altair_chart(ecpm_chart, use_container_width=True)

# =============================
# ACTION CHECKLIST
# =============================
st.subheader("🛠 Recommended Actions")

if "Monetization" in cause:
    st.markdown("""
    - Check bidder competition & bid density  
    - Review floor price or pricing rules  
    - Compare GEO & device demand shifts  
    - Check GAM auction pressure  
    """)

elif "Traffic" in cause:
    st.markdown("""
    - Analyze traffic source drops  
    - SEO / Discover / Social changes  
    - GA4 tagging & consent verification  
    """)

elif "Delivery" in cause:
    st.markdown("""
    - Inspect fill rate by ad unit  
    - Check blocked ads / ads.txt  
    - Page latency & rendering issues  
    """)

else:
    st.markdown("""
    - No urgent action required  
    - Monitor next 3–5 days  
    - Compare with peer sites  
    """)

st.caption("Revenue Intelligence • Dual-comparison • Root-cause driven • Exec-ready")
