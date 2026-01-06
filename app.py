import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Dashboard")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("📂 Upload daily metrics CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =========================
# DATE DEFINITIONS
# =========================
latest_day = df["date"].max()
yesterday = latest_day - timedelta(days=1)

today_df = df[df["date"] == latest_day]
yesterday_df = df[df["date"] == yesterday]

if today_df.empty or yesterday_df.empty:
    st.error("Not enough data for Today vs Yesterday comparison.")
    st.stop()

# =========================
# BASELINE RANGE (USER SELECTABLE)
# =========================
st.sidebar.subheader("📅 Baseline Selection")

min_date = df["date"].min()
max_date = df["date"].max() - timedelta(days=2)

baseline_range = st.sidebar.date_input(
    "Select baseline date range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

baseline_df = df[
    (df["date"] >= pd.to_datetime(baseline_range[0])) &
    (df["date"] <= pd.to_datetime(baseline_range[1]))
]

# =========================
# METRICS WE CARE ABOUT
# =========================
METRICS = [
    "revenue", "ecpm", "impressions", "ad_requests",
    "ctr", "viewability"
]

# =========================
# KPI CALCULATIONS
# =========================
def get_today(metric):
    return today_df[metric].values[0]

def get_yesterday(metric):
    return yesterday_df[metric].values[0]

def get_delta(metric):
    return get_today(metric) - get_yesterday(metric)

def get_pct_change(metric):
    y = get_yesterday(metric)
    if y == 0:
        return 0
    return (get_today(metric) - y) / y * 100

def baseline_mean(metric):
    return baseline_df[metric].mean()

# =========================
# EXECUTIVE KPI ROW
# =========================
st.subheader("📌 Executive Snapshot (Today vs Yesterday)")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Revenue",
    f"${get_today('revenue'):,.2f}",
    f"{get_delta('revenue'):+,.2f} ({get_pct_change('revenue'):+.1f}%)"
)

c2.metric(
    "eCPM",
    f"${get_today('ecpm'):,.2f}",
    f"{get_delta('ecpm'):+.2f}"
)

c3.metric(
    "Impressions",
    f"{int(get_today('impressions')):,}",
    f"{int(get_delta('impressions')):+,}"
)

c4.metric(
    "Viewability",
    f"{get_today('viewability'):.2f}%",
    f"{get_delta('viewability'):+.2f}%"
)

st.caption(
    f"Baseline revenue avg: ${baseline_mean('revenue'):,.2f}"
)

# =========================
# ROOT CAUSE ANALYSIS
# =========================
traffic_drop = get_pct_change("impressions") < -15
ecpm_up = get_pct_change("ecpm") > 5
quality_stable = get_pct_change("viewability") > -5 and get_pct_change("ctr") > -5

if traffic_drop and ecpm_up and quality_stable:
    cause = "🔴 Traffic Collapse"
    explanation = (
        "Revenue declined primarily due to a sharp drop in traffic volume. "
        "eCPM increased, indicating advertisers paid more for fewer impressions. "
        "This is a volume-driven revenue loss, not a quality issue."
    )

elif get_pct_change("ecpm") < -10:
    cause = "🔴 Monetization Pressure"
    explanation = (
        "Revenue declined mainly due to a significant drop in eCPM. "
        "This suggests reduced advertiser demand, pricing pressure, or auction softening."
    )

elif get_pct_change("viewability") < -10:
    cause = "🔴 Delivery / Visibility Issue"
    explanation = (
        "Revenue decline is linked to reduced ad viewability, "
        "which directly impacts advertiser bidding and CPMs."
    )

else:
    cause = "🟡 Normal Market Movement"
    explanation = (
        "Daily changes fall within expected historical variation. "
        "No structural monetization or traffic issues detected."
    )

st.markdown(f"### **Primary Cause: {cause}**")
st.info(explanation)

# =========================
# STORYTELLING CHARTS
# =========================
st.subheader("📈 What Changed Over Time")

# Revenue vs Impressions
rev_imp = df.melt(
    id_vars="date",
    value_vars=["revenue", "impressions"],
    var_name="metric",
    value_name="value"
)

chart1 = alt.Chart(rev_imp).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="metric:N",
    tooltip=["date", "metric", "value"]
).properties(title="Revenue vs Traffic Volume")

st.altair_chart(chart1, use_container_width=True)

# Revenue vs eCPM
rev_ecpm = df.melt(
    id_vars="date",
    value_vars=["revenue", "ecpm"],
    var_name="metric",
    value_name="value"
)

chart2 = alt.Chart(rev_ecpm).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="metric:N",
    tooltip=["date", "metric", "value"]
).properties(title="Revenue vs eCPM (Monetization Decoupling)")

st.altair_chart(chart2, use_container_width=True)

# Quality metrics
quality = df.melt(
    id_vars="date",
    value_vars=["ctr", "viewability"],
    var_name="metric",
    value_name="value"
)

chart3 = alt.Chart(quality).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="metric:N",
    tooltip=["date", "metric", "value"]
).properties(title="Traffic Quality Health")

st.altair_chart(chart3, use_container_width=True)

# =========================
# ACTION CHECKLIST
# =========================
st.subheader("🛠 Recommended Actions")

if "Traffic" in cause:
    st.markdown("""
    - Audit traffic source losses (SEO, Discover, Social)
    - Check seasonality & holidays
    - Compare sessions vs impressions
    - Validate GA4 & ad stack tracking
    """)

elif "Monetization" in cause:
    st.markdown("""
    - Review floor prices
    - Analyze advertiser bid density
    - Check GEO & device demand shifts
    """)

elif "Visibility" in cause:
    st.markdown("""
    - Review ad placements
    - Check CLS / layout shifts
    - Audit lazy-loading & viewport logic
    """)

else:
    st.markdown("""
    - No immediate action required
    - Continue monitoring trends
    """)

st.caption("Revenue Intelligence Dashboard • Today vs Yesterday vs Baseline • Explainable & Safe")
