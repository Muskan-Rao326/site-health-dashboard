import streamlit as st
import pandas as pd
import numpy as np

# ================= CONFIG =================
st.set_page_config(
    page_title="Site Health Monitor",
    layout="wide"
)

st.title("📊 AdTech Site Health Dashboard")

REVENUE_DIVISOR = 1_000_000   # micros → currency
BASELINE_DAYS = 7

# ================= LOAD DATA =================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["date"] = pd.to_datetime(df["date"])

    # Fix revenue units
    if df["revenue"].median() > 10_000:
        df["revenue"] = df["revenue"] / REVENUE_DIVISOR

    # Recalculate eCPM (source of truth)
    df["ecpm"] = (df["revenue"] / df["impressions"]) * 1000

    return df.sort_values("date")


uploaded_file = st.file_uploader("📤 Upload Site Health CSV", type="csv")
if uploaded_file is None:
    st.warning("Please upload a CSV file")
    st.stop()

df = load_data(uploaded_file)

# ================= DATE =================
selected_date = st.date_input("📅 Select Date", df["date"].max().date())
day_df = df[df["date"] == pd.to_datetime(selected_date)]
history_df = df[df["date"] < pd.to_datetime(selected_date)]

if day_df.empty:
    st.error("Selected date not found in data")
    st.stop()

# ================= HYBRID STRATEGY =================
st.divider()
st.subheader("🧠 Monitoring Mode")

if len(history_df) < BASELINE_DAYS:
    mode = "learning"
    st.info("⚪ Learning Mode — insufficient historical data")
else:
    mode = "rolling"
    baseline_df = history_df.tail(BASELINE_DAYS)
    st.success("🟢 Rolling Baseline Mode (Last 7 Days)")

# ================= CORE METRICS =================
today_rev = day_df["revenue"].sum()
today_imps = day_df["impressions"].sum()
today_req = day_df["adrequests"].sum()

today_ecpm = (today_rev / today_imps) * 1000 if today_imps > 0 else 0
fill_today = today_imps / today_req if today_req > 0 else 0

if mode == "rolling":
    base_rev = baseline_df["revenue"].mean()
    base_imps = baseline_df["impressions"].mean()
    base_req = baseline_df["adrequests"].mean()

    base_ecpm = (baseline_df["revenue"].sum() / baseline_df["impressions"].sum()) * 1000
    fill_base = base_imps / base_req if base_req > 0 else 0

    rev_change = (today_rev - base_rev) / base_rev * 100
    imps_change = (today_imps - base_imps) / base_imps * 100
    ecpm_change = (today_ecpm - base_ecpm) / base_ecpm * 100
    fill_change = (fill_today - fill_base) / fill_base * 100
else:
    rev_change = imps_change = ecpm_change = fill_change = 0

# ================= HEALTH SCORE =================
if mode == "rolling":
    health_score = (
        (100 + rev_change) * 0.50 +
        (100 + ecpm_change) * 0.25 +
        (100 + fill_change) * 0.15 +
        (100 + imps_change) * 0.10
    )
    health_score = max(0, min(100, health_score))
else:
    health_score = None

# ================= STATUS =================
st.divider()
st.subheader("🚦 Site Health")

if mode == "learning":
    st.info("⚪ LEARNING MODE — building baseline")
else:
    st.progress(health_score / 100)

    if health_score >= 80:
        st.success(f"🟢 HEALTHY — {health_score:.1f}/100")
    elif health_score >= 60:
        st.warning(f"🟡 NEEDS ATTENTION — {health_score:.1f}/100")
    else:
        st.error(f"🔴 CRITICAL — {health_score:.1f}/100")

# ================= KPI CARDS =================
st.divider()
st.subheader("📌 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"{today_rev:,.2f}", f"{rev_change:.1f}%")
c2.metric("eCPM", f"{today_ecpm:,.2f}", f"{ecpm_change:.1f}%")
c3.metric("Fill Rate", f"{fill_today:.2%}", f"{fill_change:.1f}%")
c4.metric("Impressions", f"{today_imps:,.0f}", f"{imps_change:.1f}%")

# ================= REVENUE DECOMPOSITION =================
if mode == "rolling":
    st.divider()
    st.subheader("🧩 Revenue Change Decomposition")

    imps_contribution = imps_change
    ecpm_contribution = ecpm_change

    decomp_df = pd.DataFrame({
        "Driver": ["Traffic", "eCPM"],
        "Impact (%)": [imps_contribution, ecpm_contribution]
    }).set_index("Driver")

    st.bar_chart(decomp_df)

# ================= ROOT CAUSE =================
if mode == "rolling":
    st.divider()
    st.subheader("🔥 Root Cause Impact")

    root_df = pd.DataFrame({
        "Metric": ["eCPM", "Fill Rate", "Traffic"],
        "Impact Strength": [
            abs(ecpm_change) * 0.5,
            abs(fill_change) * 0.3,
            abs(imps_change) * 0.2
        ]
    }).set_index("Metric")

    st.bar_chart(root_df)

    primary_issue = root_df["Impact Strength"].idxmax()
    st.warning(f"🎯 Primary revenue impact driven by **{primary_issue}**")

# ================= ANOMALY DETECTION =================
st.divider()
st.subheader("🚨 Risk & Anomaly Detection")

risk = False

if ecpm_change < -25:
    st.error("🚩 Sharp eCPM drop — demand or pricing issue")
    risk = True

if fill_change < -20:
    st.error("🚩 Fill rate collapse — monetization or setup issue")
    risk = True

if imps_change > 20 and rev_change < -20:
    st.error("🚩 Traffic spike but revenue drop — possible low-quality traffic")
    risk = True

if not risk:
    st.success("✅ No major anomalies detected")

# ================= TRENDS =================
st.divider()
st.subheader("📈 Revenue Trend (Last 14 Days)")

trend_df = df.tail(14).set_index("date")[["revenue"]]
st.line_chart(trend_df)

# ================= EXECUTIVE SUMMARY =================
st.divider()
st.subheader("🧠 Executive Summary")

if mode == "rolling":
    st.markdown(f"""
**Date:** {selected_date}

• Revenue changed **{rev_change:.1f}%**  
• Primary driver: **{primary_issue}**  
• Health score: **{health_score:.1f}/100**

**Recommended Actions**
- Investigate **{primary_issue}**
- Validate demand & pricing
- Monitor next 24 hours
""")
else:
    st.markdown("""
Baseline is still building.

Once 7 days of data are available:
- Health scoring
- Root-cause detection
- Anomaly alerts will activate
""")
