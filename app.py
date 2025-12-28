import streamlit as st
import pandas as pd
import numpy as np

# ================= CONFIG =================
st.set_page_config(
    page_title="Site Health Monitor",
    layout="wide"
)

st.title("📊 AdTech Site Health Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Date parsing
    df["date"] = pd.to_datetime(df["date"])

    return df

uploaded_file = st.file_uploader(
    "📤 Upload Site Health CSV",
    type="csv"
)

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue")
    st.stop()

df = load_data(uploaded_file)

# ================= DATE SELECTION =================
selected_date = st.date_input(
    "📅 Select Date",
    df["date"].max().date()
)

day_df = df[df["date"] == pd.to_datetime(selected_date)]
baseline_df = df[df["date"] < pd.to_datetime(selected_date)].tail(7)

if day_df.empty or baseline_df.empty:
    st.error("Not enough historical data to calculate health")
    st.stop()

# ================= CORE METRICS =================
today_rev = day_df["revenue"].sum()
today_imps = day_df["impressions"].sum()
today_req = day_df["adrequests"].sum()

today_ecpm = (today_rev / today_imps) * 1000 if today_imps > 0 else 0
fill_today = today_imps / today_req if today_req > 0 else 0

base_rev = baseline_df["revenue"].mean()
base_imps = baseline_df["impressions"].mean()
base_req = baseline_df["adrequests"].mean()

base_ecpm = (
    baseline_df["revenue"].sum() /
    baseline_df["impressions"].sum()
) * 1000

fill_base = base_imps / base_req if base_req > 0 else 0

# ================= % CHANGES =================
rev_change = (today_rev - base_rev) / base_rev * 100
imps_change = (today_imps - base_imps) / base_imps * 100
ecpm_change = (today_ecpm - base_ecpm) / base_ecpm * 100
fill_change = (fill_today - fill_base) / fill_base * 100

# ================= HEALTH SCORE =================
health_score = (
    (100 + rev_change) * 0.50 +
    (100 + ecpm_change) * 0.25 +
    (100 + fill_change) * 0.15 +
    (100 + imps_change) * 0.10
)

health_score = max(0, min(100, health_score))

# ================= STATUS =================
if health_score >= 80:
    status = "🟢 HEALTHY"
elif health_score >= 60:
    status = "🟡 NEEDS ATTENTION"
else:
    status = "🔴 CRITICAL"

# ================= STATUS DISPLAY =================
st.markdown("## 🚦 Site Health Status")

if "🟢" in status:
    st.success(f"{status} | Score: {health_score:.1f}/100")
elif "🟡" in status:
    st.warning(f"{status} | Score: {health_score:.1f}/100")
else:
    st.error(f"{status} 🚨 | Score: {health_score:.1f}/100")

# ================= KPI ROW =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Revenue", f"{today_rev:,.0f}", f"{rev_change:.1f}%")
col2.metric("eCPM", f"{today_ecpm:,.0f}", f"{ecpm_change:.1f}%")
col3.metric("Fill Rate", f"{fill_today:.2%}", f"{fill_change:.1f}%")
col4.metric("Impressions", f"{today_imps:,.0f}", f"{imps_change:.1f}%")

# ================= ROOT CAUSE =================
st.divider()
st.subheader("🧩 Revenue Root Cause Ranking")

root_causes = {
    "eCPM": ecpm_change * 0.5,
    "Fill Rate": fill_change * 0.3,
    "Traffic": imps_change * 0.2
}

root_df = (
    pd.DataFrame.from_dict(root_causes, orient="index", columns=["Impact"])
    .sort_values("Impact")
)

st.dataframe(root_df.style.background_gradient(cmap="Reds"))

primary_issue = root_df.index[0]
st.warning(f"Primary Revenue Impact Driver: **{primary_issue}**")

# ================= ANOMALY DETECTION =================
st.divider()
st.subheader("🚨 Anomaly & Risk Detection")

anomaly = False

if imps_change > 20 and rev_change < -20:
    st.error("🚩 Traffic up but revenue down → Low quality or fraud traffic")
    anomaly = True

if ecpm_change < -30:
    st.error("🚩 Severe eCPM drop → Demand or pricing issue")
    anomaly = True

if fill_change < -25:
    st.error("🚩 Fill rate collapse → Monetization issue")
    anomaly = True

if not anomaly:
    st.success("✅ No major anomalies detected")

# ================= EXECUTIVE SUMMARY =================
st.divider()
st.header("📌 Executive Summary")

st.markdown(f"""
### {status}

**Date:** {selected_date}

- **Revenue:** {today_rev:,.0f}
- **Revenue Change:** {rev_change:.1f}%
- **eCPM Change:** {ecpm_change:.1f}%
- **Fill Rate Change:** {fill_change:.1f}%
- **Traffic Change:** {imps_change:.1f}%

### 🧠 Key Insight
Primary revenue impact driven by **{primary_issue}**

### 🎯 Recommended Action
- Investigate **{primary_issue}**
- Review demand, pricing & traffic quality
- Monitor next 24 hours closely
""")

# ================= TREND =================
st.divider()
st.subheader("📈 Revenue Trend (Last 14 Days)")

trend_df = df.sort_values("date").tail(14)

st.line_chart(
    trend_df.set_index("date")["revenue"]
)
