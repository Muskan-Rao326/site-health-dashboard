import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AdOps Intelligence Dashboard",
    layout="wide"
)

st.title("📊 AdOps Intelligence & Anomaly Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.subheader("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload combined_data.csv",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Please upload combined_data.csv")
    st.stop()

# ---------------- FILTERS ----------------
st.sidebar.subheader("🔍 Filters")

site = st.sidebar.selectbox(
    "Site",
    ["All"] + sorted(df["site_name"].unique())
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["date"].min(), df["date"].max()]
)

if site != "All":
    df = df[df["site_name"] == site]

df = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

df = df.sort_values("date")

# ---------------- KPI ROW ----------------
st.subheader("📌 Executive Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg eCPM", f"${df['ecpm'].mean():.2f}")
col2.metric("Avg RPM", f"${df['rpm'].mean():.2f}")
col3.metric("Avg CTR", f"{df['ctr'].mean()*100:.2f}%")
col4.metric("Avg Engagement", f"{df['engagement_rate'].mean()*100:.1f}%")

# ---------------- WHAT CHANGED TODAY ----------------
st.subheader("🧠 What Changed Today & Why")

latest = df.iloc[-1]
baseline = df.tail(7).mean(numeric_only=True)

def pct_change(today, base):
    if base == 0:
        return 0
    return ((today - base) / base) * 100

def classify(p):
    if p <= -10:
        return "🔴 Sharp Drop"
    elif p <= -3:
        return "🟠 Mild Drop"
    elif p < 3:
        return "⚪ Stable"
    elif p < 10:
        return "🟡 Mild Increase"
    else:
        return "🟢 Strong Increase"

def reason_engine(metric, change, row):
    if metric == "ecpm" and change < -5:
        if row["ctr"] < baseline["ctr"]:
            return "Lower advertiser interest (CTR drop)"
        if row["impressions"] > baseline["impressions"]:
            return "Inventory oversupply"
        if row["fill_rate"] > baseline["fill_rate"]:
            return "Cheap demand dilution"
        return "Buyer bid pressure reduced"

    if metric == "rpm" and change < -5:
        return "Revenue not scaling with traffic"

    if metric == "ctr" and change > 10 and row["ecpm"] < baseline["ecpm"]:
        return "Possible accidental or low-quality clicks"

    if metric == "engagement_rate" and change < -5:
        return "Low-intent traffic increase"

    if metric == "viewability" and change < -5:
        return "Ad placement or layout issue"

    return "Normal market movement"

metrics = [
    "sessions", "pageviews", "engagement_rate",
    "ctr", "fill_rate", "ecpm", "rpm", "viewability"
]

for m in metrics:
    today = latest[m]
    base = baseline[m]
    change = pct_change(today, base)
    status = classify(change)
    reason = reason_engine(m, change, latest)

    if "🔴" in status:
        st.error(f"{status} | {m.upper()} {change:.1f}% — {reason}")
    elif "🟠" in status:
        st.warning(f"{status} | {m.upper()} {change:.1f}% — {reason}")
    elif "🟢" in status:
        st.success(f"{status} | {m.upper()} +{change:.1f}%")
    else:
        st.info(f"{status} | {m.upper()} {change:.1f}%")

# ---------------- VISUAL SECTION ----------------
st.subheader("📈 Metric Trends")

metric_choice = st.selectbox(
    "Select Metric",
    ["ecpm", "rpm", "ctr", "fill_rate", "engagement_rate", "viewability"]
)

fig = px.line(
    df,
    x="date",
    y=metric_choice,
    color="site_name",
    markers=True,
    title=f"{metric_choice.upper()} Trend"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- CORRELATION VIEW ----------------
st.subheader("🔗 Monetization vs Traffic Quality")

fig2 = px.scatter(
    df,
    x="engagement_rate",
    y="ecpm",
    size="sessions",
    color="ctr",
    title="eCPM vs Engagement (Bubble = Sessions)"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("📋 Detailed Metrics Table")

st.dataframe(
    df.sort_values("date", ascending=False),
    use_container_width=True
)

st.caption("⚠️ Insights are diagnostic signals, not billing decisions.")
