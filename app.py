import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ad Traffic Fraud Intelligence",
    layout="wide"
)

st.title("🛡️ Ad Traffic Fraud Intelligence Dashboard")

# ---------------- LOAD LOCAL DATA ----------------
@st.cache_data
def load_data():
    file_path = "data/combined_data.csv"

    if not os.path.exists(file_path):
        st.error("❌ No local data file found and no file uploaded.")
        st.stop()

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------- SIDEBAR: FILE UPLOADER ----------------
st.sidebar.subheader("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload combined_data.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])
else:
    df = load_data()


# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

site = st.sidebar.selectbox(
    "Select Site",
    ["All"] + sorted(df["site_name"].unique().tolist())
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

# ---------------- KPI SECTION ----------------
st.subheader("📌 Key Fraud KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Fraud Score", round(df["fraud_score"].mean(), 1))
col2.metric("High Risk Days", (df["fraud_label"] == "High Risk").sum())
col3.metric("Fraud Likely Days", (df["fraud_label"] == "Fraud Likely").sum())
col4.metric("Avg RPM", f"${df['rpm'].mean():.2f}")

# ---------------- FRAUD SCORE TREND ----------------
st.subheader("📈 Fraud Score Trend")

fig_trend = px.line(
    df,
    x="date",
    y="fraud_score",
    color="site_name",
    markers=True,
    title="Fraud Score Over Time"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- FRAUD DISTRIBUTION ----------------
st.subheader("🚨 Fraud Classification Distribution")

fig_dist = px.histogram(
    df,
    x="fraud_label",
    color="fraud_label",
    title="Fraud Risk Distribution"
)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------- SIGNAL BREAKDOWN ----------------
st.subheader("🧠 Fraud Signal Breakdown")

signal_cols = [
    "traffic_quality_issue",
    "monetization_anomaly",
    "ivt_lite",
    "revenue_manipulation"
]

signal_summary = df[signal_cols].sum().reset_index()
signal_summary.columns = ["Signal", "Count"]

fig_signals = px.bar(
    signal_summary,
    x="Signal",
    y="Count",
    title="Fraud Signals Triggered"
)
st.plotly_chart(fig_signals, use_container_width=True)

# ---------------- METRIC CORRELATION ----------------
st.subheader("🔗 Key Metric Relationships")

fig_scatter = px.scatter(
    df,
    x="engagement_rate",
    y="ctr",
    size="sessions",
    color="fraud_label",
    title="CTR vs Engagement (Bubble = Sessions)"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- DETAILED TABLE ----------------
st.subheader("📋 Detailed Fraud Table")

table_cols = [
    "date",
    "site_name",
    "fraud_score",
    "fraud_label",
    "sessions",
    "engagement_rate",
    "ctr",
    "ecpm",
    "rpm",
    "traffic_quality_issue",
    "monetization_anomaly",
    "ivt_lite",
    "revenue_manipulation"
]

st.dataframe(
    df[table_cols].sort_values("fraud_score", ascending=False),
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.caption("⚠️ Fraud scores are probabilistic signals, not billing decisions.")
