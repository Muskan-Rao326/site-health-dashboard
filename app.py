import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Ad Traffic Fraud Intelligence",
    layout="wide"
)

st.title("🛡️ Ad Traffic Fraud Intelligence Dashboard")

# ================= REQUIRED SCHEMA =================
REQUIRED_SCHEMA = {
    "date": "datetime",
    "site_name": "string",
    "sessions": "numeric",
    "users": "numeric",
    "engagement_rate": "numeric",
    "pageviews": "numeric",
    "ad_requests": "numeric",
    "impressions": "numeric",
    "clicks": "numeric",
    "ctr": "numeric",
    "ecpm": "numeric",
    "revenue": "numeric",
    "viewability": "numeric",
}

# ================= HELPERS =================
def safe_divide(a, b):
    return np.where(b == 0, 0, a / b)

def zscore(series):
    std = series.std()
    return np.zeros(len(series)) if std == 0 else (series - series.mean()) / std

def validate_schema(df):
    missing = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col, dtype in REQUIRED_SCHEMA.items():
        if dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# ================= LOAD DATA =================
@st.cache_data
def load_local_data():
    path = "data/combined_data.csv"
    if not os.path.exists(path):
        st.error("❌ No data found. Upload CSV.")
        st.stop()
    return pd.read_csv(path)

# ================= SIDEBAR =================
st.sidebar.subheader("📂 Upload Data")

uploaded = st.sidebar.file_uploader("Upload combined_data.csv", type=["csv"])

df = pd.read_csv(uploaded) if uploaded else load_local_data()
df = validate_schema(df)

# ================= DERIVED METRICS =================
df["rpm"] = safe_divide(df["revenue"], df["sessions"]) * 1000
df["fill_rate"] = safe_divide(df["impressions"], df["ad_requests"])
df["impressions_per_session"] = safe_divide(df["impressions"], df["sessions"])
df["clicks_per_session"] = safe_divide(df["clicks"], df["sessions"])
df["pageviews_per_session"] = safe_divide(df["pageviews"], df["sessions"])

# ================= Z-SCORES =================
for col in [
    "engagement_rate",
    "ctr",
    "ecpm",
    "viewability",
    "rpm",
    "clicks_per_session",
]:
    df[f"{col}_z"] = zscore(df[col])

# ================= FRAUD SIGNALS =================
df["traffic_quality_issue"] = (
    (df["engagement_rate_z"] < -2) |
    (df["pageviews_per_session"] < 1.2)
)

df["monetization_anomaly"] = (
    (df["ctr_z"] > 2) &
    (df["ecpm_z"] < -2)
)

df["ivt_lite"] = (
    (df["clicks_per_session_z"] > 2) |
    ((df["viewability_z"] > 2) & (df["engagement_rate_z"] < -1))
)

df["revenue_manipulation"] = df["rpm_z"] < -2

df["fraud_score"] = (
    df["traffic_quality_issue"] * 25 +
    df["monetization_anomaly"] * 25 +
    df["ivt_lite"] * 25 +
    df["revenue_manipulation"] * 25
)

df["fraud_label"] = pd.cut(
    df["fraud_score"],
    bins=[-1, 25, 50, 75, 100],
    labels=["Clean", "Monitor", "High Risk", "Fraud Likely"]
)

# ================= FILTERS =================
st.sidebar.header("🔍 Filters")

site = st.sidebar.selectbox("Site", ["All"] + sorted(df["site_name"].unique()))
date_range = st.sidebar.date_input("Date Range", [df["date"].min(), df["date"].max()])

if site != "All":
    df = df[df["site_name"] == site]

df = df[(df["date"] >= pd.to_datetime(date_range[0])) &
        (df["date"] <= pd.to_datetime(date_range[1]))]

# ================= KPIs =================
st.subheader("📌 Key Risk KPIs")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Fraud Score", round(df["fraud_score"].mean(), 1))
c2.metric("Fraud Likely Days", (df["fraud_label"] == "Fraud Likely").sum())
c3.metric("High Risk Days", (df["fraud_label"] == "High Risk").sum())
c4.metric("Avg RPM", f"${df['rpm'].mean():.2f}")
c5.metric("Avg CTR", f"{df['ctr'].mean():.2f}%")

# ================= FRAUD SCORE BAND =================
st.subheader("🚨 Fraud Score With Risk Bands")

fig_band = go.Figure()

fig_band.add_trace(go.Scatter(
    x=df["date"],
    y=df["fraud_score"],
    mode="lines+markers",
    name="Fraud Score"
))

fig_band.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.15)
fig_band.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.15)
fig_band.add_hrect(y0=25, y1=50, fillcolor="yellow", opacity=0.15)

st.plotly_chart(fig_band, use_container_width=True)

# ================= RISK CALENDAR =================
st.subheader("📅 Risk Calendar Heatmap")

calendar_df = df.copy()
calendar_df["day"] = calendar_df["date"].dt.day
calendar_df["month"] = calendar_df["date"].dt.strftime("%b")

fig_cal = px.density_heatmap(
    calendar_df,
    x="day",
    y="month",
    z="fraud_score",
    color_continuous_scale="Reds",
    title="Fraud Intensity by Day"
)
st.plotly_chart(fig_cal, use_container_width=True)

# ================= FUNNEL =================
st.subheader("📉 Monetization Funnel")

funnel_df = pd.DataFrame({
    "Stage": ["Ad Requests", "Impressions", "Clicks"],
    "Count": [
        df["ad_requests"].sum(),
        df["impressions"].sum(),
        df["clicks"].sum()
    ]
})

fig_funnel = px.funnel(
    funnel_df,
    x="Count",
    y="Stage"
)
st.plotly_chart(fig_funnel, use_container_width=True)

# ================= SIGNAL TIMELINE =================
st.subheader("🧠 Fraud Signal Timeline")

signal_cols = [
    "traffic_quality_issue",
    "monetization_anomaly",
    "ivt_lite",
    "revenue_manipulation"
]

signal_ts = df.groupby("date")[signal_cols].sum().reset_index()

fig_signal = px.area(
    signal_ts,
    x="date",
    y=signal_cols,
    title="Fraud Signals Over Time"
)
st.plotly_chart(fig_signal, use_container_width=True)

# ================= SCATTER =================
st.subheader("🔗 CTR vs Engagement (Bubble = Sessions)")

fig_scatter = px.scatter(
    df,
    x="engagement_rate",
    y="ctr",
    size="sessions",
    color="fraud_label",
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ================= TABLE =================
st.subheader("📋 Detailed Risk Table")

st.dataframe(
    df.sort_values("fraud_score", ascending=False),
    use_container_width=True
)

st.caption("⚠️ Visualization highlights risk signals, not billing decisions.")
