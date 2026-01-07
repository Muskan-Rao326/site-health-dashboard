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

st.title("📊 Revenue Intelligence & Anomaly Diagnosis")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload combined CSV",
    type=["csv"]
)

if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.subheader("📅 Analysis Controls")

analysis_date = st.sidebar.selectbox(
    "Select analysis date",
    sorted(df["date"].unique(), reverse=True)
)

baseline_days = st.sidebar.slider(
    "Baseline window (days)",
    min_value=3,
    max_value=30,
    value=7
)

# =========================
# DATA SLICES
# =========================
today_df = df[df["date"] == analysis_date]
yesterday_df = df[df["date"] == (analysis_date - timedelta(days=1))]

baseline_df = df[
    (df["date"] < analysis_date)
].tail(baseline_days)

if today_df.empty or baseline_df.empty:
    st.warning("Not enough data for selected date.")
    st.stop()

# =========================
# METRICS TO ANALYZE
# =========================
METRICS = [
    "revenue",
    "ecpm",
    "sessions",
    "ctr",
    "viewability",
    "fill_rate"
]

# =========================
# CALCULATIONS
# =========================
rows = {}

for m in METRICS:
    today = today_df[m].values[0]
    yesterday = (
        yesterday_df[m].values[0]
        if not yesterday_df.empty else np.nan
    )

    base_mean = baseline_df[m].mean()
    base_std = baseline_df[m].std() or 1

    rows[m] = {
        "today": today,
        "yesterday": yesterday,
        "pct_vs_yesterday": (
            (today - yesterday) / yesterday * 100
            if yesterday and yesterday != 0 else 0
        ),
        "z_score": (today - base_mean) / base_std
    }

# =========================
# EXECUTIVE KPI ROW
# =========================
st.subheader(f"📌 Snapshot for {analysis_date.date()}")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Revenue",
    f"${rows['revenue']['today']:.2f}",
    f"{rows['revenue']['pct_vs_yesterday']:.1f}% vs yesterday"
)

c2.metric(
    "eCPM",
    f"${rows['ecpm']['today']:.2f}",
    f"{rows['ecpm']['pct_vs_yesterday']:.1f}%"
)

c3.metric(
    "Sessions",
    int(rows['sessions']['today']),
    f"{rows['sessions']['pct_vs_yesterday']:.1f}%"
)

c4.metric(
    "CTR",
    f"{rows['ctr']['today']:.2%}",
    f"z = {rows['ctr']['z_score']:.2f}"
)

# =========================
# ROOT CAUSE LOGIC
# =========================
if rows["ecpm"]["z_score"] < -1.5 and abs(rows["sessions"]["z_score"]) < 1:
    cause = "🔴 Monetization Pressure"
    reason = (
        "Revenue dropped mainly due to lower advertiser pricing (eCPM), "
        "while traffic volume remained stable. This typically indicates "
        "market demand softening, floor issues, or auction pressure."
    )

elif rows["sessions"]["z_score"] < -1.5:
    cause = "🔴 Traffic Loss"
    reason = (
        "Revenue declined primarily due to reduced traffic volume. "
        "Monetization efficiency remains normal."
    )

elif rows["fill_rate"]["z_score"] < -1.5:
    cause = "🔴 Delivery / Demand Issue"
    reason = (
        "Lower fill rate indicates reduced advertiser demand, blocking, "
        "or delivery constraints."
    )

else:
    cause = "🟡 Normal Market Movement"
    reason = (
        "All changes fall within historical variation. "
        "No structural issues detected."
    )

st.markdown(f"### **Primary Cause: {cause}**")
st.info(reason)

# =========================
# STORY CHARTS
# =========================
st.subheader("📈 Behavior Over Time")

chart_metrics = ["revenue", "ecpm", "sessions"]

for m in chart_metrics:
    chart = alt.Chart(df).mark_line().encode(
        x="date:T",
        y=f"{m}:Q",
        tooltip=["date", m]
    ).properties(
        title=m.upper()
    )

    st.altair_chart(chart, use_container_width=True)

# =========================
# QUALITY HEALTH
# =========================
st.subheader("🧠 Traffic Quality Signals")

quality_df = df.melt(
    id_vars="date",
    value_vars=["ctr", "viewability", "fill_rate"]
)

quality_chart = alt.Chart(quality_df).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="variable:N",
    tooltip=["date", "variable", "value"]
)

st.altair_chart(quality_chart, use_container_width=True)

# =========================
# DETAILED TABLE
# =========================
st.subheader("📋 Metric Diagnostics")

table = []
for m, v in rows.items():
    table.append({
        "Metric": m,
        "Today": v["today"],
        "Vs Yesterday %": round(v["pct_vs_yesterday"], 2),
        "Z Score (Baseline)": round(v["z_score"], 2)
    })

st.dataframe(pd.DataFrame(table), use_container_width=True)

st.caption(
    "Baseline = previous N days | Z-score used only for anomaly detection"
)
