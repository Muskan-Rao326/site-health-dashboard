import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Revenue Intelligence Dashboard")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.stop()

df = load_data(uploaded)

# =============================
# DATE CONTROLS
# =============================
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=df["date"].max().date(),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date()
    )

with col2:
    baseline_days = st.slider(
        "Baseline Window (days)",
        min_value=7,
        max_value=30,
        value=7
    )

today = pd.to_datetime(selected_date)
yesterday = today - timedelta(days=1)
baseline_start = today - timedelta(days=baseline_days)

# =============================
# DATA SCOPES (CRITICAL)
# =============================
today_df = df[df["date"] == today]
yesterday_df = df[df["date"] == yesterday]

baseline_df = df[
    (df["date"] < today) &
    (df["date"] >= baseline_start)
]

display_df = df[
    (df["date"] >= baseline_start - timedelta(days=1)) &
    (df["date"] <= today)
]

# =============================
# SAFE VALUE GETTER
# =============================
def get_value(dframe, col):
    return float(dframe[col].sum()) if not dframe.empty else 0.0

# =============================
# KPI METRICS
# =============================
metrics = [
    "revenue", "ecpm", "sessions",
    "impressions", "ctr", "viewability", "engagement_rate"
]

kpi = {}
for m in metrics:
    kpi[m] = {
        "today": get_value(today_df, m),
        "yesterday": get_value(yesterday_df, m),
        "baseline_mean": baseline_df[m].mean(),
        "baseline_std": baseline_df[m].std()
    }

# =============================
# Z-SCORE & CONFIDENCE
# =============================
def z_score(val, mean, std):
    if std == 0 or np.isnan(std):
        return 0
    return (val - mean) / std

def confidence_from_z(z):
    return norm.cdf(abs(z)) * 100

for m in metrics:
    kpi[m]["z"] = z_score(
        kpi[m]["today"],
        kpi[m]["baseline_mean"],
        kpi[m]["baseline_std"]
    )
    kpi[m]["confidence"] = confidence_from_z(kpi[m]["z"])

# =============================
# KPI DISPLAY
# =============================
st.subheader("📌 Key Metrics")

cols = st.columns(4)

def render_kpi(col, title, val, delta, z, conf):
    color = "#2ECC71" if z >= -1 else "#F1C40F" if z >= -2 else "#E74C3C"
    col.markdown(
        f"""
        <div style="background:{color};padding:15px;border-radius:10px;color:white">
            <h4>{title}</h4>
            <h2>{val:,.2f}</h2>
            <p>Δ vs Yesterday: {delta:+.2f}%</p>
            <p>Z-score: {z:.2f}</p>
            <p>Confidence: {conf:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_kpi(
    cols[0],
    "Revenue",
    kpi["revenue"]["today"],
    ((kpi["revenue"]["today"] - kpi["revenue"]["yesterday"]) /
     max(kpi["revenue"]["yesterday"], 1)) * 100,
    kpi["revenue"]["z"],
    kpi["revenue"]["confidence"]
)

render_kpi(
    cols[1],
    "eCPM",
    kpi["ecpm"]["today"],
    ((kpi["ecpm"]["today"] - kpi["ecpm"]["yesterday"]) /
     max(kpi["ecpm"]["yesterday"], 1)) * 100,
    kpi["ecpm"]["z"],
    kpi["ecpm"]["confidence"]
)

render_kpi(
    cols[2],
    "Sessions",
    kpi["sessions"]["today"],
    ((kpi["sessions"]["today"] - kpi["sessions"]["yesterday"]) /
     max(kpi["sessions"]["yesterday"], 1)) * 100,
    kpi["sessions"]["z"],
    kpi["sessions"]["confidence"]
)

render_kpi(
    cols[3],
    "CTR",
    kpi["ctr"]["today"],
    ((kpi["ctr"]["today"] - kpi["ctr"]["yesterday"]) /
     max(kpi["ctr"]["yesterday"], 1)) * 100,
    kpi["ctr"]["z"],
    kpi["ctr"]["confidence"]
)

# =============================
# AUTO INSIGHT ENGINE
# =============================
st.subheader("🧠 Auto Insight")

if kpi["revenue"]["z"] < -2:
    if abs(kpi["sessions"]["z"]) < 1:
        st.error(
            "🔴 Revenue dropped while traffic is stable.\n\n"
            "**Primary cause:** Monetization pressure (eCPM / fill / delivery issue)."
        )
    else:
        st.error(
            "🔴 Revenue drop driven by traffic decline.\n\n"
            "**Primary cause:** Demand / acquisition issue."
        )
else:
    st.success("🟢 Revenue movement is within normal baseline behavior.")

# =============================
# DYNAMIC CHARTS (FIXED)
# =============================
st.subheader("📈 Trends (Baseline → Today)")

def line_chart(cols, title):
    melted = display_df.melt(
        id_vars="date",
        value_vars=cols
    )
    return alt.Chart(melted).mark_line(point=True).encode(
        x="date:T",
        y="value:Q",
        color="variable:N",
        tooltip=["date:T", "variable:N", "value:Q"]
    ).properties(title=title)

st.altair_chart(
    line_chart(["revenue", "ecpm"], "Revenue vs eCPM"),
    use_container_width=True
)

st.altair_chart(
    line_chart(["sessions", "revenue"], "Traffic vs Revenue"),
    use_container_width=True
)

st.altair_chart(
    line_chart(["ctr", "viewability", "engagement_rate"], "Traffic Quality Signals"),
    use_container_width=True
)

# =============================
# Z-SCORE GUIDE
# =============================
st.subheader("📘 Z-Score Reading Guide")

st.markdown("""
| Z-score | Meaning | Action |
|------|--------|--------|
| 0 to -1 | Normal | No action |
| -1 to -2 | Watch | Monitor |
| < -2 | Anomaly | Investigate immediately |
""")

# =============================
# FOOTER
# =============================
st.caption(
    "Baseline-driven, date-dynamic revenue intelligence dashboard "
    "with statistical confidence & alerting."
)
