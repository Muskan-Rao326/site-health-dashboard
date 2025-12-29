import streamlit as st
import pandas as pd
import numpy as np

# ================= CONFIG =================
st.set_page_config(page_title="Site Health Monitor", layout="wide")
st.title("📊 AdTech Site Health Dashboard")

REVENUE_DIVISOR = 1_000_000   # micros → real currency
BASELINE_DAYS = 7

# ================= LOAD DATA =================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    # Date
    df["date"] = pd.to_datetime(df["date"])

    # -------- FIX UNITS --------
    if df["revenue"].median() > 10_000:
        df["revenue"] = df["revenue"] / REVENUE_DIVISOR

    # Recalculate eCPM (never trust API value)
    df["ecpm"] = (df["revenue"] / df["impressions"]) * 1000

    return df.sort_values("date")


uploaded_file = st.file_uploader("📤 Upload Site Health CSV", type="csv")
if uploaded_file is None:
    st.warning("Please upload the CSV file")
    st.stop()

df = load_data(uploaded_file)

# ================= DATE =================
selected_date = st.date_input("📅 Select Date", df["date"].max().date())
day_df = df[df["date"] == pd.to_datetime(selected_date)]
history_df = df[df["date"] < pd.to_datetime(selected_date)]

if day_df.empty:
    st.error("Selected date not found")
    st.stop()

# ================= HYBRID STRATEGY =================
st.divider()
st.subheader("🧠 Monitoring Mode")

if len(history_df) < BASELINE_DAYS:
    st.info("⚪ Learning Mode: Not enough history (minimum 7 days required)")
    mode = "learning"
else:
    baseline_df = history_df.tail(BASELINE_DAYS)
    mode = "rolling"

st.write(f"**Current Mode:** `{mode.upper()}`")

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
if mode == "learning":
    status = "⚪ LEARNING MODE"
elif health_score >= 80:
    status = "🟢 HEALTHY"
elif health_score >= 60:
    status = "🟡 NEEDS ATTENTION"
else:
    status = "🔴 CRITICAL"

st.markdown("## 🚦 Site Health Status")
if "🟢" in status:
    st.success(f"{status} | Score: {health_score:.1f}/100")
elif "🟡" in status:
    st.warning(f"{status} | Score: {health_score:.1f}/100")
elif "🔴" in status:
    st.error(f"{status} 🚨 | Score: {health_score:.1f}/100")
else:
    st.info(status)

# ================= KPI =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Revenue", f"{today_rev:,.2f}", f"{rev_change:.1f}%")
col2.metric("eCPM", f"{today_ecpm:.2f}", f"{ecpm_change:.1f}%")
col3.metric("Fill Rate", f"{fill_today:.2%}", f"{fill_change:.1f}%")
col4.metric("Impressions", f"{today_imps:,.0f}", f"{imps_change:.1f}%")

# ================= ROOT CAUSE =================
if mode == "rolling":
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

    st.dataframe(root_df)
    primary_issue = root_df.index[0]
    st.warning(f"Primary revenue impact driven by **{primary_issue}**")

# ================= TREND =================
st.divider()
st.subheader("📈 Revenue Trend (Last 14 Days)")
st.line_chart(df.set_index("date")["revenue"].tail(14))
