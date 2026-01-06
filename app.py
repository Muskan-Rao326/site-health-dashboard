import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Revenue Intelligence System",
    layout="wide"
)

st.title("📊 Revenue Intelligence & Root Cause Engine")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload daily metrics CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# =========================
# DATE RANGE
# =========================
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.date_input(
    "Baseline period (used to learn normal behavior)",
    [min_date, max_date - timedelta(days=1)],
    min_value=min_date,
    max_value=max_date
)

baseline_df = df[(df["date"] >= pd.to_datetime(date_range[0])) &
                 (df["date"] <= pd.to_datetime(date_range[1]))]

today = df["date"].max()
today_df = df[df["date"] == today]

# =========================
# SAFETY CHECK
# =========================
if today_df.empty or len(baseline_df) < 5:
    st.error("Not enough data for reliable analysis.")
    st.stop()

# =========================
# METRIC GROUPS
# =========================
TRAFFIC = ["sessions", "users", "pageviews"]
QUALITY = ["ctr", "viewability", "engagement_rate"]
DELIVERY = ["fill_rate"]
MONETIZATION = ["ecpm", "revenue"]

ALL_METRICS = TRAFFIC + QUALITY + DELIVERY + MONETIZATION

# =========================
# BASELINE STATS
# =========================
baseline_stats = {}

for m in ALL_METRICS:
    if m in baseline_df.columns:
        baseline_stats[m] = {
            "mean": baseline_df[m].mean(),
            "std": baseline_df[m].std() or 1
        }

# =========================
# TODAY VS BASELINE
# =========================
signals = {}

for m, s in baseline_stats.items():
    today_val = today_df[m].values[0]
    z = (today_val - s["mean"]) / s["std"]
    pct = (today_val - s["mean"]) / s["mean"] * 100 if s["mean"] else 0

    signals[m] = {
        "today": today_val,
        "z": z,
        "pct": pct
    }

# =========================
# NORMALIZED METRICS
# =========================
if "revenue" in signals and "sessions" in signals:
    signals["revenue_per_session"] = {
        "today": signals["revenue"]["today"] / max(signals["sessions"]["today"], 1),
        "baseline": baseline_stats["revenue"]["mean"] /
                    max(baseline_stats["sessions"]["mean"], 1)
    }

# =========================
# CORE DETECTION FLAGS
# =========================
ecpm_down = signals.get("ecpm", {}).get("z", 0) < -1.2
revenue_down = signals.get("revenue", {}).get("z", 0) < -1
traffic_down = signals.get("sessions", {}).get("z", 0) < -1
traffic_stable = abs(signals.get("sessions", {}).get("z", 0)) < 0.8
quality_ok = (
    signals.get("ctr", {}).get("z", 0) > -1 and
    signals.get("viewability", {}).get("z", 0) > -1
)
fill_down = signals.get("fill_rate", {}).get("z", 0) < -1

# =========================
# ROOT CAUSE ENGINE
# =========================
causes = []

if ecpm_down and traffic_stable:
    causes.append(("Monetization Pressure", 0.9))

if fill_down:
    causes.append(("Delivery / Demand Loss", 0.8))

if traffic_down:
    causes.append(("Traffic Volume Drop", 0.85))

if ecpm_down and quality_ok:
    causes.append(("Advertiser Pricing Softening", 0.7))

if not causes:
    causes.append(("Normal Market Movement", 0.6))

causes = sorted(causes, key=lambda x: x[1], reverse=True)
primary_cause, confidence = causes[0]

# =========================
# DELIVERY INTELLIGENCE
# =========================
delivery_note = ""

if ecpm_down and fill_down:
    delivery_note = "Ads are filling less AND pricing is weaker → Demand-side issue."
elif ecpm_down and quality_ok:
    delivery_note = "Ads are visible & engaged but priced lower → Market pressure."
elif fill_down and not traffic_down:
    delivery_note = "Inventory not filling → Possible ad blocking, policy, or demand loss."
else:
    delivery_note = "No strong delivery anomaly detected."

# =========================
# HEALTH SCORE (0–100)
# =========================
health_score = 100
for m in ["revenue", "ecpm", "sessions", "fill_rate", "viewability"]:
    if m in signals:
        health_score -= min(abs(signals[m]["z"]) * 8, 15)

health_score = max(0, int(health_score))

# =========================
# ALERTING LOGIC
# =========================
if health_score < 50:
    alert = "🔴 CRITICAL"
elif health_score < 70:
    alert = "🟠 WARNING"
else:
    alert = "🟢 HEALTHY"

# =========================
# EXECUTIVE SNAPSHOT
# =========================
st.subheader("📌 Executive Snapshot")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Revenue", f"${signals['revenue']['today']:.2f}", f"{signals['revenue']['pct']:.1f}%")
c2.metric("eCPM", f"${signals['ecpm']['today']:.2f}", f"{signals['ecpm']['pct']:.1f}%")
c3.metric("Sessions", int(signals['sessions']['today']), f"{signals['sessions']['pct']:.1f}%")
c4.metric("Health Score", f"{health_score}/100")
c5.metric("Alert", alert)

st.markdown(f"### 🧠 Primary Cause: **{primary_cause}**")
st.info(f"""
**What changed today?**

Revenue dropped primarily due to **{primary_cause.lower()}**.
Traffic levels remained **{'stable' if traffic_stable else 'volatile'}**, while
monetization efficiency showed **{'clear weakness' if ecpm_down else 'normal behavior'}**.

**Delivery insight:** {delivery_note}

Confidence: **{int(confidence*100)}%**
""")

# =========================
# STORYTELLING CHARTS
# =========================
st.subheader("📈 What changed & why")

# 1️⃣ Revenue vs eCPM (pricing story)
chart_rev = alt.Chart(df).mark_line().encode(
    x="date:T", y="revenue:Q"
).properties(title="Revenue Trend")

chart_ecpm = alt.Chart(df).mark_line(color="orange").encode(
    x="date:T", y="ecpm:Q"
)

st.altair_chart(chart_rev + chart_ecpm, use_container_width=True)

# 2️⃣ Traffic vs Revenue (decoupling)
chart_traf = alt.Chart(df).mark_line().encode(
    x="date:T", y="sessions:Q"
).properties(title="Traffic Stability Check")

st.altair_chart(chart_traf, use_container_width=True)

# 3️⃣ Quality health
quality_cols = [c for c in QUALITY if c in df.columns]
quality_df = df.melt(id_vars="date", value_vars=quality_cols)

chart_q = alt.Chart(quality_df).mark_line().encode(
    x="date:T", y="value:Q", color="variable:N"
).properties(title="Traffic Quality Signals")

st.altair_chart(chart_q, use_container_width=True)

# =========================
# ACTION CHECKLIST
# =========================
st.subheader("🛠 Recommended Actions")

if "Monetization" in primary_cause:
    st.markdown("""
    - Review floor price & bidder competition
    - Check geo/device eCPM splits
    - Compare GAM vs exchange demand
    - Validate auction pressure
    """)

elif "Delivery" in primary_cause:
    st.markdown("""
    - Check fill rate by slot & device
    - Inspect ad blocking & policy flags
    - Validate ads.txt & category blocks
    """)

elif "Traffic" in primary_cause:
    st.markdown("""
    - Investigate traffic source drops
    - Validate GA4 / consent changes
    - Review SEO & Discover trends
    """)

else:
    st.markdown("""
    - No action required
    - Monitor next 3–5 days
    - Compare with market benchmarks
    """)

st.caption("Revenue Intelligence System • Baseline-aware • False-positive protected")
