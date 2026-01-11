import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Revenue Intelligence Dashboard", layout="wide")
st.title("📊 Revenue Intelligence Dashboard (GA4 + GAM)")

# =============================
# HELPERS
# =============================
def pct_change(today, prev):
    if prev == 0:
        return 0.0 if today == 0 else 999.0
    return (today - prev) / prev * 100

def z_score(val, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return (val - mean) / std

def confidence_from_z(z):
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def safe_div(n, d, multiplier=1.0):
    return (n / d) * multiplier if d and d != 0 else 0.0

def render_kpi(col, title, val, delta_pct, z, conf):
    if z >= -1:
        color = "#2ECC71"
    elif z >= -2:
        color = "#F1C40F"
    else:
        color = "#E74C3C"

    col.markdown(
        f"""
        <div style="background:{color};padding:14px;border-radius:12px;color:white">
            <div style="font-size:14px;opacity:0.95">{title}</div>
            <div style="font-size:28px;font-weight:700">{val:,.2f}</div>
            <div style="margin-top:6px;font-size:13px">Δ vs Yesterday: {delta_pct:+.2f}%</div>
            <div style="font-size:13px">Z-score: {z:.2f}</div>
            <div style="font-size:13px">Confidence: {conf:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def status_chip(label, state, note=""):
    # state: green/yellow/red
    if state == "green":
        bg = "#2ECC71"
        icon = "🟢"
    elif state == "yellow":
        bg = "#F1C40F"
        icon = "🟠"
    else:
        bg = "#E74C3C"
        icon = "🔴"

    st.markdown(
        f"""
        <div style="background:{bg};padding:10px 12px;border-radius:12px;color:white;margin-bottom:8px">
          <div style="font-size:13px;opacity:0.95">{icon} {label}</div>
          <div style="font-size:12px;opacity:0.95">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

uploaded = st.file_uploader("Upload merged GA4+GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# =============================
# SINGLE SITE FILTER (one site only)
# =============================
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    if len(sites) == 0:
        st.error("No site_name values found in CSV.")
        st.stop()
    site = sites[0]  # single-site mode
    df_raw = df_raw[df_raw["site_name"] == site].copy()
    st.caption(f"Using site: **{site}** (single-site mode)")
else:
    st.caption("No 'site_name' column found; assuming CSV is already single-site.")

# =============================
# REQUIRED BASE METRICS (ADDITIVE)
# =============================
required_additive = [
    "revenue",
    "ad_requests",
    "impressions",
    "clicks",     # clicks from GAM (confirmed)
    "sessions",
    "users",
    "pageviews",
]

missing = [c for c in required_additive if c not in df_raw.columns]
if missing:
    st.error(f"CSV is missing required additive columns: {missing}")
    st.stop()

# Ensure numeric
for c in required_additive:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# =============================
# DAILY AGGREGATION
# =============================
daily = df_raw.groupby("date", as_index=False)[required_additive].sum()

# =============================
# DERIVED METRICS (RECOMPUTE CLEANLY)
# =============================
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)  # %
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)  # %
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)

# Optional columns (if present as additive bases)
if "engaged_sessions" in df_raw.columns:
    df_raw["engaged_sessions"] = pd.to_numeric(df_raw["engaged_sessions"], errors="coerce").fillna(0.0)
    engaged = df_raw.groupby("date", as_index=False)["engaged_sessions"].sum()
    daily = daily.merge(engaged, on="date", how="left").fillna({"engaged_sessions": 0.0})
    daily["engagement_rate"] = daily.apply(lambda r: safe_div(r["engaged_sessions"], r["sessions"], 100), axis=1)
else:
    daily["engagement_rate"] = np.nan

if "measurable_impressions" in df_raw.columns and "viewable_impressions" in df_raw.columns:
    df_raw["measurable_impressions"] = pd.to_numeric(df_raw["measurable_impressions"], errors="coerce").fillna(0.0)
    df_raw["viewable_impressions"] = pd.to_numeric(df_raw["viewable_impressions"], errors="coerce").fillna(0.0)
    vv = df_raw.groupby("date", as_index=False)[["measurable_impressions", "viewable_impressions"]].sum()
    daily = daily.merge(vv, on="date", how="left").fillna({"measurable_impressions": 0.0, "viewable_impressions": 0.0})
    daily["viewability"] = daily.apply(lambda r: safe_div(r["viewable_impressions"], r["measurable_impressions"], 100), axis=1)
else:
    daily["viewability"] = np.nan

daily = daily.sort_values("date")

# =============================
# DATE CONTROLS
# =============================
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=daily["date"].max().date(),
        min_value=daily["date"].min().date(),
        max_value=daily["date"].max().date(),
    )

with col2:
    baseline_days = st.slider("Baseline Window (days)", 7, 30, 7)

today = pd.to_datetime(selected_date).normalize()
yesterday = today - timedelta(days=1)
baseline_start = today - timedelta(days=baseline_days)

today_row = daily[daily["date"] == today]
yesterday_row = daily[daily["date"] == yesterday]
baseline_df = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)]
display_df = daily[(daily["date"] >= baseline_start - timedelta(days=1)) & (daily["date"] <= today)]

if today_row.empty:
    st.error("No data for selected date in this CSV.")
    st.stop()

t = today_row.iloc[0].to_dict()
y = yesterday_row.iloc[0].to_dict() if not yesterday_row.empty else None

# =============================
# KPI LIST
# =============================
kpi_metrics = [
    ("revenue", "Revenue"),
    ("sessions", "Sessions"),
    ("pageviews", "Pageviews"),
    ("ad_requests", "Ad Requests"),
    ("fill_rate", "Fill Rate (%)"),
    ("impressions", "Impressions"),
    ("ecpm", "eCPM"),
    ("rpm", "RPM"),
    ("requests_per_pageview", "Requests/Pageview"),
    ("ctr", "CTR (%)"),
    ("impressions_per_session", "Impressions/Session"),
]
if not np.isnan(t.get("viewability", np.nan)):
    kpi_metrics.append(("viewability", "Viewability (%)"))
if not np.isnan(t.get("engagement_rate", np.nan)):
    kpi_metrics.append(("engagement_rate", "Engagement Rate (%)"))

# =============================
# KPI CALC: baseline mean/std + z
# =============================
kpi = {}
for key, label in kpi_metrics:
    series = baseline_df[key].dropna()
    mean = float(series.mean()) if len(series) else 0.0
    std = float(series.std()) if len(series) else 0.0
    val_today = float(t.get(key, 0.0))
    val_y = float(y.get(key, 0.0)) if y else 0.0

    z = z_score(val_today, mean, std)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_y) if y else 0.0

    kpi[key] = {
        "label": label,
        "today": val_today,
        "yesterday": val_y,
        "baseline_mean": mean,
        "baseline_std": std,
        "z": z,
        "confidence": conf,
        "delta_pct": delta,
    }

# =============================
# TOP KPIs DISPLAY
# =============================
st.subheader("📌 Key Metrics")

cols = st.columns(4)
for i, k in enumerate(["revenue", "ecpm", "sessions", "ad_requests"]):
    render_kpi(cols[i], kpi[k]["label"], kpi[k]["today"], kpi[k]["delta_pct"], kpi[k]["z"], kpi[k]["confidence"])

cols2 = st.columns(4)
for i, k in enumerate(["fill_rate", "impressions", "rpm", "requests_per_pageview"]):
    render_kpi(cols2[i], kpi[k]["label"], kpi[k]["today"], kpi[k]["delta_pct"], kpi[k]["z"], kpi[k]["confidence"])

# =============================
# LEAKAGE PIPELINE: REASONS + ROOT CAUSE CLASS
# =============================
st.subheader("🧠 Key Reasons (Leakage Pipeline)")

def leakage_analysis(t, y):
    if not y:
        return ["No previous-day data available."], "info", "unknown"

    d = {
        "revenue": pct_change(t["revenue"], y["revenue"]),
        "impressions": pct_change(t["impressions"], y["impressions"]),
        "ecpm": pct_change(t["ecpm"], y["ecpm"]),
        "ad_requests": pct_change(t["ad_requests"], y["ad_requests"]),
        "fill_rate": pct_change(t["fill_rate"], y["fill_rate"]),
        "sessions": pct_change(t["sessions"], y["sessions"]),
        "pageviews": pct_change(t["pageviews"], y["pageviews"]),
        "requests_per_pageview": pct_change(t["requests_per_pageview"], y["requests_per_pageview"]),
        "viewability": pct_change(t.get("viewability", 0.0), y.get("viewability", 0.0)) if "viewability" in t and "viewability" in y else 0.0,
    }

    def big_drop(x, th):
        return x <= -th

    reasons = []
    root = "unknown"

    # Only fire if meaningful revenue drop
    if not big_drop(d["revenue"], 10):
        return ["Revenue is within normal day-to-day movement."], "success", "normal"

    reasons.append(f"Revenue is down **{d['revenue']:.1f}%** vs yesterday.")

    # Revenue = Impressions * eCPM
    if big_drop(d["impressions"], 10) and not big_drop(d["ecpm"], 15):
        reasons.append(f"Primary driver: **Impressions fell {d['impressions']:.1f}%** while eCPM stayed relatively stable ({d['ecpm']:.1f}%).")

        # Impressions = Requests * Fill
        if big_drop(d["ad_requests"], 10) and not big_drop(d["fill_rate"], 10):
            root = "ad_loading_or_traffic"
            reasons.append(f"Impressions fell because **Ad Requests fell {d['ad_requests']:.1f}%** while Fill Rate stayed stable ({d['fill_rate']:.1f}%).")

            if big_drop(d["sessions"], 10) or big_drop(d["pageviews"], 10):
                root = "traffic"
                reasons.append(f"Ad Requests fell due to **traffic/engagement decline** (Sessions {d['sessions']:.1f}%, Pageviews {d['pageviews']:.1f}%).")
            elif big_drop(d["requests_per_pageview"], 10):
                root = "ad_loading"
                reasons.append("Traffic is stable but **Requests/Pageview dropped** → likely tag/CMP/lazyload/JS issue (ads not loading).")
            else:
                root = "ad_loading"
                reasons.append("Requests fell while traffic looks stable → check tag/placement changes, CMP, latency, ad unit rendering.")
        elif big_drop(d["fill_rate"], 10) and not big_drop(d["ad_requests"], 10):
            root = "demand"
            reasons.append(f"Impressions fell because **Fill Rate fell {d['fill_rate']:.1f}%** while Requests were stable ({d['ad_requests']:.1f}%).")
            reasons.append("Likely causes: floors too high, blocking rules, policy limitation, demand partner outage, size mismatch.")
        else:
            root = "mixed_delivery"
            reasons.append("Impressions fell due to a mix of lower requests and lower fill. Investigate both technical loading and demand controls.")

    elif big_drop(d["ecpm"], 15) and not big_drop(d["impressions"], 10):
        root = "value"
        reasons.append(f"Primary driver: **eCPM fell {d['ecpm']:.1f}%** while impressions were relatively stable ({d['impressions']:.1f}%).")
        if not np.isnan(t.get("viewability", np.nan)) and big_drop(d["viewability"], 10):
            reasons.append(f"Viewability also fell {d['viewability']:.1f}% → placement/layout/latency likely reduced bids.")
        reasons.append("This points to an **auction/value** issue: geo/device mix shift, viewability, fewer bidders, floors/blocks.")
    else:
        root = "mixed"
        reasons.append(f"Mixed driver: Impressions ({d['impressions']:.1f}%) and eCPM ({d['ecpm']:.1f}%) both declined.")
        reasons.append("Investigate: traffic mix + ad loading + demand/floors + viewability/bidder competition.")

    return reasons, "error", root

reasons, severity, root_cause = leakage_analysis(t, y)

if severity == "success":
    st.success("🟢 " + reasons[0])
else:
    st.error("🔴 " + reasons[0])
    for r in reasons[1:]:
        st.write("• " + r)

# =============================
# STATUS PANEL (ONE GO)
# =============================
st.subheader("🚦 Status Panel (Where to Look Today)")

def classify_state(z):
    if z >= -1:
        return "green"
    if z >= -2:
        return "yellow"
    return "red"

c1, c2, c3, c4 = st.columns(4)

with c1:
    status_chip("Traffic", classify_state(kpi["sessions"]["z"]),
                f"Sessions Δ {kpi['sessions']['delta_pct']:+.1f}%, Z {kpi['sessions']['z']:.2f}")

with c2:
    status_chip("Ad Loading", classify_state(kpi["requests_per_pageview"]["z"]),
                f"Req/Pageview Δ {kpi['requests_per_pageview']['delta_pct']:+.1f}%, Z {kpi['requests_per_pageview']['z']:.2f}")

with c3:
    status_chip("Demand", classify_state(kpi["fill_rate"]["z"]),
                f"Fill Rate Δ {kpi['fill_rate']['delta_pct']:+.1f}%, Z {kpi['fill_rate']['z']:.2f}")

with c4:
    status_chip("Value", classify_state(kpi["ecpm"]["z"]),
                f"eCPM Δ {kpi['ecpm']['delta_pct']:+.1f}%, Z {kpi['ecpm']['z']:.2f}")

# =============================
# TOP 3 SUSPECT METRICS (AUTO-RANKED)
# =============================
st.subheader("🧯 Top 3 Suspect Metrics (Auto)")

suspects = []
for key, meta in kpi.items():
    if key == "revenue":
        continue
    # Score: prioritize big negative z + big negative delta
    z = meta["z"]
    dlt = meta["delta_pct"]
    score = 0
    if z < 0:
        score += abs(z) * 2
    if dlt < 0:
        score += abs(dlt) / 10  # scale delta weight
    suspects.append((score, key, meta["label"], meta["today"], meta["delta_pct"], meta["z"], meta["confidence"]))

suspects = sorted(suspects, key=lambda x: x[0], reverse=True)[:3]

scols = st.columns(3)
for i, s in enumerate(suspects):
    _, key, label, val, dlt, z, conf = s
    render_kpi(scols[i], f"Suspect: {label}", val, dlt, z, conf)

# =============================
# ACTION RECOMMENDATIONS (DYNAMIC)
# =============================
st.subheader("🛠 What to Check + What to Fix (Next Steps)")

def action_recommendations(root):
    if root == "traffic":
        return [
            "Check **GA4 Traffic Acquisition**: which source/medium dropped (SEO, Direct, Paid).",
            "Check **Landing pages**: did top 5 pages lose sessions?",
            "Verify site availability / speed issues (page load, errors).",
            "If SEO-driven: check if a specific country/device lost traffic first."
        ]
    if root == "ad_loading":
        return [
            "Compare **Requests/Pageview** today vs baseline (this catches tag/CMP/lazyload issues).",
            "Check if a recent deploy changed ad tags, slot IDs, or lazy load trigger.",
            "Validate CMP/consent signals: are requests blocked until consent?",
            "Check GAM ad unit rendering: any key ad unit showing near-zero requests?"
        ]
    if root == "demand":
        return [
            "Check **Unfilled impressions** and fill by **country/device/ad unit** in GAM.",
            "Audit **floor prices** (especially recent changes) and price rules.",
            "Review blocking: category blocks, advertiser domain blocks, brand safety changes.",
            "If only one geo dropped: demand partner / policy restriction for that region."
        ]
    if root == "value":
        return [
            "Break down eCPM by **country + device**: is the drop only on a segment?",
            "Check **viewability** (if available). Layout changes can reduce bids quickly.",
            "Check auction competition: did a bidder stop winning / disappear?",
            "Review floors/blocks: sometimes value drops because you changed auction dynamics."
        ]
    if root in ["mixed", "mixed_delivery"]:
        return [
            "Both delivery and value moved: split analysis into 2 threads:",
            "1) Delivery: Requests/Pageview + Fill Rate",
            "2) Value: eCPM by geo/device + viewability",
            "Look for the first metric that broke sharply vs baseline."
        ]
    return [
        "No strong root cause classification yet. Start with:",
        "Requests/Pageview → Fill Rate → eCPM, then drill into geo/device."
    ]

steps = action_recommendations(root_cause)
for s in steps:
    st.write("• " + s)

# =============================
# TRENDS
# =============================
st.subheader("📈 Trends (Baseline → Today)")

def line_chart(cols, title):
    d = display_df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("metric:N", title="Metric"),
        tooltip=["date:T", "metric:N", "value:Q"],
    ).properties(title=title, height=260)

st.altair_chart(line_chart(["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
st.altair_chart(line_chart(["sessions", "pageviews", "ad_requests"], "Traffic → Pageviews → Ad Requests"), use_container_width=True)
st.altair_chart(line_chart(["fill_rate", "impressions"], "Fill Rate → Impressions"), use_container_width=True)
st.altair_chart(line_chart(["rpm", "requests_per_pageview", "impressions_per_session"], "Leakage Signals"), use_container_width=True)

# =============================
# KPI TABLE
# =============================
st.subheader("📋 KPI Table (Today vs Yesterday vs Baseline)")
rows = []
for key, label in kpi_metrics:
    rows.append({
        "Metric": label,
        "Today": kpi[key]["today"],
        "Yesterday": kpi[key]["yesterday"],
        "Δ %": kpi[key]["delta_pct"],
        "Baseline Mean": kpi[key]["baseline_mean"],
        "Baseline Std": kpi[key]["baseline_std"],
        "Z": kpi[key]["z"],
        "Confidence %": kpi[key]["confidence"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =============================
# Z-SCORE GUIDE
# =============================
st.subheader("📘 Z-Score Guide")
st.markdown("""
| Z-score | Meaning | Action |
|---|---|---|
| 0 to -1 | Normal | No action |
| -1 to -2 | Watch | Monitor |
| < -2 | Anomaly | Investigate immediately |
""")

st.caption("Derived metrics (eCPM/CTR/Fill/RPM) are recomputed from base totals for correct daily reporting.")
