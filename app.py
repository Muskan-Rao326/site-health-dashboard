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
    # two-sided confidence
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def safe_div(n, d, multiplier=1.0):
    return (n / d) * multiplier if d and d != 0 else 0.0

def render_kpi(col, title, val, delta_pct, z, conf):
    # z-based colors (negative anomalies)
    if z >= -1:
        color = "#2ECC71"  # green
    elif z >= -2:
        color = "#F1C40F"  # yellow
    else:
        color = "#E74C3C"  # red

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
    site = sites[0]  # one site only (first)
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
    "clicks",
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
# DAILY AGGREGATION (THE KEY FIX)
# =============================
daily = df_raw.groupby("date", as_index=False)[required_additive].sum()

# =============================
# DERIVED METRICS (RECOMPUTE, DON'T TRUST CSV)
# =============================
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)  # %
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)  # %
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)

# If your CSV has these as base additive columns, we can compute these too.
# Otherwise we keep them out of KPI.
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
# KPI CONFIG
# =============================
# Additive + derived metrics that are safe to show
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
]

# Optional ratios (only if present)
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
# KPI DISPLAY (TOP ROW)
# =============================
st.subheader("📌 Key Metrics (Daily Totals + Correct Derived Rates)")

top_keys = ["revenue", "ecpm", "sessions", "ad_requests"]
cols = st.columns(4)
for i, k in enumerate(top_keys):
    render_kpi(
        cols[i],
        kpi[k]["label"],
        kpi[k]["today"],
        kpi[k]["delta_pct"],
        kpi[k]["z"],
        kpi[k]["confidence"],
    )

# Second row KPIs
st.write("")
cols2 = st.columns(4)
second_keys = ["fill_rate", "impressions", "rpm", "requests_per_pageview"]
for i, k in enumerate(second_keys):
    render_kpi(
        cols2[i],
        kpi[k]["label"],
        kpi[k]["today"],
        kpi[k]["delta_pct"],
        kpi[k]["z"],
        kpi[k]["confidence"],
    )

# =============================
# LEAKAGE PIPELINE INSIGHT ENGINE
# =============================
st.subheader("🧠 Key Reasons (Leakage Pipeline)")

def leakage_reasons(t, y):
    if not y:
        return ["No previous-day data available to explain movement."], "info"

    d_rev = pct_change(t["revenue"], y["revenue"])
    d_imp = pct_change(t["impressions"], y["impressions"])
    d_ecpm = pct_change(t["ecpm"], y["ecpm"])
    d_req = pct_change(t["ad_requests"], y["ad_requests"])
    d_fill = pct_change(t["fill_rate"], y["fill_rate"])
    d_sess = pct_change(t["sessions"], y["sessions"])
    d_pv = pct_change(t["pageviews"], y["pageviews"])
    d_rpp = pct_change(t["requests_per_pageview"], y["requests_per_pageview"])

    def big_drop(x, th):
        return x <= -th

    reasons = []
    severity = "success"

    # trigger threshold
    if not big_drop(d_rev, 10):
        reasons.append("Revenue is within normal day-to-day movement.")
        return reasons, "success"

    severity = "error"
    reasons.append(f"Revenue is down **{d_rev:.1f}%** vs yesterday.")

    # Revenue = Impressions * eCPM
    if big_drop(d_imp, 10) and not big_drop(d_ecpm, 15):
        reasons.append(f"Primary driver: **Impressions fell {d_imp:.1f}%** while eCPM stayed relatively stable ({d_ecpm:.1f}%).")

        # Impressions = Requests * Fill
        if big_drop(d_req, 10) and not big_drop(d_fill, 10):
            reasons.append(f"Impressions fell because **Ad Requests fell {d_req:.1f}%** while Fill Rate stayed stable ({d_fill:.1f}%).")

            # Requests: traffic vs technical
            if big_drop(d_sess, 10) or big_drop(d_pv, 10):
                reasons.append(f"Ad Requests fell due to **traffic/engagement decline** (Sessions {d_sess:.1f}%, Pageviews {d_pv:.1f}%).")
            elif big_drop(d_rpp, 10):
                reasons.append("Traffic is stable but **Requests/Pageview dropped** → likely tag/CMP/lazyload/JS issue (ads not loading).")
            else:
                reasons.append("Requests fell while traffic looks stable → check recent tag/placement changes, ad unit rendering, CMP, latency.")
        elif big_drop(d_fill, 10) and not big_drop(d_req, 10):
            reasons.append(f"Impressions fell because **Fill Rate fell {d_fill:.1f}%** while Requests were stable ({d_req:.1f}%).")
            reasons.append("Likely causes: floors too high, blocking rules, policy limitation, demand partner outage, size mismatch.")
        else:
            reasons.append("Impressions fell due to a mix of lower requests and lower fill. Investigate both technical loading and demand controls.")

    elif big_drop(d_ecpm, 15) and not big_drop(d_imp, 10):
        reasons.append(f"Primary driver: **eCPM fell {d_ecpm:.1f}%** while impressions were relatively stable ({d_imp:.1f}%).")
        reasons.append("This points to an **auction/value** issue (geo/device mix shift, viewability drop, fewer bidders, floors/blocks).")
    else:
        reasons.append(f"Mixed driver: Impressions ({d_imp:.1f}%) and eCPM ({d_ecpm:.1f}%) both declined.")
        reasons.append("Investigate: traffic mix + ad loading + demand/floors + viewability/bidder competition.")

    return reasons, severity

reasons, severity = leakage_reasons(t, y)

if severity == "success":
    st.success("🟢 " + reasons[0])
else:
    st.error("🔴 " + reasons[0])
    for r in reasons[1:]:
        st.write("• " + r)

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
st.altair_chart(line_chart(["rpm", "requests_per_pageview"], "RPM + Requests/Pageview (Leakage Signals)"), use_container_width=True)

# =============================
# KPI TABLE (FULL)
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
kpi_table = pd.DataFrame(rows)
st.dataframe(kpi_table, use_container_width=True)

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

st.caption("This dashboard recomputes derived metrics (eCPM/CTR/Fill/RPM) from base totals for correct daily reporting.")
