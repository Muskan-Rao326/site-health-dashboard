# app.py
# PowerBI-style Revenue Intelligence Dashboard (Single Site)
# ✅ Fixes wrong math for ratio metrics (eCPM/CTR/Fill/RPM etc)
# ✅ Root-cause is computed from metric decrements (Revenue → Impressions + eCPM → Requests + Fill)
# ✅ No CTR text blocks (CTR only as KPI + trend + optional diagnostic flag in root cause bullets)
# ✅ Baseline window is variable (7/10/14/30…)
# ✅ Uses robust baseline (Median + MAD) and proper two-sided confidence

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# PAGE CONFIG + STYLE
# =============================
st.set_page_config(page_title="Revenue Intelligence (Root Cause)", layout="wide")

st.markdown(
    """
<style>
    .title { font-size: 26px; font-weight: 800; margin-bottom: 2px; }
    .sub { font-size: 13px; opacity: 0.75; margin-bottom: 14px; }

    .card { border-radius: 14px; padding: 14px; color: white; }
    .kpi-label { font-size: 13px; opacity: 0.95; }
    .kpi-val { font-size: 26px; font-weight: 800; margin-top: 2px; }
    .kpi-meta { font-size: 12px; opacity: 0.95; margin-top: 6px; }

    .pill { display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; margin-right:8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📊 Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Root-cause driven by metric decrements (GA4 + GAM) • single-site</div>', unsafe_allow_html=True)

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(today, prev):
    if prev == 0:
        return 0.0 if today == 0 else 999.0
    return (today - prev) / prev * 100

def robust_z(val, median, mad):
    denom = 1.4826 * mad
    if denom == 0 or np.isnan(denom):
        return 0.0
    return (val - median) / denom

def confidence_from_z(z):
    # two-sided p-value -> confidence
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def color_from_z(z):
    if z >= -1:
        return "#2ECC71"  # green
    if z >= -2:
        return "#F1C40F"  # yellow
    return "#E74C3C"      # red

def kpi_card(title, value, delta_pct, z, conf, suffix=""):
    bg = color_from_z(z)
    return f"""
    <div class="card" style="background:{bg}">
        <div class="kpi-label">{title}</div>
        <div class="kpi-val">{value:,.2f}{suffix}</div>
        <div class="kpi-meta">Δ vs Compare: {delta_pct:+.2f}%</div>
        <div class="kpi-meta">Robust Z: {z:.2f} • Confidence: {conf:.1f}%</div>
    </div>
    """

def line_chart(df, cols, title, height=280):
    d = df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("metric:N", title="Metric"),
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=",.4f")],
    ).properties(title=title, height=height)

def waterfall_like(df_steps, title, height=260):
    # df_steps: Step, Value
    df = df_steps.copy()
    df["Cumulative"] = df["Value"].cumsum()
    df["Start"] = df["Cumulative"] - df["Value"]
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Step:N", sort=None),
        y=alt.Y("Start:Q", title="Value"),
        y2="Cumulative:Q",
        tooltip=["Step", alt.Tooltip("Value:Q", format=",.2f")],
    ).properties(title=title, height=height)

def median_mad(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return 0.0, 0.0
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    return med, mad

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

uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# Single-site mode
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    if len(sites) == 0:
        st.error("No site_name values found.")
        st.stop()
    site = sites[0]
    df_raw = df_raw[df_raw["site_name"] == site].copy()
    st.caption(f"Using site: **{site}** (single-site mode)")
else:
    st.caption("No site_name column found; assuming CSV is already single-site.")

# =============================
# REQUIRED BASE COLUMNS (ADDITIVE ONLY)
# =============================
# IMPORTANT: We only aggregate base totals. Ratios are derived AFTER aggregation.
base_cols = ["revenue", "ad_requests", "impressions", "clicks", "sessions", "users", "pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily truth layer (sum base totals)
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived metrics (correct math)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)  # clicks from GAM
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR (PowerBI-like slicers)
# =============================
st.sidebar.header("Slicers")

selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date(),
)

baseline_days = st.sidebar.slider("Baseline Window (days)", 7, 30, 7)

compare_mode = st.sidebar.selectbox(
    "Compare To",
    ["Yesterday", "Baseline median"],
    index=0
)

today = pd.to_datetime(selected_date).normalize()
yesterday = today - timedelta(days=1)
baseline_start = today - timedelta(days=baseline_days)

today_row = daily[daily["date"] == today]
if today_row.empty:
    st.error("No data for selected date.")
    st.stop()

t = today_row.iloc[0].to_dict()

y_row = daily[daily["date"] == yesterday]
y = y_row.iloc[0].to_dict() if not y_row.empty else None

baseline_df = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)]
display_df = daily[(daily["date"] >= baseline_start - timedelta(days=1)) & (daily["date"] <= today)]

# Compare reference row
if compare_mode == "Yesterday" and y is not None:
    base = y
    base_label = "Yesterday"
else:
    # baseline median for base totals (then we re-derive ratios from those base medians)
    # NOTE: using medians for additive totals is OK for baseline reference (robust)
    base_add = baseline_df[base_cols].median(numeric_only=True).to_dict() if not baseline_df.empty else {c: 0.0 for c in base_cols}
    base = dict(base_add)
    # derive comparable ratios from baseline-median totals
    base["ecpm"] = safe_div(base["revenue"], base["impressions"], 1000)
    base["fill_rate"] = safe_div(base["impressions"], base["ad_requests"], 100)
    base["ctr"] = safe_div(base["clicks"], base["impressions"], 100)
    base["rpm"] = safe_div(base["revenue"], base["pageviews"], 1000)
    base["requests_per_pageview"] = safe_div(base["ad_requests"], base["pageviews"], 1)
    base["impressions_per_session"] = safe_div(base["impressions"], base["sessions"], 1)
    base_label = "Baseline median"

# =============================
# ROBUST BASELINE (Median + MAD) FOR KPIs
# =============================
kpi_defs = [
    ("revenue", "Revenue", ""),
    ("impressions", "Impressions", ""),
    ("ad_requests", "Ad Requests", ""),
    ("fill_rate", "Fill Rate", "%"),
    ("ecpm", "eCPM", ""),
    ("rpm", "RPM", ""),
    ("requests_per_pageview", "Req/Pageview", ""),
    ("ctr", "CTR", "%"),
]

kpi = {}
for key, label, suffix in kpi_defs:
    med, mad = median_mad(baseline_df[key] if key in baseline_df.columns else pd.Series(dtype=float))

    val_today = float(t.get(key, 0.0))
    val_base = float(base.get(key, 0.0))

    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_base) if val_base is not None else 0.0

    kpi[key] = dict(label=label, suffix=suffix, today=val_today, base=val_base, med=med, mad=mad, z=z, conf=conf, delta=delta)

# =============================
# KPI ROW (cards)
# =============================
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi_card("Revenue", kpi["revenue"]["today"], kpi["revenue"]["delta"], kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)
c2.markdown(kpi_card("Impressions", kpi["impressions"]["today"], kpi["impressions"]["delta"], kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)
c3.markdown(kpi_card("Ad Requests", kpi["ad_requests"]["today"], kpi["ad_requests"]["delta"], kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)
c4.markdown(kpi_card("Fill Rate", kpi["fill_rate"]["today"], kpi["fill_rate"]["delta"], kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"], "%"), unsafe_allow_html=True)
c5.markdown(kpi_card("eCPM", kpi["ecpm"]["today"], kpi["ecpm"]["delta"], kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

# =============================
# EXPECTED REVENUE (Forecast)
# =============================
# Expected = median revenue over baseline window (excluding today)
expected_revenue = float(baseline_df["revenue"].median()) if not baseline_df.empty else 0.0
actual_revenue = float(t["revenue"])
lost = expected_revenue - actual_revenue
lost_pct = safe_div(lost, expected_revenue, 100) if expected_revenue else 0.0

# =============================
# ROOT CAUSE DECOMPOSITION (CORRECT LOGIC)
# =============================
st.subheader("🧠 Root Cause (computed from decrement math)")

left, right = st.columns([2, 1])

with right:
    st.subheader("🎯 Expected Revenue")
    st.metric("Expected (Baseline Median)", f"{expected_revenue:,.2f}")
    st.metric("Actual (Today)", f"{actual_revenue:,.2f}", delta=f"{pct_change(actual_revenue, expected_revenue):+.2f}%" if expected_revenue else None)
    st.metric("Lost vs Expected", f"{max(lost, 0):,.2f}", delta=f"{lost_pct:+.2f}%" if expected_revenue else None)
    st.caption(f"Baseline window: last **{baseline_days}** days (median). Compare mode: **{base_label}**.")

with left:
    # Base vs Today values (use base_label selection)
    rev_b = float(base.get("revenue", 0.0))
    rev_t = float(t.get("revenue", 0.0))
    d_rev = rev_t - rev_b

    imp_b = float(base.get("impressions", 0.0))
    imp_t = float(t.get("impressions", 0.0))

    # derive eCPM from base/today totals to keep equation consistent
    ecpm_b = safe_div(rev_b, imp_b, 1000)
    ecpm_t = safe_div(rev_t, imp_t, 1000)

    # Revenue = Impressions * eCPM / 1000
    # ΔRevenue ≈ (ΔImpressions * base_eCPM/1000) + (today_impressions * ΔeCPM/1000) + residual
    imp_effect = (imp_t - imp_b) * (ecpm_b / 1000)
    ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000)
    residual = d_rev - (imp_effect + ecpm_effect)

    wf_rev = pd.DataFrame({
        "Step": [f"{base_label} Revenue", "Impressions effect", "eCPM effect", "Residual", "Today Revenue"],
        "Value": [rev_b, imp_effect, ecpm_effect, residual, rev_t]
    })
    st.altair_chart(waterfall_like(wf_rev, "Revenue Decomposition"), use_container_width=True)

    # Impressions = Requests * FillRate/100
    req_b = float(base.get("ad_requests", 0.0))
    req_t = float(t.get("ad_requests", 0.0))

    fill_b = safe_div(imp_b, req_b, 100) if req_b else 0.0
    fill_t = safe_div(imp_t, req_t, 100) if req_t else 0.0

    d_imp = imp_t - imp_b

    req_effect = (req_t - req_b) * (fill_b / 100)
    fill_effect = req_t * ((fill_t - fill_b) / 100)
    imp_residual = d_imp - (req_effect + fill_effect)

    wf_imp = pd.DataFrame({
        "Step": [f"{base_label} Impressions", "Requests effect", "Fill effect", "Residual", "Today Impressions"],
        "Value": [imp_b, req_effect, fill_effect, imp_residual, imp_t]
    })
    st.altair_chart(waterfall_like(wf_imp, "Impressions Decomposition"), use_container_width=True)

    # =============================
    # ROOT CAUSE BULLETS (PowerBI-style, minimal text)
    # =============================
    st.markdown("### ✅ Root Cause Summary")

    direction = "down" if d_rev < 0 else "up"
    st.write(f"• Revenue moved **{direction} {abs(d_rev):,.2f}** vs **{base_label}**.")

    # Determine main driver by absolute contribution (ignore residual if tiny)
    contribs = [
        ("Impressions", imp_effect),
        ("eCPM", ecpm_effect),
        ("Residual", residual),
    ]
    contribs_sorted = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
    top_driver, top_val = contribs_sorted[0]

    # Contribution percentages (only meaningful when revenue moved)
    denom = abs(d_rev) if abs(d_rev) > 1e-9 else 1.0
    imp_share = abs(imp_effect) / denom * 100
    ecpm_share = abs(ecpm_effect) / denom * 100

    st.write(f"• Split of change (approx): **{imp_share:.0f}% Impressions**, **{ecpm_share:.0f}% eCPM**.")

    if top_driver == "Impressions":
        st.write(f"• Primary driver: **Impressions** (impact {imp_effect:,.2f}).")
        inner = [("Requests", req_effect), ("Fill rate", fill_effect)]
        inner_sorted = sorted(inner, key=lambda x: abs(x[1]), reverse=True)
        inner_driver, inner_val = inner_sorted[0]
        if inner_driver == "Requests":
            st.write(f"• Impressions moved mainly due to **Ad Requests** (impact {req_effect:,.0f} impressions).")
            # request drop: traffic vs ad-loading using requests/pageview
            pv_b = float(base.get("pageviews", 0.0))
            pv_t = float(t.get("pageviews", 0.0))
            rpp_b = safe_div(req_b, pv_b, 1)
            rpp_t = safe_div(req_t, pv_t, 1)
            if rpp_t < rpp_b * 0.9:
                st.write("• Pattern matches **Ad Loading issue** (requests per pageview decreased).")
            else:
                st.write("• Pattern matches **Traffic/engagement change** (requests per pageview stable).")
        else:
            st.write(f"• Impressions moved mainly due to **Fill Rate** (impact {fill_effect:,.0f} impressions).")
            st.write("• This points to **Demand/Blocks/Floors/Policy** type issue (requests exist but not monetizing).")

    elif top_driver == "eCPM":
        st.write(f"• Primary driver: **eCPM** (impact {ecpm_effect:,.2f}).")
        st.write("• Likely causes: **geo/device mix shift**, **auction pressure**, **floor changes**, **demand partner issues** (needs segmentation to pinpoint).")
    else:
        st.write("• Residual is unusually large → likely **data mismatch** (reporting lag, merge issue, or mixed movements).")

    # CTR as diagnostic only (no extra section, just a small flag if extreme)
    ctr_b = safe_div(float(base.get("clicks", 0.0)), imp_b, 100) if imp_b else 0.0
    ctr_t = safe_div(float(t.get("clicks", 0.0)), imp_t, 100) if imp_t else 0.0
    if ctr_b > 0 and ctr_t > ctr_b * 1.5:
        st.write("• CTR spiked sharply → **diagnostic flag** (layout change / accidental clicks / policy risk).")
    elif ctr_b > 0 and ctr_t < ctr_b * 0.6:
        st.write("• CTR dropped sharply → **diagnostic flag** (ad mix/placement engagement changed).")

# =============================
# TRENDS (PowerBI-like drill)
# =============================
st.subheader("📈 Trends (Baseline window → Today)")

t1, t2 = st.columns(2)
with t1:
    st.altair_chart(line_chart(display_df, ["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
with t2:
    st.altair_chart(line_chart(display_df, ["impressions", "ad_requests", "fill_rate"], "Requests → Fill → Impressions"), use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    st.altair_chart(line_chart(display_df, ["sessions", "pageviews"], "Traffic (Sessions + Pageviews)"), use_container_width=True)
with t4:
    st.altair_chart(line_chart(display_df, ["requests_per_pageview", "impressions_per_session"], "Ad Loading & Density"), use_container_width=True)

# =============================
# ISSUES TABLE (ranked)
# =============================
st.subheader("🧯 Top Issues (ranked by drop + anomaly)")

metrics_for_table = [
    ("revenue", "Revenue"),
    ("sessions", "Sessions"),
    ("pageviews", "Pageviews"),
    ("ad_requests", "Ad Requests"),
    ("requests_per_pageview", "Requests/Pageview"),
    ("fill_rate", "Fill Rate"),
    ("impressions", "Impressions"),
    ("ecpm", "eCPM"),
    ("rpm", "RPM"),
    ("ctr", "CTR"),
]

rows = []
for key, label in metrics_for_table:
    med, mad = median_mad(baseline_df[key] if key in baseline_df.columns else pd.Series(dtype=float))
    val_today = float(t.get(key, 0.0))
    val_base = float(base.get(key, 0.0))
    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_base) if val_base else 0.0

    # score: punish negative z + negative delta
    score = 0.0
    if z < 0:
        score += abs(z) * 2.0
    if delta < 0:
        score += abs(delta) / 10.0

    rows.append({
        "Metric": label,
        "Today": val_today,
        f"Compare ({base_label})": val_base,
        "Δ %": delta,
        "Baseline Median": med,
        "Robust Z": z,
        "Confidence %": conf,
        "_score": score
    })

issues = pd.DataFrame(rows).sort_values("_score", ascending=False).drop(columns=["_score"]).head(10)
st.dataframe(issues, use_container_width=True)

# =============================
# DATA QUALITY CHECKS (silent but critical)
# =============================
st.subheader("🧪 Data Sanity Checks")

checks = []
if t["sessions"] > 0 and t["ad_requests"] == 0:
    checks.append("Sessions > 0 but Ad Requests = 0 → tag/rendering issue or missing GAM data.")
if t["ad_requests"] > 0 and t["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed or reporting lag.")
if t["impressions"] > 0 and t["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue reporting lag or merge issue.")

if not checks:
    st.success("No obvious data integrity red flags found for selected day.")
else:
    for c in checks:
        st.warning(c)

st.caption("All ratios (eCPM/CTR/Fill/RPM/etc) are derived from daily base totals. Root cause follows the math: Revenue → (Impressions + eCPM), Impressions → (Requests + Fill).")
