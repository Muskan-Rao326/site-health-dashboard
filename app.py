# app.py
# Revenue Intelligence Dashboard (PowerBI-style) — Single Site
# ✅ Correct math: only SUM base totals; all ratios derived from totals
# ✅ Root cause from decrement math:
#    Revenue → (Impressions effect + eCPM effect)
#    Impressions → (Requests effect + Fill effect)
# ✅ “Today vs Expected Loss Meter” + loss allocation (Expected → Today)
# ✅ No CTR explanation text (CTR only KPI + small diagnostic flag if extreme)
# ✅ Baseline window variable (7–30) using median + MAD (robust)

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
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📊 Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Root-cause via decrement math (GA4 + GAM) • single-site</div>', unsafe_allow_html=True)

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(today, prev):
    if prev == 0:
        return 0.0 if today == 0 else 999.0
    return (today - prev) / prev * 100

def median_mad(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return 0.0, 0.0
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    return med, mad

def robust_z(val, median, mad):
    denom = 1.4826 * mad
    if denom == 0 or np.isnan(denom):
        return 0.0
    return (val - median) / denom

def confidence_from_z(z):
    p = 2 * (1 - norm.cdf(abs(z)))  # two-sided p
    return (1 - p) * 100

def color_from_z(z):
    if z >= -1:
        return "#2ECC71"
    if z >= -2:
        return "#F1C40F"
    return "#E74C3C"

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

def loss_meter_card(expected, actual):
    gap = expected - actual  # + means loss
    gap_abs = abs(gap)
    gap_pct = safe_div(gap, expected, 100) if expected > 0 else 0.0

    if expected <= 0:
        bg = "#7F8C8D"
        status = "No baseline"
    elif gap_pct <= 5:
        bg = "#2ECC71"
        status = "Normal"
    elif gap_pct <= 15:
        bg = "#F1C40F"
        status = "Warning"
    else:
        bg = "#E74C3C"
        status = "Critical"

    label = "Lost vs Expected" if gap > 0 else "Above Expected"
    sign = "-" if gap > 0 else "+"

    return f"""
    <div class="card" style="background:{bg}">
        <div class="kpi-label">Today vs Expected</div>
        <div class="kpi-val">{sign}{gap_abs:,.2f}</div>
        <div class="kpi-meta">{label} • Status: {status}</div>
        <div class="kpi-meta">Expected: {expected:,.2f}</div>
        <div class="kpi-meta">Actual: {actual:,.2f}</div>
        <div class="kpi-meta">Gap: {gap_pct:+.1f}%</div>
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
    df = df_steps.copy()
    df["Cumulative"] = df["Value"].cumsum()
    df["Start"] = df["Cumulative"] - df["Value"]
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Step:N", sort=None),
        y=alt.Y("Start:Q", title="Value"),
        y2="Cumulative:Q",
        tooltip=["Step", alt.Tooltip("Value:Q", format=",.2f")],
    ).properties(title=title, height=height)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

# =============================
# LOAD CSV
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# Single site (auto-pick first)
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    if not sites:
        st.error("site_name column exists but no values found.")
        st.stop()
    site = sites[0]
    df_raw = df_raw[df_raw["site_name"] == site].copy()
    st.caption(f"Using site: **{site}** (single-site mode)")
else:
    st.caption("No site_name column found; assuming already single-site.")

# =============================
# BASE TOTALS ONLY (ADDITIVE)
# =============================
base_cols = ["revenue", "ad_requests", "impressions", "clicks", "sessions", "users", "pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily totals
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived ratios (correct)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)
daily["pageviews_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR SLICERS
# =============================
st.sidebar.header("Slicers")

selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date(),
)
baseline_days = st.sidebar.slider("Baseline Window (days)", 7, 30, 7)
compare_mode = st.sidebar.selectbox("Compare To", ["Yesterday", "Baseline median"], index=0)

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

# Compare base row
if compare_mode == "Yesterday" and y is not None:
    base = y
    base_label = "Yesterday"
else:
    base_add = baseline_df[base_cols].median(numeric_only=True).to_dict() if not baseline_df.empty else {c: 0.0 for c in base_cols}
    base = dict(base_add)
    # derive base ratios from base totals (consistent)
    base["ecpm"] = safe_div(base["revenue"], base["impressions"], 1000)
    base["fill_rate"] = safe_div(base["impressions"], base["ad_requests"], 100)
    base["ctr"] = safe_div(base["clicks"], base["impressions"], 100)
    base["rpm"] = safe_div(base["revenue"], base["pageviews"], 1000)
    base["requests_per_pageview"] = safe_div(base["ad_requests"], base["pageviews"], 1)
    base["impressions_per_session"] = safe_div(base["impressions"], base["sessions"], 1)
    base["pageviews_per_session"] = safe_div(base["pageviews"], base["sessions"], 1)
    base_label = "Baseline median"

# =============================
# EXPECTED (BASELINE) FOR LOSS METER
# =============================
expected_revenue = float(baseline_df["revenue"].median()) if not baseline_df.empty else 0.0
actual_revenue = float(t["revenue"])
gap = expected_revenue - actual_revenue  # + means loss

# Also compute expected impressions and expected eCPM (for loss allocation)
expected_impressions = float(baseline_df["impressions"].median()) if not baseline_df.empty else 0.0
expected_ecpm = float(baseline_df["ecpm"].median()) if not baseline_df.empty else 0.0

# =============================
# KPI ENGINE (Median + MAD baseline) — for cards + issues
# =============================
kpi_defs = [
    ("revenue", "Revenue", ""),
    ("impressions", "Impressions", ""),
    ("ad_requests", "Ad Requests", ""),
    ("fill_rate", "Fill Rate", "%"),
    ("ecpm", "eCPM", ""),
    ("rpm", "RPM", ""),
    ("requests_per_pageview", "Req/Pageview", ""),
    ("impressions_per_session", "Imp/Session", ""),
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
# TOP KPI ROW
# =============================
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
r1c1.markdown(kpi_card("Revenue", kpi["revenue"]["today"], kpi["revenue"]["delta"], kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)
r1c2.markdown(kpi_card("Impressions", kpi["impressions"]["today"], kpi["impressions"]["delta"], kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)
r1c3.markdown(kpi_card("Ad Requests", kpi["ad_requests"]["today"], kpi["ad_requests"]["delta"], kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)
r1c4.markdown(kpi_card("Fill Rate", kpi["fill_rate"]["today"], kpi["fill_rate"]["delta"], kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"], "%"), unsafe_allow_html=True)
r1c5.markdown(kpi_card("eCPM", kpi["ecpm"]["today"], kpi["ecpm"]["delta"], kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

# =============================
# LOSS METER ROW (PowerBI-style)
# =============================
m1, m2, m3, m4, m5 = st.columns(5)
m1.markdown(loss_meter_card(expected_revenue, actual_revenue), unsafe_allow_html=True)
m2.markdown(kpi_card("RPM", kpi["rpm"]["today"], kpi["rpm"]["delta"], kpi["rpm"]["z"], kpi["rpm"]["conf"]), unsafe_allow_html=True)
m3.markdown(kpi_card("CTR", kpi["ctr"]["today"], kpi["ctr"]["delta"], kpi["ctr"]["z"], kpi["ctr"]["conf"], "%"), unsafe_allow_html=True)
m4.markdown(kpi_card("Req/Pageview", kpi["requests_per_pageview"]["today"], kpi["requests_per_pageview"]["delta"], kpi["requests_per_pageview"]["z"], kpi["requests_per_pageview"]["conf"]), unsafe_allow_html=True)
m5.markdown(kpi_card("Imp/Session", kpi["impressions_per_session"]["today"], kpi["impressions_per_session"]["delta"], kpi["impressions_per_session"]["z"], kpi["impressions_per_session"]["conf"]), unsafe_allow_html=True)

# =============================
# ROOT CAUSE (DECREMENT MATH) + WATERFALLS
# =============================
st.subheader("🧠 Root Cause (metric decrement math)")

left, right = st.columns([2, 1])

with right:
    st.subheader("🎯 Expected vs Actual")
    st.metric("Expected (baseline median)", f"{expected_revenue:,.2f}")
    st.metric("Actual (today)", f"{actual_revenue:,.2f}",
              delta=f"{pct_change(actual_revenue, expected_revenue):+.2f}%" if expected_revenue else None)
    st.metric("Gap", f"{gap:,.2f}",
              delta=f"{safe_div(gap, expected_revenue, 100):+.1f}%" if expected_revenue else None)
    st.caption(f"Baseline: last **{baseline_days}** days (excluding today). Compare: **{base_label}**.")

with left:
    # Compare-based decomposition (Today vs base_label)
    rev_b = float(base.get("revenue", 0.0))
    rev_t = float(t.get("revenue", 0.0))
    d_rev = rev_t - rev_b

    imp_b = float(base.get("impressions", 0.0))
    imp_t = float(t.get("impressions", 0.0))

    ecpm_b = safe_div(rev_b, imp_b, 1000)
    ecpm_t = safe_div(rev_t, imp_t, 1000)

    # Revenue ≈ Impressions * eCPM / 1000
    imp_effect = (imp_t - imp_b) * (ecpm_b / 1000)
    ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000)
    residual = d_rev - (imp_effect + ecpm_effect)

    wf_rev = pd.DataFrame({
        "Step": [f"{base_label} Revenue", "Impressions effect", "eCPM effect", "Residual", "Today Revenue"],
        "Value": [rev_b, imp_effect, ecpm_effect, residual, rev_t]
    })
    st.altair_chart(waterfall_like(wf_rev, "Revenue Decomposition (vs compare)"), use_container_width=True)

    # Impressions ≈ Requests * FillRate/100
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
    st.altair_chart(waterfall_like(wf_imp, "Impressions Decomposition (vs compare)"), use_container_width=True)

    # -----------------------------
    # ROOT CAUSE SUMMARY (minimal)
    # -----------------------------
    st.markdown("### ✅ Root Cause Summary")

    direction = "down" if d_rev < 0 else "up"
    st.write(f"• Revenue moved **{direction} {abs(d_rev):,.2f}** vs **{base_label}**.")

    denom = abs(d_rev) if abs(d_rev) > 1e-9 else 1.0
    imp_share = abs(imp_effect) / denom * 100
    ecpm_share = abs(ecpm_effect) / denom * 100
    st.write(f"• Approx split: **{imp_share:.0f}% Impressions**, **{ecpm_share:.0f}% eCPM**.")

    # Identify main driver
    contribs = [("Impressions", imp_effect), ("eCPM", ecpm_effect), ("Residual", residual)]
    top_driver, top_val = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[0]

    if top_driver == "Impressions":
        st.write(f"• Primary driver: **Impressions** (impact {imp_effect:,.2f}).")
        inner = [("Requests", req_effect), ("Fill", fill_effect)]
        inner_driver, inner_val = sorted(inner, key=lambda x: abs(x[1]), reverse=True)[0]
        if inner_driver == "Requests":
            st.write(f"• Impressions moved mainly due to **Requests** (impact {req_effect:,.0f} impressions).")
            pv_b = float(base.get("pageviews", 0.0))
            pv_t = float(t.get("pageviews", 0.0))
            rpp_b = safe_div(req_b, pv_b, 1)
            rpp_t = safe_div(req_t, pv_t, 1)
            if rpp_t < rpp_b * 0.9:
                st.write("• Pattern matches **Ad loading issue** (requests/pageview decreased).")
            else:
                st.write("• Pattern matches **Traffic/engagement change** (requests/pageview stable).")
        else:
            st.write(f"• Impressions moved mainly due to **Fill** (impact {fill_effect:,.0f} impressions).")
            st.write("• Pattern matches **Demand/blocks/floors/policy** type issue (requests exist, not filled).")

    elif top_driver == "eCPM":
        st.write(f"• Primary driver: **eCPM** (impact {ecpm_effect:,.2f}).")
        st.write("• Next step to pinpoint: add segmentation (country/device/ad unit) to see where value dropped.")
    else:
        st.write("• Residual unusually large → likely data mismatch (lag/merge issues) or both drivers moved sharply.")

    # CTR diagnostic flag only (no long text)
    ctr_b = safe_div(float(base.get("clicks", 0.0)), imp_b, 100) if imp_b else 0.0
    ctr_t = safe_div(float(t.get("clicks", 0.0)), imp_t, 100) if imp_t else 0.0
    if ctr_b > 0 and ctr_t > ctr_b * 1.5:
        st.write("• CTR spike flag (diagnostic): layout/ad mix/policy-risk possibility.")
    elif ctr_b > 0 and ctr_t < ctr_b * 0.6:
        st.write("• CTR drop flag (diagnostic): engagement/ad mix changed.")

    # -----------------------------
    # LOSS ALLOCATION (Expected → Today)
    # -----------------------------
    st.markdown("### 💸 Loss Allocation (Expected → Today)")
    if expected_revenue > 0 and expected_impressions > 0:
        # Allocate expected→today gap into impressions vs eCPM (using expected as base)
        # ΔRev ≈ (ΔImp * expected_ecpm/1000) + (todayImp * ΔeCPM/1000)
        # Convert to positive "loss contributions"
        imp_contrib = (imp_t - expected_impressions) * (expected_ecpm / 1000)
        ecpm_contrib = imp_t * ((ecpm_t - expected_ecpm) / 1000)

        loss_imp = max(-imp_contrib, 0.0)
        loss_ecpm = max(-ecpm_contrib, 0.0)

        total_loss = max(expected_revenue - actual_revenue, 0.0)
        denom2 = (loss_imp + loss_ecpm) if (loss_imp + loss_ecpm) > 0 else 1.0
        share_imp2 = loss_imp / denom2 * 100
        share_ecpm2 = loss_ecpm / denom2 * 100

        st.write(f"• Estimated loss today (vs expected): **{total_loss:,.2f}**")
        st.write(f"• Loss split: **{share_imp2:.0f}% from Impressions**, **{share_ecpm2:.0f}% from eCPM**")
    else:
        st.write("• Not enough baseline data to allocate expected-loss into Impressions vs eCPM.")

# =============================
# TRENDS
# =============================
st.subheader("📈 Trends (baseline window → today)")

t1, t2 = st.columns(2)
with t1:
    st.altair_chart(line_chart(display_df, ["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
with t2:
    st.altair_chart(line_chart(display_df, ["ad_requests", "fill_rate", "impressions"], "Requests → Fill → Impressions"), use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    st.altair_chart(line_chart(display_df, ["sessions", "pageviews"], "Traffic (Sessions + Pageviews)"), use_container_width=True)
with t4:
    st.altair_chart(line_chart(display_df, ["requests_per_pageview", "impressions_per_session", "pageviews_per_session"],
                               "Ad Loading & Engagement Density"), use_container_width=True)

# =============================
# TOP ISSUES TABLE
# =============================
st.subheader("🧯 Top Issues (ranked by drop + anomaly)")

metrics_for_table = [
    ("revenue", "Revenue"),
    ("sessions", "Sessions"),
    ("pageviews", "Pageviews"),
    ("pageviews_per_session", "Pageviews/Session"),
    ("ad_requests", "Ad Requests"),
    ("requests_per_pageview", "Requests/Pageview"),
    ("fill_rate", "Fill Rate"),
    ("impressions", "Impressions"),
    ("impressions_per_session", "Impressions/Session"),
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
        "_score": score,
    })

issues = pd.DataFrame(rows).sort_values("_score", ascending=False).drop(columns=["_score"]).head(12)
st.dataframe(issues, use_container_width=True)

# =============================
# SANITY CHECKS (DATA INTEGRITY)
# =============================
st.subheader("🧪 Data Sanity Checks")

checks = []
if t["sessions"] > 0 and t["ad_requests"] == 0:
    checks.append("Sessions > 0 but Ad Requests = 0 → tagging/ad call issue or missing GAM data.")
if t["ad_requests"] > 0 and t["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed or reporting lag.")
if t["impressions"] > 0 and t["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue reporting lag or merge mismatch.")
if t["pageviews"] > 0 and t["sessions"] == 0:
    checks.append("Pageviews > 0 but Sessions = 0 → GA4 export mismatch.")

if not checks:
    st.success("No obvious integrity red flags for selected day.")
else:
    for c in checks:
        st.warning(c)

st.caption(
    "Ratios are derived from daily totals (never summed). Root cause follows: Revenue → (Impressions + eCPM), "
    "Impressions → (Requests + Fill). Loss meter compares Actual vs Expected (baseline median)."
)
