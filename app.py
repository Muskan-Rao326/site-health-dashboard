# app.py — Revenue Intelligence Dashboard (Story-fixed)
# Fixes:
# ✅ One story reference: Expected baseline median used everywhere
# ✅ Adds story confidence using residual ratio (prevents false certainty)
# ✅ Revenue gap split uses only LOSS contributions (no misleading abs shares)
# ✅ Compare-to-yesterday kept only as context (not mixing narratives)

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
st.markdown('<div class="sub">Story-first root cause (GA4 + GAM) • single-site • consistent baseline</div>', unsafe_allow_html=True)

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
        <div class="kpi-meta">Δ vs Yesterday: {delta_pct:+.2f}%</div>
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

daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived ratios from totals (correct)
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

# =============================
# STORY BASE: EXPECTED (BASELINE MEDIAN TOTALS)
# =============================
# Expected totals
expected_totals = baseline_df[base_cols].median(numeric_only=True).to_dict() if not baseline_df.empty else {c: 0.0 for c in base_cols}

exp = dict(expected_totals)
# Derive expected ratios from expected totals (consistent)
exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
exp["ctr"] = safe_div(exp["clicks"], exp["impressions"], 100)
exp["rpm"] = safe_div(exp["revenue"], exp["pageviews"], 1000)
exp["requests_per_pageview"] = safe_div(exp["ad_requests"], exp["pageviews"], 1)
exp["impressions_per_session"] = safe_div(exp["impressions"], exp["sessions"], 1)
exp["pageviews_per_session"] = safe_div(exp["pageviews"], exp["sessions"], 1)

expected_revenue = float(exp["revenue"])
actual_revenue = float(t["revenue"])
gap = expected_revenue - actual_revenue  # + means loss

# =============================
# KPI ENGINE (Robust baseline) — Z score is still baseline-based
# KPI cards show Δ vs Yesterday only (context), NOT used for story logic
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
    val_y = float(y.get(key, 0.0)) if y is not None else 0.0
    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta_y = pct_change(val_today, val_y) if y is not None else 0.0
    kpi[key] = dict(label=label, suffix=suffix, today=val_today, yesterday=val_y, med=med, mad=mad, z=z, conf=conf, delta_y=delta_y)

# =============================
# KPI ROWS
# =============================
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
r1c1.markdown(kpi_card("Revenue", kpi["revenue"]["today"], kpi["revenue"]["delta_y"], kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)
r1c2.markdown(kpi_card("Impressions", kpi["impressions"]["today"], kpi["impressions"]["delta_y"], kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)
r1c3.markdown(kpi_card("Ad Requests", kpi["ad_requests"]["today"], kpi["ad_requests"]["delta_y"], kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)
r1c4.markdown(kpi_card("Fill Rate", kpi["fill_rate"]["today"], kpi["fill_rate"]["delta_y"], kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"], "%"), unsafe_allow_html=True)
r1c5.markdown(kpi_card("eCPM", kpi["ecpm"]["today"], kpi["ecpm"]["delta_y"], kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.markdown(loss_meter_card(expected_revenue, actual_revenue), unsafe_allow_html=True)
m2.markdown(kpi_card("RPM", kpi["rpm"]["today"], kpi["rpm"]["delta_y"], kpi["rpm"]["z"], kpi["rpm"]["conf"]), unsafe_allow_html=True)
m3.markdown(kpi_card("CTR", kpi["ctr"]["today"], kpi["ctr"]["delta_y"], kpi["ctr"]["z"], kpi["ctr"]["conf"], "%"), unsafe_allow_html=True)
m4.markdown(kpi_card("Req/Pageview", kpi["requests_per_pageview"]["today"], kpi["requests_per_pageview"]["delta_y"], kpi["requests_per_pageview"]["z"], kpi["requests_per_pageview"]["conf"]), unsafe_allow_html=True)
m5.markdown(kpi_card("Imp/Session", kpi["impressions_per_session"]["today"], kpi["impressions_per_session"]["delta_y"], kpi["impressions_per_session"]["z"], kpi["impressions_per_session"]["conf"]), unsafe_allow_html=True)

# =============================
# ROOT CAUSE — ALWAYS VS EXPECTED (STORY CONSISTENT)
# =============================
st.subheader("🧠 Root Cause Story (vs Expected baseline median)")

left, right = st.columns([2, 1])

with right:
    st.subheader("🎯 Expected vs Actual (the story reference)")
    st.metric("Expected (baseline median)", f"{expected_revenue:,.2f}")
    st.metric("Actual (today)", f"{actual_revenue:,.2f}",
              delta=f"{pct_change(actual_revenue, expected_revenue):+.2f}%" if expected_revenue else None)
    st.metric("Gap (Expected − Actual)", f"{gap:,.2f}",
              delta=f"{safe_div(gap, expected_revenue, 100):+.1f}%" if expected_revenue else None)

    # Yesterday is context only
    if y is not None:
        st.caption(f"Context only: Revenue vs Yesterday = {pct_change(actual_revenue, float(y['revenue'])):+.2f}%")
    st.caption(f"Baseline window: last **{baseline_days}** days (excluding today).")

with left:
    # --- Revenue decomposition vs expected ---
    rev_b = float(exp["revenue"])
    rev_t = float(t["revenue"])
    d_rev = rev_t - rev_b

    imp_b = float(exp["impressions"])
    imp_t = float(t["impressions"])

    ecpm_b = safe_div(rev_b, imp_b, 1000)
    ecpm_t = safe_div(rev_t, imp_t, 1000)

    # Revenue ≈ Impressions * eCPM / 1000
    imp_effect = (imp_t - imp_b) * (ecpm_b / 1000)
    ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000)
    residual = d_rev - (imp_effect + ecpm_effect)

    wf_rev = pd.DataFrame({
        "Step": ["Expected Revenue", "Impressions effect", "eCPM effect", "Residual", "Today Revenue"],
        "Value": [rev_b, imp_effect, ecpm_effect, residual, rev_t]
    })
    st.altair_chart(waterfall_like(wf_rev, "Revenue Decomposition (Expected → Today)"), use_container_width=True)

    # --- Impressions decomposition vs expected ---
    req_b = float(exp["ad_requests"])
    req_t = float(t["ad_requests"])

    fill_b = safe_div(imp_b, req_b, 100) if req_b else 0.0
    fill_t = safe_div(imp_t, req_t, 100) if req_t else 0.0

    d_imp = imp_t - imp_b
    req_effect = (req_t - req_b) * (fill_b / 100)
    fill_effect = req_t * ((fill_t - fill_b) / 100)
    imp_residual = d_imp - (req_effect + fill_effect)

    wf_imp = pd.DataFrame({
        "Step": ["Expected Impressions", "Requests effect", "Fill effect", "Residual", "Today Impressions"],
        "Value": [imp_b, req_effect, fill_effect, imp_residual, imp_t]
    })
    st.altair_chart(waterfall_like(wf_imp, "Impressions Decomposition (Expected → Today)"), use_container_width=True)

    # =============================
    # STORY CONFIDENCE (RESIDUAL RATIO)
    # =============================
    st.markdown("### 🎚️ Story Confidence")
    denom = abs(d_rev) if abs(d_rev) > 1e-9 else max(abs(rev_b), 1.0)
    residual_ratio = abs(residual) / denom  # how much of change is unexplained

    if residual_ratio <= 0.10:
        story_conf = "High"
    elif residual_ratio <= 0.25:
        story_conf = "Medium"
    else:
        story_conf = "Low"

    st.write(f"• Residual ratio: **{residual_ratio:.2f}** → Confidence: **{story_conf}**")
    if story_conf == "Low":
        st.warning("Residual is large. This often means revenue/impressions reporting lag or merged data mismatch. Treat the story as directional for this day.")

    # =============================
    # ROOT CAUSE SUMMARY (LOSS-FIRST, NOT BUZZ)
    # Uses only LOSS contributions when gap exists
    # =============================
    st.markdown("### ✅ Root Cause Summary (money story)")

    if expected_revenue <= 0:
        st.write("• Not enough baseline revenue to build a story.")
    else:
        if gap > 0:
            st.write(f"• Today missed expected revenue by **{gap:,.2f}**.")

            # Convert effects into "loss contributions" (only count negative contributors when revenue dropped vs expected)
            # If d_rev is negative, then losses are positive numbers from negative effects.
            loss_from_imp = max(-imp_effect, 0.0)
            loss_from_ecpm = max(-ecpm_effect, 0.0)
            loss_unexplained = max(-residual, 0.0)

            denom_loss = (loss_from_imp + loss_from_ecpm + loss_unexplained) if (loss_from_imp + loss_from_ecpm + loss_unexplained) > 0 else 1.0

            s_imp = loss_from_imp / denom_loss * 100
            s_ecpm = loss_from_ecpm / denom_loss * 100
            s_res = loss_unexplained / denom_loss * 100

            st.write(f"• Loss split (Expected → Today): **{s_imp:.0f}% from Impressions**, **{s_ecpm:.0f}% from eCPM**, **{s_res:.0f}% unexplained (residual)**.")

            # Primary + Secondary driver (based on loss parts)
            parts = [("Impressions", loss_from_imp), ("eCPM", loss_from_ecpm), ("Residual", loss_unexplained)]
            parts_sorted = sorted(parts, key=lambda x: x[1], reverse=True)

            primary, primary_val = parts_sorted[0]
            secondary, secondary_val = parts_sorted[1]

            st.write(f"• Primary driver: **{primary}** (≈ {primary_val:,.2f} of the loss).")
            if secondary_val > 0:
                st.write(f"• Secondary driver: **{secondary}** (≈ {secondary_val:,.2f}).")

            # If impressions is the driver, explain whether it is Requests or Fill
            if primary == "Impressions":
                # compute impression loss split
                imp_loss_from_req = max(-req_effect * (ecpm_b / 1000), 0.0)  # converted to revenue-equivalent for readability
                imp_loss_from_fill = max(-fill_effect * (ecpm_b / 1000), 0.0)
                denom_imp_loss = (imp_loss_from_req + imp_loss_from_fill) if (imp_loss_from_req + imp_loss_from_fill) > 0 else 1.0

                st.write(f"• Inside Impressions: **{imp_loss_from_req/denom_imp_loss*100:.0f}% Requests**, **{imp_loss_from_fill/denom_imp_loss*100:.0f}% Fill** (revenue-equivalent).")

                # Improved “ad loading vs traffic mix” heuristic
                pv_b = float(exp["pageviews"])
                pv_t = float(t["pageviews"])
                rpp_b = safe_div(req_b, pv_b, 1)
                rpp_t = safe_div(req_t, pv_t, 1)

                pageviews_stable = (pv_b > 0 and pv_t > pv_b * 0.95)
                rpp_down = (rpp_b > 0 and rpp_t < rpp_b * 0.90)

                if pageviews_stable and rpp_down:
                    st.write("• Pattern: **Pageviews stable but Requests/Pageview down** → likely **ad loading / tag / CMP / adblock / template issue**.")
                else:
                    st.write("• Pattern: Requests aligned with pageview changes → likely **traffic mix / engagement / page template mix change**.")

            if primary == "eCPM":
                st.write("• eCPM drop means buyers paid less per 1,000 impressions. Next step: add segmentation (country/device/ad unit) to pinpoint where value fell.")

        else:
            st.write(f"• Today is at or above expected revenue (Gap: {gap:,.2f}). No loss story required.")

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
    "Story reference is always Expected baseline median. Yesterday is context only. "
    "Ratios derived from totals. Revenue → (Impressions + eCPM). Impressions → (Requests + Fill). "
    "Story confidence decreases when residual is large."
)
