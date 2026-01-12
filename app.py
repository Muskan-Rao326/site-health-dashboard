import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Revenue Intelligence", layout="wide")

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

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

def conf_from_z(z):
    # two-sided
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def fmt_money(x):
    return f"{x:,.2f}"

def fmt_pct(x):
    return f"{x:,.2f}%"

def pct_change(today, prev):
    if prev == 0:
        return 0.0 if today == 0 else 999.0
    return (today - prev) / prev * 100

def line_chart(df, cols, title, height=260):
    melted = df[["date"] + cols].melt("date", var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=""),
        color=alt.Color("metric:N", title=""),
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=",.4f")]
    ).properties(title=title, height=height)

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

# =============================
# TITLE
# =============================
st.title("📊 Revenue Intelligence Dashboard")
st.caption("Single-site • GA4 + GAM • Clean math • Root-cause tree • Baseline-aware")

# =============================
# UPLOAD
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# =============================
# SINGLE SITE FILTER
# =============================
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    site = st.sidebar.selectbox("Site", sites, index=0)
    df_raw = df_raw[df_raw["site_name"] == site].copy()
else:
    site = "Single Site"
st.sidebar.caption(f"Using: **{site}**")

# =============================
# REQUIRED BASE METRICS (ONLY ADDITIVE)
# =============================
base_cols = ["revenue", "ad_requests", "impressions", "clicks", "sessions", "users", "pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived ratios (always from totals)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["req_per_pv"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["imp_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)
daily["pv_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)
daily["dow"] = daily["date"].dt.day_name()

# =============================
# CONTROLS
# =============================
st.sidebar.header("Controls")
selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date()
)
baseline_days = st.sidebar.slider("Baseline window (days)", 7, 30, 7)

baseline_mode = st.sidebar.selectbox(
    "Baseline mode",
    ["Rolling median (last N days)", "Same weekday median (reduces weekly seasonality)"],
    index=1
)

today = pd.to_datetime(selected_date).normalize()
yesterday = today - timedelta(days=1)

today_row = daily[daily["date"] == today]
if today_row.empty:
    st.error("No data for selected date.")
    st.stop()
T = today_row.iloc[0].to_dict()

Y_row = daily[daily["date"] == yesterday]
Y = Y_row.iloc[0].to_dict() if not Y_row.empty else None

baseline_start = today - timedelta(days=baseline_days)
baseline_df = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)].copy()

if baseline_mode.startswith("Same weekday"):
    dow = pd.to_datetime(today).day_name()
    baseline_df = baseline_df[baseline_df["dow"] == dow].copy()

display_df = daily[(daily["date"] >= baseline_start - timedelta(days=1)) & (daily["date"] <= today)].copy()

# =============================
# EXPECTED: TWO VERSIONS (IMPORTANT)
# =============================
# A) Finance expectation: median of revenue directly
expected_rev_fin = float(baseline_df["revenue"].median()) if not baseline_df.empty else 0.0

# B) Model-consistent expectation: median(Impressions) and median(eCPM) -> revenue
exp_imp = float(baseline_df["impressions"].median()) if not baseline_df.empty else 0.0
exp_ecpm = float(baseline_df["ecpm"].median()) if not baseline_df.empty else 0.0
expected_rev_model = (exp_imp * exp_ecpm) / 1000.0

actual_rev = float(T["revenue"])
gap_fin = expected_rev_fin - actual_rev
gap_model = expected_rev_model - actual_rev

baseline_inconsistency = expected_rev_fin - expected_rev_model  # should be small; if big -> baseline mix / lag

# =============================
# ROBUST Z ON CORE METRICS (vs baseline)
# =============================
core_for_z = ["revenue", "impressions", "ad_requests", "fill_rate", "ecpm", "rpm", "req_per_pv", "imp_per_session", "ctr"]
z = {}
for m in core_for_z:
    med, mad = median_mad(baseline_df[m]) if (not baseline_df.empty and m in baseline_df.columns) else (0.0, 0.0)
    zv = robust_z(float(T.get(m, 0.0)), med, mad)
    z[m] = {"z": zv, "conf": conf_from_z(zv), "med": med, "mad": mad}

# =============================
# SHAPLEY-LIKE DECOMPOSITION (SYMMETRIC, CLEAN)
# Revenue = (Impressions * eCPM) / 1000
# Compare TODAY vs EXPECTED_MODEL (consistent)
# =============================
imp_t = float(T["impressions"])
ecpm_t = float(T["ecpm"])

imp_e = exp_imp
ecpm_e = exp_ecpm

# Symmetric attribution
# ΔR_imp = 0.5*(Imp_t - Imp_e)*(eCPM_t + eCPM_e)/1000
# ΔR_ecpm = 0.5*(eCPM_t - eCPM_e)*(Imp_t + Imp_e)/1000
delta_rev_model = actual_rev - expected_rev_model
rev_imp_effect = 0.5 * (imp_t - imp_e) * ((ecpm_t + ecpm_e) / 1000.0)
rev_ecpm_effect = 0.5 * (ecpm_t - ecpm_e) * ((imp_t + imp_e) / 1000.0)
rev_residual = delta_rev_model - (rev_imp_effect + rev_ecpm_effect)  # should be near 0

# Impressions = Requests * FillRate
req_t = float(T["ad_requests"])
fill_t = float(T["fill_rate"]) / 100.0

req_e = float(baseline_df["ad_requests"].median()) if not baseline_df.empty else 0.0
fill_e = float(baseline_df["fill_rate"].median()) / 100.0 if not baseline_df.empty else 0.0

delta_imp_model = imp_t - (req_e * fill_e)
imp_req_effect = 0.5 * (req_t - req_e) * (fill_t + fill_e)
imp_fill_effect = 0.5 * (fill_t - fill_e) * (req_t + req_e)
imp_residual2 = (imp_t - (req_e * fill_e)) - (imp_req_effect + imp_fill_effect)

# =============================
# ROOT CAUSE TREE (DECISION LOGIC)
# =============================
def classify_root_cause():
    # Use expected_model as reference because it’s consistent with decomposition
    if expected_rev_model <= 0:
        return ("No baseline", "Not enough baseline data to diagnose.")

    if gap_model <= 0:
        return ("Normal / Above expected", "Today is at or above expected. No revenue leak.")

    # revenue leak exists
    # choose dominant driver by loss contribution
    loss_imp = max(-rev_imp_effect, 0.0)
    loss_ecpm = max(-rev_ecpm_effect, 0.0)

    if loss_imp >= loss_ecpm:
        # impressions-driven
        # inside impressions: requests vs fill
        loss_req = max(-(imp_req_effect * (ecpm_e / 1000.0)), 0.0)  # convert to revenue-equivalent for “which to look at”
        loss_fill = max(-(imp_fill_effect * (ecpm_e / 1000.0)), 0.0)

        # heuristic: ad loading vs traffic mix
        pv_t = float(T["pageviews"])
        pv_e = float(baseline_df["pageviews"].median()) if not baseline_df.empty else 0.0
        rpp_t = float(T["req_per_pv"])
        rpp_e = float(baseline_df["req_per_pv"].median()) if not baseline_df.empty else 0.0

        pageviews_stable = (pv_e > 0 and pv_t > pv_e * 0.95)
        rpp_down = (rpp_e > 0 and rpp_t < rpp_e * 0.90)

        if loss_req >= loss_fill:
            if pageviews_stable and rpp_down:
                return ("Ad loading / tag / CMP issue", "Pageviews stable but requests per pageview fell → ad calls not firing fully.")
            return ("Traffic / page-mix issue", "Requests fell mainly because traffic or page type mix changed.")
        else:
            return ("Demand / fill issue", "Requests exist but fewer got filled → buyers/floors/policy/blocks.")

    # ecpm-driven
    return ("Auction value drop (eCPM)", "Impressions near expected but value per 1,000 impressions fell.")

root_title, root_explain = classify_root_cause()

# =============================
# UI
# =============================
tab1, tab2, tab3 = st.tabs(["🧭 Overview", "🧠 Root Cause", "📈 Trends"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Revenue", fmt_money(actual_rev), delta=(fmt_pct(pct_change(actual_rev, float(Y["revenue"]))) if Y else None))
    c2.metric("Impressions", f"{imp_t:,.0f}", delta=(fmt_pct(pct_change(imp_t, float(Y["impressions"]))) if Y else None))
    c3.metric("Ad Requests", f"{req_t:,.0f}", delta=(fmt_pct(pct_change(req_t, float(Y["ad_requests"]))) if Y else None))
    c4.metric("Fill Rate", fmt_pct(float(T["fill_rate"])), delta=(fmt_pct(pct_change(float(T["fill_rate"]), float(Y["fill_rate"]))) if Y else None))
    c5.metric("eCPM", fmt_money(float(T["ecpm"])), delta=(fmt_pct(pct_change(float(T["ecpm"]), float(Y["ecpm"]))) if Y else None))

    st.divider()

    a1, a2, a3 = st.columns([1.2, 1.2, 1.6])

    with a1:
        st.subheader("Today vs Expected")
        st.metric("Expected (Model)", fmt_money(expected_rev_model))
        st.metric("Actual", fmt_money(actual_rev))
        st.metric("Gap (Model)", fmt_money(gap_model), delta=fmt_pct(safe_div(gap_model, expected_rev_model, 100) if expected_rev_model else 0.0))

    with a2:
        st.subheader("Baseline health")
        st.metric("Expected (Finance median)", fmt_money(expected_rev_fin))
        st.metric("Baseline inconsistency", fmt_money(baseline_inconsistency))
        st.caption("If inconsistency is large, baseline medians are unstable (mix/lag). Use Same-weekday baseline or longer window.")

    with a3:
        st.subheader("Root cause (one line)")
        st.success(f"**{root_title}**")
        st.write(root_explain)

        # “Story confidence” based on residual being small
        denom = max(abs(delta_rev_model), 1.0)
        residual_ratio = abs(rev_residual) / denom
        if residual_ratio <= 0.10:
            st.caption(f"Story confidence: **High** (residual ratio {residual_ratio:.2f})")
        elif residual_ratio <= 0.25:
            st.caption(f"Story confidence: **Medium** (residual ratio {residual_ratio:.2f})")
        else:
            st.warning(f"Story confidence: **Low** (residual ratio {residual_ratio:.2f}). Likely lag/mismatch today.")

with tab2:
    st.subheader("Revenue leak attribution (Expected → Today)")

    # Show loss split as money (only if loss)
    if gap_model > 0:
        loss_imp = max(-rev_imp_effect, 0.0)
        loss_ecpm = max(-rev_ecpm_effect, 0.0)
        denom_loss = max(loss_imp + loss_ecpm, 1e-9)
        st.write(
            f"Loss split: **{loss_imp/denom_loss*100:.0f}% from Impressions** "
            f"and **{loss_ecpm/denom_loss*100:.0f}% from eCPM**"
        )

    r1, r2, r3 = st.columns(3)
    r1.metric("Impressions effect", fmt_money(rev_imp_effect))
    r2.metric("eCPM effect", fmt_money(rev_ecpm_effect))
    r3.metric("Residual", fmt_money(rev_residual))

    st.divider()
    st.subheader("Impressions leak attribution (Requests → Fill)")

    i1, i2, i3 = st.columns(3)
    i1.metric("Requests effect (impressions)", f"{imp_req_effect:,.0f}")
    i2.metric("Fill effect (impressions)", f"{imp_fill_effect:,.0f}")
    i3.metric("Residual", f"{imp_residual2:,.0f}")

    st.divider()
    st.subheader("Anomaly signals (robust)")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Revenue Z", f"{z['revenue']['z']:.2f}", delta=f"{z['revenue']['conf']:.1f}% conf")
    s2.metric("Impressions Z", f"{z['impressions']['z']:.2f}", delta=f"{z['impressions']['conf']:.1f}% conf")
    s3.metric("Requests Z", f"{z['ad_requests']['z']:.2f}", delta=f"{z['ad_requests']['conf']:.1f}% conf")
    s4.metric("Fill Z", f"{z['fill_rate']['z']:.2f}", delta=f"{z['fill_rate']['conf']:.1f}% conf")
    s5.metric("eCPM Z", f"{z['ecpm']['z']:.2f}", delta=f"{z['ecpm']['conf']:.1f}% conf")

with tab3:
    st.subheader("Trends")
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(line_chart(display_df, ["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
    with c2:
        st.altair_chart(line_chart(display_df, ["ad_requests", "fill_rate", "impressions"], "Requests → Fill → Impressions"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(line_chart(display_df, ["sessions", "pageviews"], "Traffic"), use_container_width=True)
    with c4:
        st.altair_chart(line_chart(display_df, ["req_per_pv", "imp_per_session", "pv_per_session"], "Density ratios"), use_container_width=True)

# =============================
# DATA SANITY CHECKS
# =============================
st.divider()
st.subheader("🧪 Sanity checks (data trust)")

checks = []
if T["sessions"] > 0 and T["ad_requests"] == 0:
    checks.append("Sessions > 0 but Ad Requests = 0 → tag/ad calls missing OR GAM export missing.")
if T["ad_requests"] > 0 and T["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed OR reporting lag.")
if T["impressions"] > 0 and T["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue lag or mismatch.")
if expected_rev_model > 0 and abs(baseline_inconsistency) / expected_rev_model > 0.15:
    checks.append("Baseline inconsistency is high → baseline medians unstable. Prefer Same-weekday mode or longer window.")

if not checks:
    st.success("No obvious red flags for selected day.")
else:
    for c in checks:
        st.warning(c)

st.caption(
    "Notes: Ratios are derived from totals. Expected(Model) keeps decomposition consistent. "
    "Same-weekday baseline reduces weekday seasonality. Residual ratio indicates story reliability."
)
