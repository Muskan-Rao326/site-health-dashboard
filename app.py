import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Revenue Leak Detective", layout="wide")

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(a, b):
    if b == 0:
        return 0.0 if a == 0 else 999.0
    return (a - b) / b * 100

def line_chart(df, cols, title, height=260):
    melted = df[["date"] + cols].melt("date", var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=""),
        color=alt.Color("metric:N", title=""),
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=",.4f")]
    ).properties(title=title, height=height)

def story_confidence(residual_ratio: float):
    if residual_ratio <= 0.10:
        return "High"
    if residual_ratio <= 0.25:
        return "Medium"
    return "Low"

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

# =============================
# UI HEADER
# =============================
st.title("🕵️ Revenue Leak Detective")
st.caption("Single site • clean math • one consistent story (Expected → Today)")

# =============================
# LOAD
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_csv(uploaded)

# Site picker (or single site)
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    site = st.sidebar.selectbox("Site", sites, index=0)
    df_raw = df_raw[df_raw["site_name"] == site].copy()
else:
    site = "Single Site"

st.sidebar.header("Controls")

# Required base totals
required = ["revenue", "impressions", "ad_requests", "clicks", "sessions", "pageviews"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# Clean numeric
for c in required:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily totals
daily = df_raw.groupby("date", as_index=False)[required].sum().sort_values("date")

# Recompute ratios from totals (ignore any precomputed columns)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)          # $ per 1000 imp
daily["fill"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 1.0), axis=1)     # fraction
daily["ctr"]  = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 1.0), axis=1)          # fraction
daily["req_per_pv"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1.0), axis=1)

# Date controls
selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date()
)
baseline_days = st.sidebar.slider("Baseline window (days)", 7, 30, 7)

baseline_mode = st.sidebar.selectbox(
    "Baseline mode",
    ["Rolling median", "Same weekday median"],
    index=1
)

today = pd.to_datetime(selected_date).normalize()
yesterday = today - timedelta(days=1)

Trow = daily[daily["date"] == today]
if Trow.empty:
    st.error("No data for selected date.")
    st.stop()
T = Trow.iloc[0].to_dict()

Yrow = daily[daily["date"] == yesterday]
Y = Yrow.iloc[0].to_dict() if not Yrow.empty else None

baseline_start = today - timedelta(days=baseline_days)
base = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)].copy()
if base.empty:
    st.error("Not enough baseline data in the selected window.")
    st.stop()

if baseline_mode == "Same weekday median":
    dow = today.day_name()
    base = base[base["date"].dt.day_name() == dow].copy()
    if base.empty:
        st.error("No same-weekday baseline data in the chosen window. Switch to Rolling median or expand window.")
        st.stop()

# =============================
# EXPECTED = MEDIAN OF TOTALS (CONSISTENT BASE)
# =============================
E = {c: float(base[c].median()) for c in required}

# Derive expected ratios from expected totals (consistent)
E_ecpm = safe_div(E["revenue"], E["impressions"], 1000)
E_fill = safe_div(E["impressions"], E["ad_requests"], 1.0)
E_ctr  = safe_div(E["clicks"], E["impressions"], 1.0)
E_rpp  = safe_div(E["ad_requests"], E["pageviews"], 1.0)

# Today ratios
T_ecpm = float(T["ecpm"])
T_fill = float(T["fill"])
T_ctr  = float(T["ctr"])
T_rpp  = float(T["req_per_pv"])

# =============================
# CORE GAP (THE ONLY STORY)
# =============================
expected_rev = float(E["revenue"])
actual_rev   = float(T["revenue"])
gap = expected_rev - actual_rev  # + means loss

# =============================
# CLEAN DECOMPOSITION (SYMMETRIC / STABLE)
# Revenue = Imp * eCPM / 1000
# =============================
Imp_t = float(T["impressions"])
Imp_e = float(E["impressions"])

# Symmetric split (stable):
rev_imp_effect  = 0.5 * (Imp_t - Imp_e) * ((T_ecpm + E_ecpm) / 1000.0)
rev_ecpm_effect = 0.5 * (T_ecpm - E_ecpm) * ((Imp_t + Imp_e) / 1000.0)

# Model delta
delta_rev_model = actual_rev - (Imp_e * E_ecpm / 1000.0)
rev_residual = delta_rev_model - (rev_imp_effect + rev_ecpm_effect)

# Residual ratio = how much of movement is unexplained
denom = max(abs(delta_rev_model), 1.0)
residual_ratio = abs(rev_residual) / denom
conf = story_confidence(residual_ratio)

# =============================
# SECOND LEVEL: Impressions = Requests * Fill
# =============================
Req_t = float(T["ad_requests"])
Req_e = float(E["ad_requests"])

imp_req_effect  = 0.5 * (Req_t - Req_e) * (T_fill + E_fill)
imp_fill_effect = 0.5 * (T_fill - E_fill) * (Req_t + Req_e)

imp_model = Req_e * E_fill
imp_residual = (Imp_t - imp_model) - (imp_req_effect + imp_fill_effect)

# =============================
# SIMPLE ROOT CAUSE DECISION (NO BUZZ)
# =============================
def root_cause():
    if expected_rev <= 0:
        return ("No baseline", "Not enough baseline to diagnose.")

    if gap <= 0:
        return ("Healthy", "Revenue is at/above expected today.")

    # Loss contributions (only count negative contributions as loss)
    loss_imp  = max(-rev_imp_effect, 0.0)
    loss_ecpm = max(-rev_ecpm_effect, 0.0)
    loss_res  = max(-rev_residual, 0.0)

    # Primary driver
    parts = sorted([("Impressions", loss_imp), ("eCPM", loss_ecpm), ("Residual", loss_res)],
                   key=lambda x: x[1], reverse=True)
    primary, pval = parts[0]

    # Explain based on chain
    if primary == "Residual" and conf == "Low":
        return ("Data not reliable today",
                "A large part of the change is unexplained (likely GAM revenue lag or merge mismatch). Recheck tomorrow or extend baseline.")

    if primary == "eCPM":
        return ("Auction value drop (eCPM)",
                "Impressions did not fall enough to explain the loss. Buyers paid less per 1,000 impressions.")

    # primary impressions
    # decide inside impressions: requests or fill
    # convert impression-effects into revenue-equivalent using expected ecpm
    req_rev_equiv  = max(-(imp_req_effect  * (E_ecpm / 1000.0)), 0.0)
    fill_rev_equiv = max(-(imp_fill_effect * (E_ecpm / 1000.0)), 0.0)

    if fill_rev_equiv >= req_rev_equiv:
        return ("Demand / fill issue",
                "Requests exist but fewer got filled. Look at floors, blocks, policy, demand partner downtime.")
    else:
        # request-driven: ad loading vs traffic/page-mix
        pv_t = float(T["pageviews"])
        pv_e = float(E["pageviews"])
        pageviews_stable = pv_e > 0 and pv_t > pv_e * 0.95
        rpp_down = E_rpp > 0 and T_rpp < E_rpp * 0.90

        if pageviews_stable and rpp_down:
            return ("Ad loading / tag / template issue",
                    "Pageviews are stable but requests per pageview dropped. Ads likely not calling fully.")
        return ("Traffic / page-mix change",
                "Requests dropped mainly because pageviews or page mix changed (fewer ad opportunities).")

root_title, root_msg = root_cause()

# =============================
# OVERVIEW (CLEAN)
# =============================
top = st.container()
with top:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Revenue", f"{actual_rev:,.2f}", delta=(f"{pct_change(actual_rev, float(Y['revenue'])):+.2f}%" if Y else None))
    c2.metric("Impressions", f"{Imp_t:,.0f}", delta=(f"{pct_change(Imp_t, float(Y['impressions'])):+.2f}%" if Y else None))
    c3.metric("Ad Requests", f"{Req_t:,.0f}", delta=(f"{pct_change(Req_t, float(Y['ad_requests'])):+.2f}%" if Y else None))
    c4.metric("Fill Rate", f"{T_fill*100:,.2f}%", delta=(f"{pct_change(T_fill, float(Y['fill'])):+.2f}%" if Y else None))
    c5.metric("eCPM", f"{T_ecpm:,.2f}", delta=(f"{pct_change(T_ecpm, float(Y['ecpm'])):+.2f}%" if Y else None))

st.divider()

# =============================
# STORY PANEL
# =============================
s1, s2, s3 = st.columns([1.2, 1.4, 1.4])

with s1:
    st.subheader("Today vs Expected")
    st.metric("Expected", f"{expected_rev:,.2f}")
    st.metric("Actual", f"{actual_rev:,.2f}")
    st.metric("Gap (loss if +)", f"{gap:,.2f}", delta=f"{safe_div(gap, expected_rev, 100):+.1f}%" if expected_rev else None)

with s2:
    st.subheader("Root Cause (one line)")
    if root_title in ["Healthy"]:
        st.success(f"**{root_title}**")
    elif root_title in ["Data not reliable today"]:
        st.warning(f"**{root_title}**")
    else:
        st.error(f"**{root_title}**")
    st.write(root_msg)
    st.caption(f"Story confidence: **{conf}** (residual ratio {residual_ratio:.2f})")

with s3:
    st.subheader("Where to look (checklist)")
    if root_title == "Ad loading / tag / template issue":
        st.markdown("- Compare **requests/pageview** by template\n- Tag / header bidding errors\n- Consent changes / CMP\n- Adblock impact / script blocked\n- New layout removing slots")
    elif root_title == "Demand / fill issue":
        st.markdown("- Floor price change\n- Blocking rules / protections\n- Demand partner outage\n- Policy limitation\n- Geo mix moved to weak demand")
    elif root_title == "Auction value drop (eCPM)":
        st.markdown("- Country/device eCPM drop\n- Ad unit mix changed\n- Fewer premium formats\n- Brand-safety / IVT flags\n- Seasonality (same weekday baseline helps)")
    elif root_title == "Data not reliable today":
        st.markdown("- GAM revenue lag (same-day)\n- Merge mismatch (date boundaries)\n- Missing rows for today\n- Retry tomorrow / extend window")
    else:
        st.markdown("- Check traffic quality\n- Check ad loading density\n- Check demand delivery")

# =============================
# SIMPLE “WHY” SPLIT (MONEY)
# =============================
st.subheader("Why did revenue miss expected? (money split)")

loss_imp  = max(-rev_imp_effect, 0.0)
loss_ecpm = max(-rev_ecpm_effect, 0.0)
loss_res  = max(-rev_residual, 0.0)

den = (loss_imp + loss_ecpm + loss_res) if (loss_imp + loss_ecpm + loss_res) > 0 else 1.0
st.write(
    f"• Impressions-driven loss: **{loss_imp:,.2f}** ({loss_imp/den*100:.0f}%)  \n"
    f"• eCPM-driven loss: **{loss_ecpm:,.2f}** ({loss_ecpm/den*100:.0f}%)  \n"
    f"• Unexplained (residual): **{loss_res:,.2f}** ({loss_res/den*100:.0f}%)"
)

# =============================
# TRENDS (ONLY THE ONES THAT MATTER)
# =============================
st.subheader("Trends (baseline window → today)")
trend_df = daily[(daily["date"] >= (today - timedelta(days=baseline_days+1))) & (daily["date"] <= today)].copy()

t1, t2 = st.columns(2)
with t1:
    st.altair_chart(line_chart(trend_df, ["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
with t2:
    # show fill as % for readability
    tmp = trend_df.copy()
    tmp["fill_pct"] = tmp["fill"] * 100
    st.altair_chart(line_chart(tmp, ["ad_requests", "fill_pct", "impressions"], "Requests → Fill → Impressions"), use_container_width=True)

# =============================
# DATA TRUST CHECKS
# =============================
st.subheader("Data trust checks")
checks = []
if T["sessions"] > 0 and T["ad_requests"] == 0:
    checks.append("Sessions > 0 but Ad Requests = 0 → ad tags not firing OR GAM export missing.")
if T["ad_requests"] > 0 and T["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed OR reporting lag.")
if T["impressions"] > 0 and T["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue lag / mismatch.")

if conf == "Low":
    checks.append("Low story confidence due to high residual ratio → treat root cause as directional only.")

if not checks:
    st.success("No obvious red flags for this day.")
else:
    for c in checks:
        st.warning(c)

st.caption("All ratios are recomputed from totals. Baseline is median of totals (consistent). Root cause is Expected → Today only.")
