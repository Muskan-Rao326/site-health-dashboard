# app.py — Revenue Intelligence Dashboard (Story-first + Metric-aware Status + Impact)
# ✅ Fixes requested:
# 1) Status logic is NOT generic anymore:
#    - "Lower is bad" metrics: revenue, impressions, ad_requests, fill_rate, ecpm, rpm, requests_per_pageview, impressions_per_session
#    - "Two-sided" metrics: CTR (abnormally high OR low is flagged)
# 2) Cards now show TWO chips:
#    - Anomaly (baseline): Good / Watch / Bad (from robust Z)
#    - Impact (money): High / Medium / Low (based on ₹ loss allocation vs Expected)
# 3) UI clearer: cards explicitly say Δ vs Yesterday (context) + baseline anomaly is separate.
# 4) Root-cause story uses Expected baseline median ONLY (no mixed narratives)
# 5) Altair v6-safe waterfall (no alt.condition) — fixes your TypeError.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# PAGE CONFIG + STYLE
# =============================
st.set_page_config(page_title="Revenue Intelligence (Story)", layout="wide")

st.markdown(
    """
<style>
    :root{
        --text:#e5e7eb;
        --muted:#9ca3af;
        --border: rgba(255,255,255,0.10);
    }
    .title { font-size: 26px; font-weight: 900; margin-bottom: 0px; color: var(--text); }
    .sub { font-size: 13px; color: var(--muted); margin-bottom: 14px; }
    .card {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 14px; padding: 14px;
    }
    .kpiTop { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }
    .kpiName { font-size: 12px; color: var(--muted); letter-spacing: .2px; }
    .kpiVal { font-size: 26px; font-weight: 900; color: var(--text); margin-top: 2px; }
    .kpiSub { font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.25; }
    .pill {
        font-size: 11px; padding: 4px 8px; border-radius: 999px;
        background: rgba(255,255,255,0.06); color: var(--text); border: 1px solid var(--border);
        white-space:nowrap;
    }
    .pill.good{ background: rgba(22,163,74,0.14); border-color: rgba(22,163,74,0.30); }
    .pill.warn{ background: rgba(245,158,11,0.14); border-color: rgba(245,158,11,0.30); }
    .pill.bad { background: rgba(239,68,68,0.14); border-color: rgba(239,68,68,0.30); }
    .pill.neutral{ background: rgba(148,163,184,0.14); border-color: rgba(148,163,184,0.30); }
    .sectionTitle { font-size: 16px; font-weight: 800; color: var(--text); margin-top: 6px; }
    .divider { height: 1px; background: var(--border); margin: 10px 0 14px; }
    .bullet { margin: 0; padding-left: 16px; color: var(--text); }
    .bullet li { margin: 6px 0; color: var(--text); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(a, b):
    if b == 0:
        return 0.0 if a == 0 else 999.0
    return (a - b) / b * 100

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

def story_confidence(residual_ratio: float):
    if residual_ratio <= 0.10:
        return "High"
    if residual_ratio <= 0.25:
        return "Medium"
    return "Low"

def pill(label: str, status: str, kind: str):
    # kind: good / warn / bad / neutral
    return f'<span class="pill {kind}">{label}: {status}</span>'

def line_chart(df, cols, title, height=280):
    d = df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=""),
        color=alt.Color("metric:N", title=""),
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=",.4f")],
    ).properties(title=title, height=height)

# =============================
# ✅ ALTair v6-safe Waterfall (NO alt.condition) — fixes your error
# =============================
def waterfall_steps(start_total, deltas, labels, end_label):
    steps = []
    running = float(start_total)

    # Start total
    steps.append({"Step": labels[0], "Start": 0.0, "End": running, "Delta": running, "Type": "total"})

    # Deltas
    for lab, d in zip(labels[1:], deltas):
        d = float(d)
        steps.append({"Step": lab, "Start": running, "End": running + d, "Delta": d, "Type": "delta"})
        running += d

    # End total
    steps.append({"Step": end_label, "Start": 0.0, "End": running, "Delta": running, "Type": "total"})

    df = pd.DataFrame(steps)

    def pick_color(row):
        if row["Type"] == "total":
            return "#334155"          # slate totals
        return "#16a34a" if row["Delta"] >= 0 else "#ef4444"  # green/red deltas

    df["Color"] = df.apply(pick_color, axis=1)

    for c in ["Start", "End", "Delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def waterfall_chart(df, title, height=260):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Step:N", sort=None, title=""),
            y=alt.Y("Start:Q", title=""),
            y2="End:Q",
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Step:N"),
                alt.Tooltip("Delta:Q", format=",.2f"),
                alt.Tooltip("End:Q", format=",.2f"),
            ],
        )
        .properties(title=title, height=height)
    )

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

def fmt_value(v, kind):
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{v:,.2f}%"
    return f"{v:,.2f}"

# =============================
# METRIC-AWARE ANOMALY STATUS
# =============================
def anomaly_status_from_z(z, mode="lower_is_bad"):
    """
    mode:
      - lower_is_bad: only negative deviations matter (drop monitoring)
      - two_sided: deviations in either direction matter (CTR-style)
    returns: ("Good"/"Watch"/"Bad", css_kind)
    """
    if mode == "two_sided":
        az = abs(z)
        if az < 1:
            return "Good", "good"
        if az < 2:
            return "Watch", "warn"
        return "Bad", "bad"
    else:
        # lower is bad
        if z >= -1:
            return "Good", "good"
        if z >= -2:
            return "Watch", "warn"
        return "Bad", "bad"

# =============================
# IMPACT STATUS (₹ share of total loss vs expected)
# =============================
def impact_status(share):
    """
    share is 0..1
    """
    if share >= 0.45:
        return "High", "bad"
    if share >= 0.20:
        return "Medium", "warn"
    return "Low", "neutral"

# =============================
# HEADER
# =============================
st.markdown('<div class="title">📊 Revenue Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Metric-aware anomaly status + impact-based story (GA4 + GAM) • single-site</div>', unsafe_allow_html=True)

# =============================
# LOAD CSV
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# single site
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
# BASE TOTALS
# =============================
base_cols = ["revenue", "ad_requests", "impressions", "clicks", "sessions", "users", "pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# ratios derived from totals
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)
daily["pageviews_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("Controls")

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

if baseline_df.empty:
    st.error("Baseline window has no data. Increase window or pick another day.")
    st.stop()

# =============================
# EXPECTED (baseline median totals) — single story reference
# =============================
expected_totals = baseline_df[base_cols].median(numeric_only=True).to_dict()
exp = dict(expected_totals)

# expected ratios from expected totals
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
# SYMMETRIC (Shapley-like) DECOMPOSITION (minimizes artificial residual)
# Revenue = Impressions * eCPM / 1000
# Impressions = Requests * FillFraction
# =============================
rev_b = float(exp["revenue"])
rev_t = float(t["revenue"])
d_rev = rev_t - rev_b

imp_b = float(exp["impressions"])
imp_t = float(t["impressions"])

ecpm_b = safe_div(rev_b, imp_b, 1000)
ecpm_t = safe_div(rev_t, imp_t, 1000)

imp_effect = 0.5 * (imp_t - imp_b) * ((ecpm_t + ecpm_b) / 1000)
ecpm_effect = 0.5 * (ecpm_t - ecpm_b) * ((imp_t + imp_b) / 1000)
rev_residual = d_rev - (imp_effect + ecpm_effect)

den = max(abs(d_rev), 1.0)
residual_ratio = abs(rev_residual) / den
story_conf = story_confidence(residual_ratio)

# Impressions decomposition
req_b = float(exp["ad_requests"])
req_t = float(t["ad_requests"])

fill_b_frac = safe_div(imp_b, req_b, 1) if req_b else 0.0
fill_t_frac = safe_div(imp_t, req_t, 1) if req_t else 0.0

d_imp = imp_t - imp_b
req_effect = 0.5 * (req_t - req_b) * (fill_t_frac + fill_b_frac)
fill_effect = 0.5 * (fill_t_frac - fill_b_frac) * (req_t + req_b)
imp_residual = d_imp - (req_effect + fill_effect)

# =============================
# LOSS CONTRIBUTIONS (for Impact chips)
# We only allocate when there is a loss vs expected (gap > 0)
# =============================
loss_total = max(gap, 0.0)

loss_from_imp = max(-imp_effect, 0.0)
loss_from_ecpm = max(-ecpm_effect, 0.0)
loss_from_resid = max(-rev_residual, 0.0)

loss_parts_sum = loss_from_imp + loss_from_ecpm + loss_from_resid
if loss_parts_sum <= 0:
    # avoid divide by zero; show neutral
    loss_share_imp = loss_share_ecpm = loss_share_res = 0.0
else:
    loss_share_imp = loss_from_imp / loss_parts_sum
    loss_share_ecpm = loss_from_ecpm / loss_parts_sum
    loss_share_res = loss_from_resid / loss_parts_sum

# metric impact shares (only meaningful for key drivers)
impact_map = {
    "revenue": 1.0 if loss_total > 0 else 0.0,
    "impressions": loss_share_imp if loss_total > 0 else 0.0,
    "ecpm": loss_share_ecpm if loss_total > 0 else 0.0,
    "residual": loss_share_res if loss_total > 0 else 0.0,
    # We can proxy impact for requests/fill by splitting impressions contribution:
    "ad_requests": 0.0,
    "fill_rate": 0.0,
    "ctr": 0.0,
}
# split the impressions loss into requests vs fill (revenue-equivalent, using baseline ecpm)
rev_per_imp = ecpm_b / 1000.0
req_loss_rev_eq = max(-(req_effect * rev_per_imp), 0.0)
fill_loss_rev_eq = max(-(fill_effect * rev_per_imp), 0.0)
den2 = req_loss_rev_eq + fill_loss_rev_eq
if den2 > 0 and loss_total > 0:
    impact_map["ad_requests"] = (req_loss_rev_eq / den2) * loss_share_imp
    impact_map["fill_rate"] = (fill_loss_rev_eq / den2) * loss_share_imp

# =============================
# KPI DEFINITIONS (metric-aware anomaly mode)
# =============================
kpi_defs = [
    ("revenue", "Revenue", "money", "lower_is_bad"),
    ("impressions", "Impressions", "int", "lower_is_bad"),
    ("ad_requests", "Ad Requests", "int", "lower_is_bad"),
    ("fill_rate", "Fill Rate", "pct", "lower_is_bad"),
    ("ecpm", "eCPM", "money", "lower_is_bad"),
    ("ctr", "CTR", "pct", "two_sided"),  # <-- important fix
]

# robust baseline for each metric
kpi = {}
for key, label, kind, mode in kpi_defs:
    med, mad = median_mad(baseline_df[key])
    val_today = float(t.get(key, 0.0))
    val_y = float(y.get(key, 0.0)) if y is not None else 0.0
    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)

    anomaly_lbl, anomaly_kind = anomaly_status_from_z(z, mode=mode)

    # Impact: based on loss allocation (money story). If no loss, impact is Low/Neutral.
    share = float(impact_map.get(key, 0.0))
    imp_lbl, imp_kind = impact_status(share)

    delta_y = pct_change(val_today, val_y) if y is not None else 0.0

    kpi[key] = dict(
        label=label, kind=kind,
        today=val_today, yesterday=val_y,
        z=z, conf=conf, delta_y=delta_y,
        anomaly_lbl=anomaly_lbl, anomaly_kind=anomaly_kind,
        impact_share=share, impact_lbl=imp_lbl, impact_kind=imp_kind
    )

def kpi_card(name, v_str, delta_context_str, z, conf, anomaly_lbl, anomaly_kind, impact_lbl, impact_kind):
    return f"""
    <div class="card">
        <div class="kpiTop">
            <div>
                <div class="kpiName">{name}</div>
                <div class="kpiVal">{v_str}</div>
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;">
                {pill("Anomaly", anomaly_lbl, anomaly_kind)}
                {pill("Impact", impact_lbl, impact_kind)}
            </div>
        </div>
        <div class="kpiSub">{delta_context_str}</div>
        <div class="kpiSub">Baseline anomaly (robust): Z {z:.2f} • Confidence {conf:.1f}%</div>
    </div>
    """

def meter_card(expected, actual):
    g = expected - actual
    g_pct = safe_div(g, expected, 100) if expected > 0 else 0.0
    if expected <= 0:
        lbl, kind = "No baseline", "neutral"
    elif g_pct <= 5:
        lbl, kind = "Normal", "good"
    elif g_pct <= 15:
        lbl, kind = "Warning", "warn"
    else:
        lbl, kind = "Critical", "bad"
    sign = "-" if g > 0 else "+"
    label2 = "Lost vs Expected" if g > 0 else "Above Expected"
    return f"""
    <div class="card">
        <div class="kpiTop">
            <div>
                <div class="kpiName">Today vs Expected</div>
                <div class="kpiVal">{sign}{abs(g):,.2f}</div>
            </div>
            <div>{pill("Status", lbl, kind)}</div>
        </div>
        <div class="kpiSub">{label2} • Expected {expected:,.2f} • Actual {actual:,.2f}</div>
        <div class="kpiSub">Gap: {g_pct:+.1f}%</div>
    </div>
    """

# =============================
# KPI ROW (PowerBI-ish)
# =============================
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.markdown(kpi_card(
    "Revenue",
    fmt_value(kpi["revenue"]["today"], "money"),
    f"Δ vs Yesterday (context): {kpi['revenue']['delta_y']:+.2f}%",
    kpi["revenue"]["z"], kpi["revenue"]["conf"],
    kpi["revenue"]["anomaly_lbl"], kpi["revenue"]["anomaly_kind"],
    kpi["revenue"]["impact_lbl"], kpi["revenue"]["impact_kind"],
), unsafe_allow_html=True)

c2.markdown(kpi_card(
    "Impressions",
    fmt_value(kpi["impressions"]["today"], "int"),
    f"Δ vs Yesterday (context): {kpi['impressions']['delta_y']:+.2f}%",
    kpi["impressions"]["z"], kpi["impressions"]["conf"],
    kpi["impressions"]["anomaly_lbl"], kpi["impressions"]["anomaly_kind"],
    kpi["impressions"]["impact_lbl"], kpi["impressions"]["impact_kind"],
), unsafe_allow_html=True)

c3.markdown(kpi_card(
    "Ad Requests",
    fmt_value(kpi["ad_requests"]["today"], "int"),
    f"Δ vs Yesterday (context): {kpi['ad_requests']['delta_y']:+.2f}%",
    kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"],
    kpi["ad_requests"]["anomaly_lbl"], kpi["ad_requests"]["anomaly_kind"],
    kpi["ad_requests"]["impact_lbl"], kpi["ad_requests"]["impact_kind"],
), unsafe_allow_html=True)

c4.markdown(kpi_card(
    "Fill Rate",
    fmt_value(kpi["fill_rate"]["today"], "pct"),
    f"Δ vs Yesterday (context): {kpi['fill_rate']['delta_y']:+.2f}%",
    kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"],
    kpi["fill_rate"]["anomaly_lbl"], kpi["fill_rate"]["anomaly_kind"],
    # impact for fill rate comes from the impressions-loss split
    kpi["fill_rate"]["impact_lbl"], kpi["fill_rate"]["impact_kind"],
), unsafe_allow_html=True)

c5.markdown(kpi_card(
    "eCPM",
    fmt_value(kpi["ecpm"]["today"], "money"),
    f"Δ vs Yesterday (context): {kpi['ecpm']['delta_y']:+.2f}%",
    kpi["ecpm"]["z"], kpi["ecpm"]["conf"],
    kpi["ecpm"]["anomaly_lbl"], kpi["ecpm"]["anomaly_kind"],
    kpi["ecpm"]["impact_lbl"], kpi["ecpm"]["impact_kind"],
), unsafe_allow_html=True)

c6.markdown(kpi_card(
    "CTR",
    fmt_value(kpi["ctr"]["today"], "pct"),
    f"Δ vs Yesterday (context): {kpi['ctr']['delta_y']:+.2f}%",
    kpi["ctr"]["z"], kpi["ctr"]["conf"],
    kpi["ctr"]["anomaly_lbl"], kpi["ctr"]["anomaly_kind"],  # <-- two-sided anomaly is used here
    kpi["ctr"]["impact_lbl"], kpi["ctr"]["impact_kind"],
), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# =============================
# ROOT CAUSE STORY (Expected → Today)
# =============================
def build_root_story():
    if expected_revenue <= 0:
        return "No baseline", ["Not enough baseline window to diagnose."]

    if gap <= 0:
        return "Healthy", ["Revenue is at/above expected today."]

    lines = [
        f"Missed expected revenue by **{gap:,.2f}** (Expected {expected_revenue:,.2f} → Today {actual_revenue:,.2f}).",
        f"Loss split: **{loss_share_imp*100:.0f}% Impressions**, **{loss_share_ecpm*100:.0f}% eCPM**, **{loss_share_res*100:.0f}% Unexplained**.",
        f"Story confidence: **{story_conf}** (residual ratio {residual_ratio:.2f}).",
    ]

    # Primary driver
    parts = sorted(
        [("Impressions", loss_from_imp), ("eCPM", loss_from_ecpm), ("Residual", loss_from_resid)],
        key=lambda x: x[1],
        reverse=True
    )
    primary = parts[0][0]

    if primary == "Residual" and story_conf == "Low":
        lines.append("Main signal looks unexplained → likely **GAM revenue lag** or **merge mismatch** today.")
        return "Data confidence issue", lines

    if primary == "eCPM":
        lines.append("Primary driver is **eCPM** → buyers paid less per 1,000 impressions.")
        lines.append("Next: segment by **country / device / ad unit** to find where eCPM dropped.")
        return "eCPM drop", lines

    # Impressions path: requests vs fill
    if den2 > 0:
        lines.append(f"Inside impressions: **{(req_loss_rev_eq/den2)*100:.0f}% Requests**, **{(fill_loss_rev_eq/den2)*100:.0f}% Fill** (revenue-equivalent).")

    # Heuristic to label ad-loading vs traffic mix
    pv_b = float(exp["pageviews"])
    pv_t = float(t["pageviews"])
    rpp_b = safe_div(req_b, pv_b, 1)
    rpp_t = safe_div(req_t, pv_t, 1)

    pageviews_stable = (pv_b > 0 and pv_t > pv_b * 0.95)
    rpp_down = (rpp_b > 0 and rpp_t < rpp_b * 0.90)

    if fill_loss_rev_eq >= req_loss_rev_eq:
        lines.append("Pattern matches **Fill issue** (requests exist, fewer filled). Check floors, blocks, demand delivery, policy.")
        return "Fill issue", lines

    if pageviews_stable and rpp_down:
        lines.append("Pattern matches **Ad loading issue** (pageviews stable, requests/pageview down). Check tags, CMP, adblock, templates.")
        return "Ad loading issue", lines

    lines.append("Pattern matches **Requests/traffic/mix** (requests moved with traffic/mix).")
    return "Requests/traffic issue", lines

root_title, root_lines = build_root_story()

# Story row
s1, s2 = st.columns([1.1, 1.9])
with s1:
    st.markdown(meter_card(expected_revenue, actual_revenue), unsafe_allow_html=True)
with s2:
    conf_kind = "good" if story_conf == "High" else "warn" if story_conf == "Medium" else "bad"
    st.markdown(
        f"""
        <div class="card">
            <div class="kpiTop">
                <div>
                    <div class="kpiName">Root cause (Expected → Today)</div>
                    <div class="kpiVal">{root_title}</div>
                </div>
                <div>{pill("Story", story_conf, conf_kind)}</div>
            </div>
            <ul class="bullet">
                {''.join([f"<li>{ln}</li>" for ln in root_lines])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Context only: yesterday
if y is not None:
    st.caption(f"Context only: Revenue vs Yesterday = {pct_change(actual_revenue, float(y['revenue'])):+.2f}%")

# =============================
# WATERFALLS
# =============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sectionTitle">📉 What drove the change</div>', unsafe_allow_html=True)
st.caption("Correct waterfall: Start total → delta steps → end total (Altair v6-safe)")

w1, w2 = st.columns(2)
with w1:
    wf_rev = waterfall_steps(
        start_total=rev_b,
        deltas=[imp_effect, ecpm_effect, rev_residual],
        labels=["Expected Revenue", "Impressions effect", "eCPM effect", "Residual"],
        end_label="Today Revenue",
    )
    st.altair_chart(waterfall_chart(wf_rev, "Revenue (Expected → Today)"), use_container_width=True)

with w2:
    wf_imp = waterfall_steps(
        start_total=imp_b,
        deltas=[req_effect, fill_effect, imp_residual],
        labels=["Expected Impressions", "Requests effect", "Fill effect", "Residual"],
        end_label="Today Impressions",
    )
    st.altair_chart(waterfall_chart(wf_imp, "Impressions (Expected → Today)"), use_container_width=True)

# =============================
# TRENDS
# =============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sectionTitle">📈 Trends</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    st.altair_chart(line_chart(display_df, ["revenue", "ecpm"], "Revenue vs eCPM"), use_container_width=True)
with t2:
    st.altair_chart(line_chart(display_df, ["ad_requests", "fill_rate", "impressions"], "Requests → Fill → Impressions"), use_container_width=True)

t3, t4 = st.columns(2)
with t3:
    st.altair_chart(line_chart(display_df, ["sessions", "pageviews"], "Traffic (Sessions + Pageviews)"), use_container_width=True)
with t4:
    st.altair_chart(
        line_chart(display_df, ["requests_per_pageview", "impressions_per_session", "pageviews_per_session"], "Ad Loading & Engagement Density"),
        use_container_width=True
    )

# =============================
# SANITY CHECKS
# =============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sectionTitle">🧪 Data sanity checks</div>', unsafe_allow_html=True)

checks = []
if t["sessions"] > 0 and t["ad_requests"] == 0:
    checks.append("Sessions > 0 but Ad Requests = 0 → tagging/ad call issue or missing GAM data.")
if t["ad_requests"] > 0 and t["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed or reporting lag.")
if t["impressions"] > 0 and t["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue reporting lag or merge mismatch.")
if t["pageviews"] > 0 and t["sessions"] == 0:
    checks.append("Pageviews > 0 but Sessions = 0 → GA4 export mismatch.")
if story_conf == "Low":
    checks.append("Story confidence LOW due to high residual ratio → treat root cause as directional today.")

if not checks:
    st.success("No obvious integrity red flags for selected day.")
else:
    for c in checks:
        st.warning(c)

st.caption(
    "Status chips: Anomaly = baseline robust Z (metric-aware: CTR is two-sided). "
    "Impact = ₹ share of expected→today loss allocation. Expected baseline median is the only story reference."
)
