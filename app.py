# app.py — Revenue Intelligence Dashboard (Story-first, Math-correct, Cleaner UI)
# Fixes included:
# ✅ One story reference everywhere: Expected baseline median used for root-cause
# ✅ Yesterday is context only (no narrative mixing)
# ✅ Ratios derived from totals ONLY (never summed)
# ✅ Symmetric (Shapley-style) decomposition → avoids fake “residual” interaction noise
# ✅ Correct waterfall charts (start total + delta steps + end total)
# ✅ Story confidence based on residual ratio (warns when data can’t explain movement)
# ✅ Cleaner, less noisy UI (PowerBI-like layout)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# PAGE CONFIG + THEME
# =============================
st.set_page_config(page_title="Revenue Intelligence (Story)", layout="wide")

st.markdown(
    """
<style>
    :root{
        --bg:#0b1220;
        --panel:#111a2e;
        --panel2:#0f172a;
        --text:#e5e7eb;
        --muted:#9ca3af;
        --good:#16a34a;
        --warn:#f59e0b;
        --bad:#ef4444;
        --ink:#111827;
        --chip:#1f2a44;
        --border: rgba(255,255,255,0.08);
    }
    .wrap { padding-top: 4px; }
    .title { font-size: 26px; font-weight: 900; margin-bottom: 0px; color: var(--text); }
    .sub { font-size: 13px; color: var(--muted); margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
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
    .sectionTitle { font-size: 16px; font-weight: 800; color: var(--text); margin-top: 6px; }
    .hint { font-size: 12px; color: var(--muted); }
    .divider { height: 1px; background: var(--border); margin: 10px 0 14px; }
    .bullet { margin: 0; padding-left: 16px; color: var(--text); }
    .bullet li { margin: 6px 0; color: var(--text); }
    .small { font-size: 12px; color: var(--muted); }
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

def status_pill(label: str, status: str):
    cls = "good" if status == "Good" else "warn" if status == "Watch" else "bad"
    return f'<span class="pill {cls}">{label}: {status}</span>'

def kpi_card_html(name, value_str, delta_str, z, conf):
    # status based on robust z
    if z >= -1:
        status = "Good"
    elif z >= -2:
        status = "Watch"
    else:
        status = "Bad"

    pill = status_pill("Status", status)
    return f"""
    <div class="card">
        <div class="kpiTop">
            <div>
                <div class="kpiName">{name}</div>
                <div class="kpiVal">{value_str}</div>
            </div>
            <div>{pill}</div>
        </div>
        <div class="kpiSub">{delta_str}</div>
        <div class="kpiSub">Robust Z: {z:.2f} • Confidence: {conf:.1f}%</div>
    </div>
    """

def meter_card_html(expected, actual):
    gap = expected - actual  # + means loss
    gap_pct = safe_div(gap, expected, 100) if expected > 0 else 0.0

    if expected <= 0:
        status = "No baseline"
        cls = "warn"
    elif gap_pct <= 5:
        status = "Normal"
        cls = "good"
    elif gap_pct <= 15:
        status = "Warning"
        cls = "warn"
    else:
        status = "Critical"
        cls = "bad"

    sign = "-" if gap > 0 else "+"
    label = "Lost vs Expected" if gap > 0 else "Above Expected"
    return f"""
    <div class="card">
        <div class="kpiTop">
            <div>
                <div class="kpiName">Today vs Expected</div>
                <div class="kpiVal">{sign}{abs(gap):,.2f}</div>
            </div>
            <div><span class="pill {cls}">{status}</span></div>
        </div>
        <div class="kpiSub">{label} • Expected {expected:,.2f} • Actual {actual:,.2f}</div>
        <div class="kpiSub">Gap: {gap_pct:+.1f}%</div>
    </div>
    """

def line_chart(df, cols, title, height=280):
    d = df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=""),
        color=alt.Color("metric:N", title=""),
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=",.4f")],
    ).properties(title=title, height=height)

def waterfall_steps(start_total, deltas, labels, end_label):
    # start_total and end total are totals; middle are deltas
    steps = []
    running = float(start_total)

    steps.append({"Step": labels[0], "Start": 0.0, "End": running, "Delta": running, "Type": "total"})
    for lab, d in zip(labels[1:], deltas):
        d = float(d)
        steps.append({"Step": lab, "Start": running, "End": running + d, "Delta": d, "Type": "delta"})
        running += d

    steps.append({"Step": end_label, "Start": 0.0, "End": running, "Delta": running, "Type": "total"})
    return pd.DataFrame(steps)

def waterfall_chart(df, title, height=260):
    base = alt.Chart(df).encode(
        x=alt.X("Step:N", sort=None),
        tooltip=["Step", alt.Tooltip("Delta:Q", format=",.2f"), alt.Tooltip("End:Q", format=",.2f")]
    )

    bars = base.mark_bar().encode(
        y=alt.Y("Start:Q", title=""),
        y2="End:Q",
        color=alt.condition(
            alt.datum.Type == "total",
            alt.value("#334155"),
            alt.condition(alt.datum.Delta >= 0, alt.value("#16a34a"), alt.value("#ef4444"))
        )
    ).properties(title=title, height=height)

    return bars

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

# =============================
# HEADER
# =============================
st.markdown('<div class="wrap">', unsafe_allow_html=True)
st.markdown('<div class="title">📊 Revenue Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Story-first root cause (GA4 + GAM) • single-site • mathematically consistent</div>', unsafe_allow_html=True)

# =============================
# LOAD CSV
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# Single site
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

# Derived ratios from totals ONLY
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)
daily["pageviews_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR
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
# STORY BASE: EXPECTED (BASELINE MEDIAN TOTALS)
# =============================
expected_totals = baseline_df[base_cols].median(numeric_only=True).to_dict()
exp = dict(expected_totals)

# Derive expected ratios from expected totals
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
# KPI ENGINE (Robust baseline)
# KPI cards show Δ vs Yesterday (context), NOT used for story logic
# =============================
kpi_defs = [
    ("revenue", "Revenue"),
    ("impressions", "Impressions"),
    ("ad_requests", "Ad Requests"),
    ("fill_rate", "Fill Rate (%)"),
    ("ecpm", "eCPM"),
]

kpi = {}
for key, label in kpi_defs:
    med, mad = median_mad(baseline_df[key])
    val_today = float(t.get(key, 0.0))
    val_y = float(y.get(key, 0.0)) if y is not None else 0.0
    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta_y = pct_change(val_today, val_y) if y is not None else 0.0
    kpi[key] = dict(label=label, today=val_today, yesterday=val_y, med=med, mad=mad, z=z, conf=conf, delta_y=delta_y)

# =============================
# TOP KPI ROW (clean)
# =============================
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(kpi_card_html("Revenue", f"{kpi['revenue']['today']:,.2f}",
                          f"Δ vs Yesterday: {kpi['revenue']['delta_y']:+.2f}%",
                          kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)

c2.markdown(kpi_card_html("Impressions", f"{kpi['impressions']['today']:,.0f}",
                          f"Δ vs Yesterday: {kpi['impressions']['delta_y']:+.2f}%",
                          kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)

c3.markdown(kpi_card_html("Ad Requests", f"{kpi['ad_requests']['today']:,.0f}",
                          f"Δ vs Yesterday: {kpi['ad_requests']['delta_y']:+.2f}%",
                          kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)

c4.markdown(kpi_card_html("Fill Rate", f"{kpi['fill_rate']['today']:,.2f}%",
                          f"Δ vs Yesterday: {kpi['fill_rate']['delta_y']:+.2f}%",
                          kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"]), unsafe_allow_html=True)

c5.markdown(kpi_card_html("eCPM", f"{kpi['ecpm']['today']:,.2f}",
                          f"Δ vs Yesterday: {kpi['ecpm']['delta_y']:+.2f}%",
                          kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# =============================
# STORY ROW (Expected → Today)
# =============================
s1, s2, s3 = st.columns([1.1, 1.4, 1.5])

s1.markdown(meter_card_html(expected_revenue, actual_revenue), unsafe_allow_html=True)

# =============================
# SYMMETRIC DECOMPOSITION (Revenue)
# Revenue = Impressions * eCPM / 1000
# =============================
rev_b = float(exp["revenue"])
rev_t = float(t["revenue"])
d_rev = rev_t - rev_b

imp_b = float(exp["impressions"])
imp_t = float(t["impressions"])

ecpm_b = safe_div(rev_b, imp_b, 1000)
ecpm_t = safe_div(rev_t, imp_t, 1000)

# Shapley-style symmetric contributions
imp_effect = 0.5 * (imp_t - imp_b) * ((ecpm_t + ecpm_b) / 1000)
ecpm_effect = 0.5 * (ecpm_t - ecpm_b) * ((imp_t + imp_b) / 1000)

rev_residual = d_rev - (imp_effect + ecpm_effect)

# Story confidence from residual ratio
den = max(abs(d_rev), 1.0)
residual_ratio = abs(rev_residual) / den
conf_label = story_confidence(residual_ratio)

if conf_label == "High":
    conf_pill = '<span class="pill good">Story confidence: High</span>'
elif conf_label == "Medium":
    conf_pill = '<span class="pill warn">Story confidence: Medium</span>'
else:
    conf_pill = '<span class="pill bad">Story confidence: Low</span>'

# =============================
# SYMMETRIC DECOMPOSITION (Impressions)
# Impressions = Requests * FillFraction
# =============================
req_b = float(exp["ad_requests"])
req_t = float(t["ad_requests"])

fill_b_frac = safe_div(imp_b, req_b, 1) if req_b else 0.0
fill_t_frac = safe_div(imp_t, req_t, 1) if req_t else 0.0

d_imp = imp_t - imp_b

req_effect = 0.5 * (req_t - req_b) * (fill_t_frac + fill_b_frac)
fill_effect = 0.5 * (fill_t_frac - fill_b_frac) * (req_t + req_b)

imp_residual = d_imp - (req_effect + fill_effect)

# =============================
# ROOT CAUSE SUMMARY (loss-only)
# =============================
def root_cause_summary():
    if expected_revenue <= 0:
        return ("No baseline", ["Not enough baseline revenue to diagnose."])

    if gap <= 0:
        return ("Healthy", ["Revenue is at/above expected today."])

    # Loss contributions only
    loss_imp = max(-imp_effect, 0.0)
    loss_ecpm = max(-ecpm_effect, 0.0)
    loss_res = max(-rev_residual, 0.0)

    total_loss_parts = loss_imp + loss_ecpm + loss_res
    if total_loss_parts <= 0:
        # revenue dropped but effects show positive — possible very small deltas/rounding
        return ("Unclear", ["The model cannot allocate the loss cleanly today (very small deltas/rounding)."])

    s_imp = loss_imp / total_loss_parts * 100
    s_ecpm = loss_ecpm / total_loss_parts * 100
    s_res = loss_res / total_loss_parts * 100

    lines = [
        f"Today missed expected revenue by **{gap:,.2f}**.",
        f"Loss split: **{s_imp:.0f}% Impressions**, **{s_ecpm:.0f}% eCPM**, **{s_res:.0f}% Unexplained**.",
        f"Residual ratio: **{residual_ratio:.2f}** → Confidence: **{conf_label}**."
    ]

    # Primary driver
    parts = sorted([("Impressions", loss_imp), ("eCPM", loss_ecpm), ("Residual", loss_res)], key=lambda x: x[1], reverse=True)
    primary = parts[0][0]

    if primary == "Residual" and conf_label == "Low":
        lines.append("Primary driver looks unexplained → likely **GAM revenue lag** or **merge mismatch** for today.")
        return ("Data confidence issue", lines)

    if primary == "eCPM":
        lines.append("Primary driver: **Auction value drop (eCPM)**. Next: segment by **country/device/ad unit**.")
        return ("eCPM drop", lines)

    # Impressions
    # Convert request/fill effects to revenue-equivalent using expected eCPM (baseline)
    rev_per_imp = ecpm_b / 1000.0
    req_rev_eq = max(-(req_effect * rev_per_imp), 0.0)
    fill_rev_eq = max(-(fill_effect * rev_per_imp), 0.0)

    denom2 = req_rev_eq + fill_rev_eq
    if denom2 > 0:
        lines.append(f"Inside impressions: **{req_rev_eq/denom2*100:.0f}% Requests**, **{fill_rev_eq/denom2*100:.0f}% Fill** (revenue-equivalent).")

    # Ad loading heuristic (requests per pageview)
    pv_b = float(exp["pageviews"])
    pv_t = float(t["pageviews"])
    rpp_b = safe_div(req_b, pv_b, 1)
    rpp_t = safe_div(req_t, pv_t, 1)

    pageviews_stable = (pv_b > 0 and pv_t > pv_b * 0.95)
    rpp_down = (rpp_b > 0 and rpp_t < rpp_b * 0.90)

    if fill_rev_eq >= req_rev_eq:
        lines.append("Pattern: **Demand / fill issue** (requests exist, fewer filled). Look at floors, blocks, partner delivery, policy.")
        return ("Fill issue", lines)

    if pageviews_stable and rpp_down:
        lines.append("Pattern: **Ad loading issue** (pageviews stable, requests/pageview down). Look at tags, CMP, adblock, template changes.")
        return ("Ad loading issue", lines)

    lines.append("Pattern: **Traffic/page-mix change** (requests moved with pageviews/mix).")
    return ("Requests/traffic issue", lines)

root_title, root_lines = root_cause_summary()

with s2:
    st.markdown(f'<div class="card"><div class="kpiTop"><div><div class="kpiName">Root Cause</div><div class="kpiVal">{root_title}</div></div><div>{conf_pill}</div></div>', unsafe_allow_html=True)
    st.markdown("<ul class='bullet'>" + "".join([f"<li>{ln}</li>" for ln in root_lines]) + "</ul></div>", unsafe_allow_html=True)

with s3:
    st.markdown('<div class="card"><div class="kpiName">Where to look next</div><div class="sectionTitle">Fast checks (5 mins)</div>', unsafe_allow_html=True)

    checklist = []
    if root_title in ["Data confidence issue"]:
        checklist = [
            "Check if **GAM revenue** for today is final (often lags).",
            "Verify merge keys: **date timezone** (GA4 vs GAM), missing rows.",
            "Re-run after 24 hours and compare."
        ]
    elif root_title in ["Fill issue"]:
        checklist = [
            "Recent **floor / pricing rule** changes?",
            "Any **blocking / protections / policy** triggers?",
            "Demand partner delivery drop (AdX / HB / Open Bidding)?",
            "Geo/device mix shifted to weaker demand?"
        ]
    elif root_title in ["Ad loading issue"]:
        checklist = [
            "Requests/pageview down → check **tag firing**, header bidding errors.",
            "CMP/consent changes blocking requests.",
            "Adblock/script blocked or template slot removed.",
            "New lazy-load settings too aggressive."
        ]
    elif root_title in ["eCPM drop"]:
        checklist = [
            "Segment eCPM by **country/device/ad unit** to find where value fell.",
            "Ad unit mix changed (more low-value placements)?",
            "Viewability or IVT signals causing bidder pullback?"
        ]
    else:
        checklist = [
            "Check sessions/pageviews changes (traffic mix).",
            "Check requests/pageview and impressions/session.",
            "Validate if today is same weekday pattern."
        ]

    st.markdown("<ul class='bullet'>" + "".join([f"<li>{x}</li>" for x in checklist]) + "</ul></div>", unsafe_allow_html=True)

# =============================
# WATERFALLS (Correct)
# =============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sectionTitle">📉 What drove the change (waterfalls)</div>', unsafe_allow_html=True)
st.caption("Correct waterfall: Start total → delta steps → end total")

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
# TRENDS (2x2, clean)
# =============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sectionTitle">📈 Trends</div>', unsafe_allow_html=True)
st.caption("Baseline window → today")

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

if conf_label == "Low":
    checks.append("Story confidence LOW due to high residual ratio → treat root-cause as directional today.")

if not checks:
    st.success("No obvious integrity red flags for selected day.")
else:
    for c in checks:
        st.warning(c)

st.caption(
    "Story reference is always Expected baseline median. Yesterday is context only. "
    "All ratios derived from totals. Decomposition is symmetric (reduces fake residual)."
)

st.markdown("</div>", unsafe_allow_html=True)
