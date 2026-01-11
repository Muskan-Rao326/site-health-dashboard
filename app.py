import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# CONFIG + STYLE
# =============================
st.set_page_config(page_title="Revenue Intelligence (Root Cause)", layout="wide")

st.markdown("""
<style>
    .title { font-size: 26px; font-weight: 800; margin-bottom: 2px; }
    .sub { font-size: 13px; opacity: 0.8; margin-bottom: 14px; }
    .card { border-radius: 14px; padding: 14px; color: white; }
    .kpi-label { font-size: 13px; opacity: 0.95; }
    .kpi-val { font-size: 26px; font-weight: 800; margin-top: 2px; }
    .kpi-meta { font-size: 12px; opacity: 0.95; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Root-cause driven by metric decrements (GA4 + GAM) — single-site</div>', unsafe_allow_html=True)

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
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def color_from_z(z):
    if z >= -1: return "#2ECC71"
    if z >= -2: return "#F1C40F"
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

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

def line_chart(df, cols, title):
    d = df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("metric:N", title="Metric"),
        tooltip=["date:T", "metric:N", "value:Q"]
    ).properties(title=title, height=270)

def waterfall_like(df_steps, title):
    # df_steps columns: Step, Value
    df = df_steps.copy()
    df["Cumulative"] = df["Value"].cumsum()
    df["Start"] = df["Cumulative"] - df["Value"]

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Step:N", sort=None),
        y=alt.Y("Start:Q", title="Value"),
        y2="Cumulative:Q",
        tooltip=["Step", alt.Tooltip("Value:Q", format=",.2f")]
    ).properties(title=title, height=270)
    return chart

# =============================
# LOAD CSV
# =============================
uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# Single site mode
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    if len(sites) == 0:
        st.error("No site_name values found.")
        st.stop()
    site = sites[0]
    df_raw = df_raw[df_raw["site_name"] == site].copy()
    st.caption(f"Using site: **{site}**")
else:
    st.caption("No site_name column found; assuming already single-site.")

# Required additive columns
base_cols = ["revenue","ad_requests","impressions","clicks","sessions","users","pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily totals (truth)
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)  # clicks from GAM (confirmed)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR SLICERS
# =============================
st.sidebar.header("Slicers")

selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date()
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

# compare reference
if compare_mode == "Yesterday" and y:
    ref = y
else:
    ref = baseline_df.median(numeric_only=True).to_dict() if not baseline_df.empty else {}

# =============================
# KPI ENGINE (Median + MAD baseline)
# =============================
kpi_list = [
    ("revenue","Revenue",""),
    ("impressions","Impressions",""),
    ("ad_requests","Ad Requests",""),
    ("fill_rate","Fill Rate","%"),
    ("ecpm","eCPM",""),
    ("rpm","RPM",""),
    ("requests_per_pageview","Req/Pageview",""),
    ("ctr","CTR","%"),
]

kpi = {}
for key, label, suffix in kpi_list:
    s = baseline_df[key].dropna().astype(float)
    med = float(s.median()) if len(s) else 0.0
    mad = float(np.median(np.abs(s - med))) if len(s) else 0.0

    val_today = float(t.get(key, 0.0))
    val_ref = float(ref.get(key, 0.0)) if ref else 0.0

    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_ref) if val_ref else 0.0

    kpi[key] = dict(label=label, suffix=suffix, today=val_today, ref=val_ref, med=med, mad=mad, z=z, conf=conf, delta=delta)

# =============================
# EXPECTED REVENUE (Forecast)
# =============================
expected_revenue = float(baseline_df["revenue"].median()) if not baseline_df.empty else 0.0
actual_revenue = float(t["revenue"])
lost = expected_revenue - actual_revenue
lost_pct = safe_div(lost, expected_revenue, 100) if expected_revenue else 0.0

# =============================
# KPI ROW
# =============================
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi_card("Revenue", kpi["revenue"]["today"], kpi["revenue"]["delta"], kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)
c2.markdown(kpi_card("Impressions", kpi["impressions"]["today"], kpi["impressions"]["delta"], kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)
c3.markdown(kpi_card("Ad Requests", kpi["ad_requests"]["today"], kpi["ad_requests"]["delta"], kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)
c4.markdown(kpi_card("Fill Rate", kpi["fill_rate"]["today"], kpi["fill_rate"]["delta"], kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"], "%"), unsafe_allow_html=True)
c5.markdown(kpi_card("eCPM", kpi["ecpm"]["today"], kpi["ecpm"]["delta"], kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

# =============================
# ROOT CAUSE (DECREMENT-BASED) + WATERFALLS
# =============================
st.subheader("🧠 Root Cause from Metric Decrements")

left, right = st.columns([2, 1])

with right:
    st.subheader("🎯 Expected Revenue (Baseline Median)")
    st.metric("Expected", f"{expected_revenue:,.2f}")
    st.metric("Actual (Today)", f"{actual_revenue:,.2f}", delta=f"{pct_change(actual_revenue, expected_revenue):+.2f}%" if expected_revenue else None)
    st.metric("Lost vs Expected", f"{max(lost,0):,.2f}", delta=f"{lost_pct:+.2f}%" if expected_revenue else None)
    st.caption(f"Expected = median of last {baseline_days} days.")

with left:
    # Compare-based analysis (use yesterday if available, else baseline median)
    if compare_mode == "Yesterday" and y:
        base = y
        base_label = "Yesterday"
    else:
        base = ref
        base_label = "Baseline median"

    rev_base = float(base.get("revenue", 0.0))
    rev_today = float(t.get("revenue", 0.0))
    delta_rev = rev_today - rev_base

    imp_base = float(base.get("impressions", 0.0))
    imp_today = float(t.get("impressions", 0.0))

    # Compute eCPM from the same base objects (not from baseline_df)
    ecpm_base = safe_div(rev_base, imp_base, 1000)
    ecpm_today = safe_div(rev_today, imp_today, 1000)

    # Revenue decomposition:
    # ΔRevenue ≈ (ΔImpressions * base eCPM/1000) + (today impressions * ΔeCPM/1000) + residual
    imp_effect = (imp_today - imp_base) * (ecpm_base / 1000)
    ecpm_effect = imp_today * ((ecpm_today - ecpm_base) / 1000)
    residual = delta_rev - (imp_effect + ecpm_effect)

    wf_rev = pd.DataFrame({
        "Step": [f"{base_label} Revenue", "Impressions effect", "eCPM effect", "Residual", "Today Revenue"],
        "Value": [rev_base, imp_effect, ecpm_effect, residual, rev_today]
    })
    st.altair_chart(waterfall_like(wf_rev, "Revenue Change Decomposition"), use_container_width=True)

    # Now decompose impressions if impressions effect is the driver
    # Impressions ≈ Requests * FillRate/100
    req_base = float(base.get("ad_requests", 0.0))
    req_today = float(t.get("ad_requests", 0.0))
    fill_base = safe_div(imp_base, req_base, 100) if req_base else 0.0
    fill_today = safe_div(imp_today, req_today, 100) if req_today else 0.0
    delta_imp = imp_today - imp_base

    # ΔImpressions ≈ (ΔRequests * base fill/100) + (today requests * ΔFill/100) + residual
    req_effect = (req_today - req_base) * (fill_base / 100)
    fill_effect = req_today * ((fill_today - fill_base) / 100)
    imp_residual = delta_imp - (req_effect + fill_effect)

    wf_imp = pd.DataFrame({
        "Step": [f"{base_label} Impressions", "Requests effect", "Fill effect", "Residual", "Today Impressions"],
        "Value": [imp_base, req_effect, fill_effect, imp_residual, imp_today]
    })
    st.altair_chart(waterfall_like(wf_imp, "Impressions Change Decomposition"), use_container_width=True)

    # Auto Root Cause Statement (based on contributions)
    st.markdown("### ✅ Root Cause Summary")

    # Contribution magnitudes
    contribs = [
        ("Impressions effect", imp_effect),
        ("eCPM effect", ecpm_effect),
        ("Residual", residual)
    ]
    contribs_sorted = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
    top_name, top_val = contribs_sorted[0]

    direction = "down" if delta_rev < 0 else "up"
    st.write(f"• Revenue moved **{direction} {abs(delta_rev):,.2f}** vs {base_label}.")

    if top_name == "Impressions effect":
        st.write(f"• Primary driver: **Impressions** (impact {imp_effect:,.2f}).")
        # Drill inside impressions
        inner = [("Requests effect", req_effect), ("Fill effect", fill_effect)]
        inner_sorted = sorted(inner, key=lambda x: abs(x[1]), reverse=True)
        inner_name, inner_val = inner_sorted[0]
        if inner_name == "Requests effect":
            st.write(f"• Impressions fell mainly because **Ad Requests** fell (impact {req_effect:,.2f}).")
            # Determine if request drop is traffic or loading
            # requests/pageview change
            rpp_base = safe_div(req_base, float(base.get("pageviews", 0.0)), 1)
            rpp_today = safe_div(req_today, float(t.get("pageviews", 0.0)), 1)
            if rpp_today < rpp_base * 0.9:
                st.write("• Request drop looks like **Ad Loading issue** (requests per pageview decreased).")
            else:
                st.write("• Request drop looks like **Traffic/engagement issue** (requests per pageview stable).")
        else:
            st.write(f"• Impressions fell mainly because **Fill Rate** fell (impact {fill_effect:,.2f}).")
            st.write("• This points to **Demand/Floors/Blocks/Policy** type issue.")
    elif top_name == "eCPM effect":
        st.write(f"• Primary driver: **eCPM** (impact {ecpm_effect:,.2f}).")
        st.write("• This points to **Value/Auction** changes (geo mix, viewability, bidder competition, floors).")
    else:
        st.write("• Residual is the largest piece (usually means both drivers moved or rounding + metric mismatch).")

    # CTR is diagnostic only
    ctr_base = float(base.get("ctr", safe_div(float(base.get("clicks", 0.0)), float(base.get("impressions", 0.0)), 100)))
    ctr_today = float(t.get("ctr", safe_div(float(t.get("clicks", 0.0)), float(t.get("impressions", 0.0)), 100)))
    if ctr_today > ctr_base * 1.3:
        st.write("• CTR jumped sharply: treat this as a **diagnostic** (possible layout change or accidental clicks), not a revenue driver.")
    elif ctr_today < ctr_base * 0.7:
        st.write("• CTR dropped sharply: usually indicates lower engagement or different ad mix; still **not** the direct revenue driver.")

# =============================
# CTR SECTION (UPDATED EXPLANATION + USE)
# =============================
st.subheader("🧾 CTR (How to use it correctly)")

st.write("""
CTR is **Clicks / Impressions** (from GAM).
CTR is not a direct part of revenue for most CPM inventory.
So CTR is best used as a **health + risk signal**, not as “why revenue dropped”.
""")

st.write("Use CTR for:")
st.write("• Detecting **accidental click risk** (CTR spikes after layout changes)")
st.write("• Detecting **ad format changes** (CTR changes with sticky/interstitial/etc.)")
st.write("• Detecting **policy risk** (sudden CTR spikes can trigger scrutiny)")

# =============================
# TRENDS + TABS
# =============================
st.subheader("📈 Trends (Baseline → Today)")
st.altair_chart(line_chart(display_df, ["revenue","impressions","ecpm"], "Revenue vs Impressions vs eCPM"), use_container_width=True)
st.altair_chart(line_chart(display_df, ["ad_requests","fill_rate","requests_per_pageview"], "Requests vs Fill vs Requests/Pageview"), use_container_width=True)
st.altair_chart(line_chart(display_df, ["ctr","rpm"], "CTR (diagnostic) + RPM"), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Issues", "Delivery Detail", "Value Detail"])

with tab1:
    st.subheader("🧯 Top Issues (ranked)")
    metrics = [
        ("revenue","Revenue"),
        ("sessions","Sessions"),
        ("pageviews","Pageviews"),
        ("ad_requests","Ad Requests"),
        ("requests_per_pageview","Req/Pageview"),
        ("fill_rate","Fill Rate"),
        ("impressions","Impressions"),
        ("ecpm","eCPM"),
        ("rpm","RPM"),
        ("ctr","CTR"),
    ]
    rows = []
    for key, label in metrics:
        s = baseline_df[key].dropna().astype(float)
        med = float(s.median()) if len(s) else 0.0
        mad = float(np.median(np.abs(s - med))) if len(s) else 0.0
        val_today = float(t.get(key, 0.0))
        val_ref = float(ref.get(key, 0.0)) if ref else 0.0
        z = robust_z(val_today, med, mad)
        conf = confidence_from_z(z)
        delta = pct_change(val_today, val_ref) if val_ref else 0.0
        score = (abs(z) * 2 if z < 0 else 0) + (abs(delta) / 10 if delta < 0 else 0)
        rows.append([label, val_today, val_ref, delta, med, z, conf, score])

    issues = pd.DataFrame(rows, columns=["Metric","Today","Compare","Δ %","Baseline Median","Robust Z","Confidence %","Score"])
    issues = issues.sort_values("Score", ascending=False).drop(columns=["Score"]).head(10)
    st.dataframe(issues, use_container_width=True)

with tab2:
    st.subheader("🚚 Delivery Detail")
    st.write("Focus when impressions drop is the main driver:")
    st.write("• Requests and Requests/Pageview (ad loading)")
    st.write("• Fill rate (demand/floors/blocks)")
    st.altair_chart(line_chart(display_df, ["ad_requests","requests_per_pageview","fill_rate","impressions"], "Delivery pipeline"), use_container_width=True)

with tab3:
    st.subheader("💰 Value Detail")
    st.write("Focus when eCPM drop is the main driver:")
    st.write("• eCPM trend + RPM trend")
    st.write("• If you later add geo/device columns, we can show top segments causing the drop")
    st.altair_chart(line_chart(display_df, ["ecpm","rpm","revenue"], "Value pipeline"), use_container_width=True)

st.caption("Root cause is computed from metric decrements: Revenue → (Impressions + eCPM) and Impressions → (Requests + Fill). CTR is diagnostic, not a revenue driver.")
