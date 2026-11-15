"""Streamlit dashboard - Khaadi Audience & Campaign Intelligence Engine (Project 3).

Run:  streamlit run app.py     (after  python pipeline.py)
"""
from pathlib import Path

import json

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Khaadi Audience Intelligence", page_icon="👗", layout="wide")
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,600&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(900px 500px at 85% -5%,rgba(232,70,124,.22),transparent 60%),
        radial-gradient(800px 480px at -5% 8%,rgba(246,185,59,.16),transparent 60%),#0E0B1F;}
.block-container {padding-top:3.4rem; max-width:1180px;}
.kp-brand {font-family:'Fraunces',Georgia,serif; font-size:1.5rem; font-weight:700; letter-spacing:-.02em;}
.kp-brand b {background:linear-gradient(90deg,#F6B93B,#E8467C); -webkit-background-clip:text; background-clip:text; color:transparent;}
.kp-hero h1 {font-family:'Fraunces',Georgia,serif; font-size:clamp(1.9rem,4.5vw,3rem); line-height:1.08; letter-spacing:-.025em; margin:.3rem 0 .6rem; padding:0;}
.kp-hero h1 em {font-style:italic; background:linear-gradient(90deg,#F6B93B,#E8467C 60%,#b58bff); -webkit-background-clip:text; background-clip:text; color:transparent;}
.kp-hero p {color:#d8d2ee; max-width:62ch; font-size:1.02rem;}
.kp-pills {display:flex; gap:8px; flex-wrap:wrap; margin:.6rem 0 1rem;}
.kp-pills a {padding:8px 16px; border-radius:999px; text-decoration:none; font-weight:600; font-size:.9rem; border:1px solid #2E2753; color:#F7F1E6;}
.kp-pills a.pri {background:linear-gradient(90deg,#F6B93B,#E8467C); color:#1a1030; border:0;}
[data-testid="stMetric"] {background:#1B1636; border:1px solid #2E2753; border-radius:14px; padding:12px 14px;}
[data-testid="stMetricValue"] {font-family:'Fraunces',Georgia,serif;}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {border-radius:999px; padding:6px 14px; background:#15112B;}
.stTabs [aria-selected="true"] {background:linear-gradient(90deg,#F6B93B,#E8467C); color:#1a1030 !important; font-weight:700;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
h2, h3 {font-family:'Fraunces',Georgia,serif; letter-spacing:-.01em;}
.kp-maker {margin:2rem 0 .5rem; display:flex; gap:18px; align-items:center; padding:20px; border-radius:20px; flex-wrap:wrap;
  background:linear-gradient(135deg,rgba(246,185,59,.12),rgba(232,70,124,.14) 55%,rgba(39,196,176,.12)); border:1px solid #2E2753;}
.kp-pic {width:92px; height:92px; border-radius:50%; padding:3px; flex:none; background:conic-gradient(#F6B93B,#E8467C,#27C4B0,#F6B93B);}
.kp-pic img {width:100%; height:100%; border-radius:50%; object-fit:cover; display:block; border:3px solid #0E0B1F;}
.kp-made {color:#A8A1C4; font-size:.82rem;}
.kp-name {font-family:'Fraunces',Georgia,serif; font-size:1.35rem; font-weight:700;}
.kp-links a {display:inline-block; margin:10px 8px 0 0; padding:7px 16px; border-radius:999px; border:1px solid #2E2753; color:#F7F1E6; text-decoration:none; font-weight:600; font-size:.85rem;}
.kp-foot {color:#A8A1C4; font-size:.78rem; text-align:center; margin:.8rem 0 1rem;}

.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {color:#F7F1E6;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {color:#F7F1E6 !important;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#15112B;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color:#cfc8ea !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#F6B93B !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color:#bdb6da !important;}
.stTabs [data-baseweb="tab"] p {color:#cfc8ea !important;}
.stTabs [aria-selected="true"] p {color:#1a1030 !important;}
[data-baseweb="select"] > div {background:#1B1636 !important; color:#F7F1E6 !important; border-color:#2E2753 !important;}
[data-baseweb="select"] * {color:#F7F1E6 !important;}
[data-baseweb="tag"] {background:#E8467C !important;}
[data-baseweb="tag"] * {color:#1a1030 !important;}
.kp-name {color:#F7F1E6 !important;}
.kp-made {color:#cfc8ea !important;}
.kp-links a, .kp-pills a {color:#F7F1E6 !important;}
.kp-pills a.pri {color:#1a1030 !important;}
.kp-tw {overflow-x:auto; background:#1B1636; border:1px solid #2E2753; border-radius:14px; margin:.5rem 0 1rem;}
table.kp-table {border-collapse:collapse; width:100%; font-size:.84rem; color:#F7F1E6;}
table.kp-table th {color:#F6B93B; text-align:left; padding:9px 10px; border-bottom:1px solid #2E2753; white-space:nowrap; background:#1B1636;}
table.kp-table td {padding:8px 10px; border-bottom:1px solid #2E2753; vertical-align:top; color:#F7F1E6;}
table.kp-table td a {color:#ffd98a !important;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)
PAL = ["#F6B93B", "#E8467C", "#27C4B0", "#8E7CFF", "#FF8A5B", "#7BD88F", "#4DA3FF", "#F28BB8", "#C9B6FF"]
GITHUB = "https://github.com/WaqasNoor08/Khaadi-Audience-Intelligence"
VERCEL = "https://khaadi-audience-intelligence-web-8w.vercel.app/"
BRAND_A, BRAND_B = "Khaadi ", "Insights"


def _kp_theme():
    return {"config": {"background": "#1B1636", "view": {"stroke": "transparent"},
                       "axis": {"labelColor": "#cfc8ea", "titleColor": "#cfc8ea", "gridColor": "#2E2753",
                                "domainColor": "#2E2753", "tickColor": "#2E2753"},
                       "legend": {"labelColor": "#cfc8ea", "titleColor": "#cfc8ea"}}}


alt.themes.register("khaadi", _kp_theme)
alt.themes.enable("khaadi")


def show(df, index=False, link_cols=None, **_ignored):
    """Dark, always-readable HTML table (replaces st.dataframe / st.table)."""
    import html as _h
    df = df.copy()
    if index:
        df = df.reset_index()
    df = df.fillna("")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].map(lambda x: _h.escape(str(x)))
    for col, label in (link_cols or {}).items():
        df[col] = df[col].map(lambda u: f'<a href="{u}" target="_blank" rel="noopener">{label}</a>' if u else "")
    st.markdown('<div class="kp-tw">' + df.to_html(index=False, escape=False, border=0, classes="kp-table") + "</div>",
                unsafe_allow_html=True)

BASE = Path(__file__).parent
OUT, DATA = BASE / "output", BASE / "data"


@st.cache_data
def load():
    need = ["comments_enriched", "video_features", "content_type_summary", "cluster_summary",
            "cluster_examples", "cluster_map", "recommendations", "run_summary"]
    missing = [n for n in need if not (OUT / f"{n}.csv").exists()]
    if missing:
        st.error("Run `python pipeline.py` first - missing: " + ", ".join(missing))
        st.stop()
    return {n: pd.read_csv(OUT / f"{n}.csv", index_col=0 if n == "run_summary" else None) for n in need}


D = load()


def _opt(name):
    f = OUT / name
    return pd.read_csv(f).fillna("") if f.exists() else None


LEADS, LSUM, TPL, HCT, HTR, ALR, PLAN = [_opt(n) for n in (
    "leads.csv", "lead_summary.csv", "reply_templates.csv", "health_by_content.csv", "health_trend.csv", "alerts.csv", "weekly_plan.csv")]
IM = json.loads((OUT / "insights_meta.json").read_text(encoding="utf-8")) if (OUT / "insights_meta.json").exists() else {}
c, v = D["comments_enriched"], D["video_features"]

st.markdown(f"""
<div class="kp-brand">{BRAND_A}<b>{BRAND_B}</b></div>
<div class="kp-hero">
<h1>The shoppers under Khaadi's videos are <em>telling you</em> what to launch next.</h1>
<p>Audience segments, unusual posts and marketing ideas from public YouTube comments on Khaadi clothing videos.
Project 3 · Big Data Analytics · builds on Project 1 (comment sentiment) and Project 2 (price-tier budget).
An observational sample, not Khaadi's own analytics. Data updated {IM.get('updated', '-')}.</p>
</div>
<div class="kp-pills"><a class="pri" href="{VERCEL}" target="_blank">Static version on Vercel</a><a href="{GITHUB}" target="_blank">GitHub</a></div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------ sidebar ----
st.sidebar.header("Filters")
cats = st.sidebar.multiselect("Product category", sorted(c["product_category"].unique()), default=sorted(c["product_category"].unique()))
cts = st.sidebar.multiselect("Content type", sorted(c["content_type"].unique()), default=sorted(c["content_type"].unique()))
langs = st.sidebar.multiselect("Comment language", sorted(c["language_type"].unique()), default=sorted(c["language_type"].unique()))
cf = c[c["product_category"].isin(cats) & c["content_type"].isin(cts) & c["language_type"].isin(langs)]
vf = v[v["product_category"].isin(cats) & v["content_type"].isin(cts)]
st.sidebar.metric("Comments in view", f"{len(cf):,}")
st.sidebar.metric("Posts (videos) in view", f"{cf['post_id'].nunique():,}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Comments", f"{len(cf):,}")
k2.metric("Posts", cf["post_id"].nunique())
k3.metric("Positive", f"{100 * (cf['sentiment'] == 'Positive').mean():.1f}%" if len(cf) else "-")
k4.metric("Buying-signal comments", f"{100 * cf['actionable'].mean():.1f}%" if len(cf) else "-")

tabs = st.tabs(["1 · Content", "2 · What the audience wants", "3 · Unusual posts", "4 · Audience segments",
                "5 · What to do next", "Leads", "Ready replies", "Brand health", "Alerts", "Weekly plan",
                "About / limits"])

# ---------------------------------------------------------- 1 content ----
with tabs[0]:
    st.subheader("Which content attracts attention?")
    st.write("Attention is measured with video views, video likes and comments (YouTube does not expose shares). "
             "Medians are used because a few very large videos would dominate any average.")
    g = (vf.groupby("content_type")
         .agg(posts=("post_id", "count"), comments=("comments", "sum"), median_views=("views", "median"),
              median_comments_per_1k_views=("comments_per_1k_views", "median"), median_like_rate=("like_rate", "median"),
              median_likes_per_comment=("likes_per_comment", "median"),
              median_buying_signal=("actionable_rate", "median"), median_positive=("pos_rate", "median"))
         .reset_index())
    if len(g):
        m = st.selectbox("Metric", ["median_comments_per_1k_views", "median_views", "median_like_rate", "comments",
                                    "median_likes_per_comment", "median_buying_signal", "median_positive"])
        st.altair_chart(alt.Chart(g).mark_bar(color="#F6B93B").encode(
            x=alt.X("content_type:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=220)), y=alt.Y(f"{m}:Q"),
            tooltip=list(g.columns)).properties(height=320), width="stretch", theme=None)
        show(g.round(3), width="stretch", hide_index=True)
        small = g[g["posts"] < 10]["content_type"].tolist()
        if small:
            st.warning("Fewer than 10 posts, treat as anecdotal (small sample): " + ", ".join(small))
    eid = vf.groupby("eid_campaign").agg(posts=("post_id", "count"), comments=("comments", "sum"),
                                         median_views=("views", "median"),
                                         median_comments_per_1k_views=("comments_per_1k_views", "median")).reset_index()
    eid["eid_campaign"] = eid["eid_campaign"].map({True: "Eid campaign", False: "Other"})
    st.markdown("**Eid vs other campaigns**")
    show(eid, width="stretch", hide_index=True)

# ------------------------------------------------------- 2 audience ----
with tabs[1]:
    st.subheader("What does the audience want?")
    a, b = st.columns([3, 2])
    t = cf["topic"].value_counts().rename_axis("topic").reset_index(name="comments")
    t["share_pct"] = (100 * t["comments"] / max(len(cf), 1)).round(1)
    a.altair_chart(alt.Chart(t).mark_bar(color="#E8467C").encode(
        y=alt.Y("topic:N", sort="-x", title=None), x="comments:Q", tooltip=["topic", "comments", "share_pct"]
    ).properties(height=330), width="stretch", theme=None)
    lg = cf["language_type"].value_counts().rename_axis("language").reset_index(name="comments")
    b.altair_chart(alt.Chart(lg).mark_arc(innerRadius=50).encode(
        theta="comments:Q", color=alt.Color("language:N", scale=alt.Scale(range=PAL)), tooltip=["language", "comments"]).properties(height=330), width="stretch", theme=None)
    st.markdown("**Topic by language** (share of each language's comments)")
    ct = pd.crosstab(cf["topic"], cf["language_type"], normalize="columns").mul(100).round(1)
    show(ct, index=True)
    topic = st.selectbox("See example comments for a topic", t["topic"].tolist() if len(t) else [])
    if topic:
        show(cf[cf["topic"] == topic][["comment", "language_type", "sentiment", "video_title"]].head(25),
                     width="stretch", hide_index=True)
    st.caption("Topics come from transparent keyword rules (see pipeline.py). They are a first pass: about "
               f"{100 * (c['topic'] == 'Other / general').mean():.0f}% of comments fit no rule.")

# --------------------------------------------------------- 3 unusual ----
with tabs[2]:
    st.subheader("Which marketing patterns are unusual?")
    st.write("Each post is compared with posts of the **same content type** (robust z-score using the median and MAD) and with an "
             "Isolation Forest over seven engagement/comment metrics. Posts need at least 10 collected comments and 1,000 views. "
             "Flags are leads for investigation, not proof of a problem.")
    el = vf[vf["eligible"]]
    sc = alt.Chart(el).mark_circle(size=90).encode(
        x=alt.X("views:Q", scale=alt.Scale(type="log"), title="Video views (log scale)"),
        y=alt.Y("pos_rate:Q", title="Share positive"),
        color=alt.Color("unusual:N", scale=alt.Scale(domain=[True, False], range=["#E8467C", "#8a83b5"])),
        tooltip=["post_id", "video_title", "content_type", "views", "comments", "why_unusual"]).properties(height=360)
    st.altair_chart(sc, width="stretch", theme=None)
    un = vf[vf["unusual"]].sort_values("max_abs_z", ascending=False)
    st.markdown(f"**{len(un)} unusual posts** of {len(el)} checked")
    show(un[["post_id", "video_title", "video_url", "content_type", "views", "days_since_post", "comments", "pos_rate",
                     "neg_rate", "question_rate", "why_unusual"]].round(2),
                 width="stretch", hide_index=True,
                 link_cols={"video_url": "open"})

# --------------------------------------------------------- 4 segments ----
with tabs[3]:
    st.subheader("Which audience needs should the brand address?")
    cs, ex, cm = D["cluster_summary"], D["cluster_examples"], D["cluster_map"]
    rs = D["run_summary"]["value"]
    st.write(f"TF-IDF + SVD + KMeans on {int(rs['clustered'])} comments with at least two content words, k = {int(rs['k'])} chosen by silhouette. "
             f"Silhouette is low ({rs['silhouette_k' + str(int(rs['k']))]:.2f}): short mixed-language comments do not separate cleanly, "
             "so treat segments as exploratory.")
    show(cs[["segment", "comments", "share_pct", "actionable_pct", "question_pct", "positive_pct", "negative_pct",
                     "roman_urdu_pct", "top_terms"]], width="stretch", hide_index=True)
    st.altair_chart(alt.Chart(cm).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("x:Q", title="PCA 1"), y=alt.Y("y:Q", title="PCA 2"), color=alt.Color("segment:N", scale=alt.Scale(range=PAL)),
        tooltip=["segment", "comment"]).properties(height=420), width="stretch", theme=None)
    seg = st.selectbox("Example comments for a segment", ex["segment"].unique())
    show(ex[ex["segment"] == seg][["comment"]])

# ------------------------------------------------------ 5 recommendations ----
with tabs[4]:
    st.subheader("What should the marketing team do next?")
    r = D["recommendations"]
    for _, row in r.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['area']}** - {row['recommendation']}")
            st.caption(f"Evidence: {row['evidence']}  ·  Confidence: {row['confidence']}")
    st.markdown("### Link with Project 2 (price tiers & budget)")
    bp = pd.read_csv(DATA / "budget_plan.csv")
    show(bp, width="stretch", hide_index=True)
    disc = 100 * (c["topic"] == "Discount / sale").mean()
    price = 100 * (c["topic"] == "Price inquiry").mean()
    st.write(f"Project 2 showed no significant sentiment difference between price tiers. Here, {price:.1f}% of comments ask for prices and "
             f"{disc:.1f}% ask about discounts, so price visibility and sale clarity matter in every tier, "
             "especially Budget and Mid-range where sale content is common.")

# -------------------------------------------------------------- leads ----
with tabs[5]:
    st.subheader("Who is ready to buy?")
    if LEADS is None:
        st.info("Run `python pipeline.py` to create the lead list.")
    else:
        st.write(f"{IM.get('leads_total', len(LEADS)):,} comments show a buying signal ({IM.get('leads_hot', 0):,} hot: order, contact or delivery). "
                 f"Spam, links and phone numbers are removed. Below: the {len(LEADS)} most useful (questions first, then most liked).")
        c1, c2 = st.columns(2)
        pr = c1.selectbox("Priority", ["All"] + sorted(LEADS["priority"].unique()))
        tp = c2.selectbox("Topic", ["All"] + sorted(LEADS["topic"].unique()))
        f = LEADS
        if pr != "All":
            f = f[f["priority"] == pr]
        if tp != "All":
            f = f[f["topic"] == tp]
        st.caption(f"{len(f)} comments in this view" + (" (showing 100)" if len(f) > 100 else ""))
        show(f.head(100)[["priority", "comment", "topic", "language", "likes", "video", "url"]], link_cols={"url": "open"})

# ------------------------------------------------------ ready replies ----
with tabs[6]:
    st.subheader("Ready replies")
    st.write("Draft answers for the most common questions, in English and Roman Urdu. Fill every [BRACKET] with real details "
             "(price, dates, links) before posting. These are drafts, not approved Khaadi wording. Use the copy icon in each box.")
    if TPL is None:
        st.info("Run `python pipeline.py` to create the reply drafts.")
    else:
        for _, t in TPL.iterrows():
            with st.container(border=True):
                st.markdown(f"**{t['topic']}**")
                a, b = st.columns(2)
                a.caption("English")
                a.code(t["english"], language=None, wrap_lines=True)
                b.caption("Roman Urdu")
                b.code(t["roman_urdu"], language=None, wrap_lines=True)

# -------------------------------------------------------- brand health ----
with tabs[7]:
    st.subheader("Brand health")
    h = IM.get("health", {})
    if not h:
        st.info("Run `python pipeline.py` to calculate brand health.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Health score (0-100)", f"{h['score']:.0f}", IM.get("health_label", ""), delta_color="off")
        m2.metric("Positive", f"{h['positive_pct']:.1f}%")
        m3.metric("Negative", f"{h['negative_pct']:.1f}%")
        m4.metric("Complaints", f"{h['complaint_pct']:.1f}%")
        st.caption("Score = 50 + (positive % - negative %) / 2 - complaint % / 2, kept between 0 and 100. "
                   "Sentiment is VADER, which under-scores Roman Urdu, so read it as a trend, not an exact grade.")
        st.markdown("**By content type**")
        st.altair_chart(alt.Chart(HCT).mark_bar(color="#F6B93B").encode(
            x=alt.X("content_type:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=220)),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="health score"),
            tooltip=list(HCT.columns)).properties(height=300), width="stretch", theme=None)
        st.markdown("**By month (comment date, months with at least 100 comments)**")
        st.altair_chart(alt.Chart(HTR).mark_line(point=True, color="#E8467C").encode(
            x=alt.X("month:N", title=None), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="health score"),
            tooltip=list(HTR.columns)).properties(height=300), width="stretch", theme=None)

# -------------------------------------------------------------- alerts ----
with tabs[8]:
    st.subheader("Alerts")
    st.write("Rule-based flags from the latest data: rising topics, videos with unusually many negative comments and videos with many "
             "buying-signal comments. They are leads to check, not proof of a problem.")
    if ALR is None or not len(ALR):
        st.info("No alerts in the latest data.")
    else:
        for _, a in ALR.iterrows():
            with st.container(border=True):
                st.markdown(f"**{a['severity']}** · **{a['title']}**")
                link = f" · [open video]({a['url']})" if a["url"] else ""
                st.caption(f"{a['type']} · {a['detail']}{link}")

# --------------------------------------------------------- weekly plan ----
with tabs[9]:
    st.subheader("Weekly plan")
    st.write("Five actions for the marketing team, built from the numbers on the other tabs. Treat them as hypotheses to test.")
    if PLAN is None:
        st.info("Run `python pipeline.py` to create the weekly plan.")
    else:
        for _, p_ in PLAN.iterrows():
            with st.container(border=True):
                st.markdown(f"**{p_['step']}** - {p_['action']}")
                link = f" · [open video]({p_['url']})" if p_["url"] else ""
                st.caption(f"Why: {p_['why']}{link}")

# ---------------------------------------------------------------- about ----
with tabs[10]:
    st.markdown("""
**Data.** Public YouTube comments from Project 1, collected 12 Nov 2026 with the YouTube Data API v3 from Khaadi *clothing* videos
only (perfume, "Khadi" herbal/stationery and IPO videos are filtered out). Video views, likes and publish dates come from the same API.
All records are real; none are simulated.

**Limits.** One platform (YouTube); shares are not available; some videos feature several brands (Khaadi only appears in the title);
topic rules and language detection are keyword based and Roman Urdu spelling varies; VADER under-scores Roman Urdu.
Recommendations are hypotheses to test, not proven effects.
""")

# --------------------------------------------------------------- footer ----
st.markdown(f"""
<div class="kp-maker">
<div class="kp-pic"><img src="https://github.com/WaqasNoor08.png?size=240" alt="Waqas Noor"></div>
<div>
<div class="kp-made">Made by</div>
<div class="kp-name">Waqas Noor</div>
<div class="kp-links"><a href="https://github.com/WaqasNoor08" target="_blank">GitHub profile</a><a href="{GITHUB}" target="_blank">Project code</a><a href="{VERCEL}" target="_blank">Vercel version</a></div>
</div>
</div>
<div class="kp-foot">© 2026 {BRAND_A}{BRAND_B} · Made by Waqas Noor · Student project, not affiliated with Khaadi.</div>
""", unsafe_allow_html=True)
