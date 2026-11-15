"""Streamlit dashboard - Khaadi Audience & Campaign Intelligence Engine (Project 3).

Run:  streamlit run app.py     (after  python pipeline.py)
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KhaadiPulse · Audience Intelligence", page_icon="👗", layout="wide")
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,600&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(900px 500px at 85% -5%,rgba(232,70,124,.22),transparent 60%),
        radial-gradient(800px 480px at -5% 8%,rgba(246,185,59,.16),transparent 60%),#0E0B1F;}
.block-container {padding-top:1.4rem; max-width:1180px;}
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
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)
PAL = ["#F6B93B", "#E8467C", "#27C4B0", "#8E7CFF", "#FF8A5B", "#7BD88F", "#4DA3FF", "#F28BB8", "#C9B6FF"]
GITHUB = "https://github.com/WaqasNoor08/Khaadi-Audience-Intelligence"
VERCEL = "https://khaadi-audience-intelligence-web-8w.vercel.app/"
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
c, v = D["comments_enriched"], D["video_features"]

st.markdown(f"""
<div class="kp-brand">Khaadi<b>Pulse</b></div>
<div class="kp-hero">
<h1>The shoppers under Khaadi's videos are <em>telling you</em> what to launch next.</h1>
<p>Audience segments, unusual posts and marketing ideas from public YouTube comments on Khaadi clothing videos.
Project 3 · Big Data Analytics · builds on Project 1 (comment sentiment) and Project 2 (price-tier budget).
An observational sample, not Khaadi's own analytics.</p>
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
                "5 · What to do next", "About / limits"])

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
            x=alt.X("content_type:N", sort="-y", title=None), y=alt.Y(f"{m}:Q"),
            tooltip=list(g.columns)).properties(height=320), width="stretch")
        st.dataframe(g.round(3), width="stretch", hide_index=True)
        small = g[g["posts"] < 10]["content_type"].tolist()
        if small:
            st.warning("Fewer than 10 posts, treat as anecdotal (small sample): " + ", ".join(small))
    eid = vf.groupby("eid_campaign").agg(posts=("post_id", "count"), comments=("comments", "sum"),
                                         median_views=("views", "median"),
                                         median_comments_per_1k_views=("comments_per_1k_views", "median")).reset_index()
    eid["eid_campaign"] = eid["eid_campaign"].map({True: "Eid campaign", False: "Other"})
    st.markdown("**Eid vs other campaigns**")
    st.dataframe(eid, width="stretch", hide_index=True)

# ------------------------------------------------------- 2 audience ----
with tabs[1]:
    st.subheader("What does the audience want?")
    a, b = st.columns([3, 2])
    t = cf["topic"].value_counts().rename_axis("topic").reset_index(name="comments")
    t["share_pct"] = (100 * t["comments"] / max(len(cf), 1)).round(1)
    a.altair_chart(alt.Chart(t).mark_bar(color="#E8467C").encode(
        y=alt.Y("topic:N", sort="-x", title=None), x="comments:Q", tooltip=["topic", "comments", "share_pct"]
    ).properties(height=330), width="stretch")
    lg = cf["language_type"].value_counts().rename_axis("language").reset_index(name="comments")
    b.altair_chart(alt.Chart(lg).mark_arc(innerRadius=50).encode(
        theta="comments:Q", color=alt.Color("language:N", scale=alt.Scale(range=PAL)), tooltip=["language", "comments"]).properties(height=330), width="stretch")
    st.markdown("**Topic by language** (share of each language's comments)")
    ct = pd.crosstab(cf["topic"], cf["language_type"], normalize="columns").mul(100).round(1)
    st.dataframe(ct, width="stretch")
    topic = st.selectbox("See example comments for a topic", t["topic"].tolist() if len(t) else [])
    if topic:
        st.dataframe(cf[cf["topic"] == topic][["comment", "language_type", "sentiment", "video_title"]].head(25),
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
    st.altair_chart(sc, width="stretch")
    un = vf[vf["unusual"]].sort_values("max_abs_z", ascending=False)
    st.markdown(f"**{len(un)} unusual posts** of {len(el)} checked")
    st.dataframe(un[["post_id", "video_title", "video_url", "content_type", "views", "days_since_post", "comments", "pos_rate",
                     "neg_rate", "question_rate", "why_unusual"]].round(2),
                 width="stretch", hide_index=True,
                 column_config={"video_url": st.column_config.LinkColumn("video", display_text="open")})

# --------------------------------------------------------- 4 segments ----
with tabs[3]:
    st.subheader("Which audience needs should the brand address?")
    cs, ex, cm = D["cluster_summary"], D["cluster_examples"], D["cluster_map"]
    rs = D["run_summary"]["value"]
    st.write(f"TF-IDF + SVD + KMeans on {int(rs['clustered'])} comments with at least two content words, k = {int(rs['k'])} chosen by silhouette. "
             f"Silhouette is low ({rs['silhouette_k' + str(int(rs['k']))]:.2f}): short mixed-language comments do not separate cleanly, "
             "so treat segments as exploratory.")
    st.dataframe(cs[["segment", "comments", "share_pct", "actionable_pct", "question_pct", "positive_pct", "negative_pct",
                     "roman_urdu_pct", "top_terms"]], width="stretch", hide_index=True)
    st.altair_chart(alt.Chart(cm).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("x:Q", title="PCA 1"), y=alt.Y("y:Q", title="PCA 2"), color=alt.Color("segment:N", scale=alt.Scale(range=PAL)),
        tooltip=["segment", "comment"]).properties(height=420), width="stretch")
    seg = st.selectbox("Example comments for a segment", ex["segment"].unique())
    st.table(ex[ex["segment"] == seg][["comment"]])

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
    st.dataframe(bp, width="stretch", hide_index=True)
    disc = 100 * (c["topic"] == "Discount / sale").mean()
    price = 100 * (c["topic"] == "Price inquiry").mean()
    st.write(f"Project 2 showed no significant sentiment difference between price tiers. Here, {price:.1f}% of comments ask for prices and "
             f"{disc:.1f}% ask about discounts, so price visibility and sale clarity matter in every tier, "
             "especially Budget and Mid-range where sale content is common.")

# ---------------------------------------------------------------- about ----
with tabs[5]:
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
<div class="kp-made">F2023376280 · Section A2 · UMT Lahore</div>
<div class="kp-links"><a href="https://github.com/WaqasNoor08" target="_blank">GitHub profile</a><a href="{GITHUB}" target="_blank">Project code</a><a href="{VERCEL}" target="_blank">Vercel version</a></div>
</div>
</div>
<div class="kp-foot">© 2026 KhaadiPulse · Made by Waqas Noor · Student project, not affiliated with Khaadi.</div>
""", unsafe_allow_html=True)
