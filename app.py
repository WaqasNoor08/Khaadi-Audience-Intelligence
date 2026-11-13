"""Streamlit dashboard - Khaadi Audience & Campaign Intelligence Engine (Project 3).

Run:  streamlit run app.py     (after  python pipeline.py)
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Khaadi Audience Intelligence", page_icon="👗", layout="wide")
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

st.title("👗 Khaadi Audience & Campaign Intelligence Engine")
st.caption("Project 3 · Big Data Analytics · builds on Project 1 (comment sentiment) and Project 2 (price-tier budget). "
           "Data: public YouTube comments on Khaadi videos - an observational sample, not Khaadi's own analytics.")

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
        st.altair_chart(alt.Chart(g).mark_bar().encode(
            x=alt.X("content_type:N", sort="-y", title=None), y=alt.Y(f"{m}:Q"),
            tooltip=list(g.columns)).properties(height=320), width="stretch")
        st.dataframe(g.round(3), width="stretch", hide_index=True)
        small = g[g["posts"] < 5]["content_type"].tolist()
        if small:
            st.warning("Fewer than 5 posts, treat as anecdotal: " + ", ".join(small))
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
    a.altair_chart(alt.Chart(t).mark_bar().encode(
        y=alt.Y("topic:N", sort="-x", title=None), x="comments:Q", tooltip=["topic", "comments", "share_pct"]
    ).properties(height=330), width="stretch")
    lg = cf["language_type"].value_counts().rename_axis("language").reset_index(name="comments")
    b.altair_chart(alt.Chart(lg).mark_arc(innerRadius=50).encode(
        theta="comments:Q", color="language:N", tooltip=["language", "comments"]).properties(height=330), width="stretch")
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
        color=alt.Color("unusual:N", scale=alt.Scale(domain=[True, False], range=["#d62728", "#9aa5b1"])),
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
        x=alt.X("x:Q", title="PCA 1"), y=alt.Y("y:Q", title="PCA 2"), color="segment:N",
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
