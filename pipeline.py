"""Project 3 - Pakistan Social Audience & Campaign Intelligence Engine (Khaadi).

Reads the YouTube comments collected in Project 1 (VADER sentiment already attached)
and produces everything the dashboard needs:

  1. content-type comparison          -> output/content_type_summary.csv
  2. audience topics / language mix   -> output/comments_enriched.csv
  3. unusual posts (anomaly detection)-> output/video_features.csv
  4. audience segments (clustering)   -> output/cluster_summary.csv, cluster_examples.csv, cluster_map.csv
  5. evidence-based recommendations   -> output/recommendations.csv

Run:  python pipeline.py                 (Khaadi CLOTHING videos only, default)
      python pipeline.py --include-perfume (Khaadi clothing + perfume)
      python pipeline.py --include-all   (keep every video from Project 1)
"""
import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import Normalizer

BASE = Path(__file__).parent
DATA, OUT = BASE / "data", BASE / "output"
SEED = 42
COLLECTED = "2026-11-03"   # fallback collection date for old-format files without a collection_date column

# ---------------------------------------------------------------- rules ----
ROMAN_STRONG = set("""hai hain nahi nahin kya kitne kitna kitni kab kahan mein bohat bahut boht bht bohot acha achi
achay zabardast bata batao btao bta dein dain krdo karo kro apki aapki yeh wala wali hogi hoga milega shukriya
jazakallah mashallah masha kesy kese kaisa kaise liye lye tha thi hy sy""".split())
ROMAN_WEAK = set("ka ki ke ko kr dy hi bhi aur se pe sab ap aap wo woh".split())
URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
OTHER_SCRIPT = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F]")

TOPIC_RULES = [  # order = priority
    ("Complaint", r"ghatya|worst|waste|fake|scam|\bbad\b|poor|overpriced|mehng|expensive|disappoint|fraud|cheat|not good|hate"),
    ("Delivery / order status", r"deliver|shipping|courier|parcel|\bcod\b|received|waiting|tracking|ordered|my order"),
    ("Price inquiry", r"price|\brate\b|kitne|kitna|kitni|qeemat|\bcost\b|how much|\brs\b|\bpkr\b|rupee|\d+\s*/-"),
    ("Discount / sale", r"discount|\bsale\b|\boff\b|offer|promo|voucher|\bcode\b|sale khatam|\d+\s*%"),
    ("Availability / location", r"available|availab|in stock|stock|milega|restock|branch|outlet|store|karachi|lahore|islamabad|"
                                r"abroad|\buk\b|\busa\b|international|pakistan se|ship to|where (can|to)|kahan"),
    ("Quality / fabric", r"fabric|quality|material|cotton|lawn|stitch|shrink|fade|wash|durable|texture|smell|fragrance|scent|long.?lasting"),
    ("Sizing / fit", r"\bsize\b|sizes|\bfit\b|fitting|\bsmall\b|\blarge\b|medium|length|measurement"),
    ("Purchase intent / contact", r"want to buy|will buy|buying|bought|need this|i want|inbox|\bdm\b|contact|whatsapp|\bnumber\b|link|"
                                  r"order kar|buy kar|lena hai|chahiye"),
    ("Praise", r"beautiful|love|nice|gorgeous|amazing|mashallah|masha allah|pretty|\bbest\b|stunning|great|\bgood\b|lovely|"
               r"perfect|awesome|wow|elegant|zabardast|khoobsurat|acha|achi|slay|ma ?sha ?-?allah|mash?aa?llah|subhan ?allah"),
]
TOPIC_RULES = [(t, re.compile(p, re.I)) for t, p in TOPIC_RULES]
ACTION_TOPICS = {"Price inquiry", "Discount / sale", "Availability / location", "Delivery / order status",
                 "Sizing / fit", "Purchase intent / contact"}

CONTENT_RULES = [
    ("Haul / try-on", r"haul|try.?on"),
    ("Sale announcement", r"sale|% ?off|discount|flat \d+|dhamaka"),
    ("Unboxing / PR package", r"unbox|pr package"),
    ("Collection showcase", r"collection|pret|new arrival|eid|signature|khaas|casual"),
    ("Review / comparison", r"review|comparison|comparision|honest|worth"),
]
CONTENT_RULES = [(t, re.compile(p, re.I)) for t, p in CONTENT_RULES]
STOP = set("""the and for you your this that with are was have has not but can will its just very too from they them what
when who how any all out one get got hai hain ka ki ke ko ho hi bhi aur se pe ye yeh wo woh to me mein ap aap kr kar
a an is it in on of at be as so if my me we do did does""".split())


def language_type(text: str) -> str:
    if URDU_SCRIPT.search(text):
        return "Urdu script"
    if OTHER_SCRIPT.search(text):
        return "Other script"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "Emoji / symbols only"
    strong = sum(w in ROMAN_STRONG for w in words)
    weak = sum(w in ROMAN_WEAK for w in words)
    return "Roman Urdu" if (strong >= 1 or weak >= 2) else "English"


def topic_of(text: str) -> str:
    for name, rx in TOPIC_RULES:
        if rx.search(text):
            return name
    return "Other / general"


def content_type(title: str) -> str:
    for name, rx in CONTENT_RULES:
        if rx.search(title):
            return name
    return "Other"


def product_category(title: str) -> str:
    """Project 1 now collects clothing videos only; this just guards against perfume/body-mist titles."""
    return "Perfume" if re.search(r"perfume|fragrance|attar|body ?mist|deo\b", title, re.I) else "Clothing"


def tokens(text: str) -> str:
    t = re.sub(r"https?://\S+|@\w+", " ", text.lower())
    return " ".join(w for w in re.findall(r"[a-z\u0600-\u06FF']{2,}", t) if w not in STOP)


# ------------------------------------------------------------- loaders ----
def load(include_all: bool, include_perfume: bool = False) -> pd.DataFrame:
    d = pd.read_csv(DATA / "comments_project1.csv")
    d["video_title"] = d["video_title"].map(html.unescape)
    d["comment"] = d["comment"].astype(str).map(html.unescape)
    d["published_at"] = pd.to_datetime(d["published_at"], errors="coerce", utc=True)
    for col in ("views", "video_likes", "video_comment_count"):
        if col not in d:
            d[col] = np.nan                      # old Project 1 format had no video statistics
    if "video_published_at" not in d:
        d["video_published_at"] = pd.NaT
    d["video_published_at"] = pd.to_datetime(d["video_published_at"], errors="coerce", utc=True)
    if "video_id" not in d:
        d["video_id"] = d["video_title"]
    if "video_url" not in d:
        d["video_url"] = ""
    if "collection_date" not in d:
        d["collection_date"] = COLLECTED
    n_all = len(d)
    if not include_all:
        # Project 1 searched "Khaadi" and "Khadi"; 'Khadi' is mostly an Indian herbal/stationery brand.
        keep = d["video_title"].str.contains("khaadi", case=False) & \
            ~d["video_title"].str.contains(r"\bIPO\b|go public|shares|stock", case=False)
        d = d[keep].copy()
    d["product_category"] = d["video_title"].map(product_category)
    if not include_all and not include_perfume:
        d = d[d["product_category"] == "Clothing"].copy()   # Khaadi CLOTHING only (perfume etc. dropped)
    print(f"Comments: {n_all} in Project 1 -> {len(d)} used")
    d["post_id"] = pd.Series(pd.factorize(d["video_id"])[0] + 1, index=d.index).map(lambda i: f"P{i:03d}")
    d["days_since_post"] = (pd.to_datetime(d["collection_date"], utc=True) - d["video_published_at"]).dt.days
    d["content_type"] = d["video_title"].map(content_type)
    d["eid_campaign"] = d["video_title"].str.contains("eid", case=False)
    d["language_type"] = d["comment"].map(language_type)
    d["topic"] = d["comment"].map(topic_of)
    d["is_question"] = d["comment"].str.contains(r"\?|\bkya\b|\bkitn|\bkab\b|\bkahan\b|\bhow\b|\bwhere\b|\bwhich\b|\bwhen\b|\bany\b", case=False)
    d["actionable"] = d["topic"].isin(ACTION_TOPICS)
    d["purchase_intent"] = d["topic"].map(lambda t: "Strong intent" if t in ("Purchase intent / contact", "Delivery / order status")
                                          else "Possible inquiry" if t in ("Price inquiry", "Availability / location", "Sizing / fit", "Discount / sale")
                                          else "No intent shown")
    d["campaign"] = np.where(d["eid_campaign"], "Eid collection",
                     np.where(d["content_type"] == "Sale announcement", "Sale / discount",
                     np.where(d["content_type"] == "Collection showcase", "New collection", "Creator content")))
    d["platform"] = "YouTube"
    d["data_provenance"] = "real - public YouTube comment via YouTube Data API v3 (Project 1); not simulated"
    d["tokens"] = d["comment"].map(tokens)
    return d.reset_index(drop=True)


# --------------------------------------------------------- 1. content ----
def content_summary(d: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby("content_type").agg(
        videos=("post_id", "count"), comments=("comments", "sum"),
        median_views=("views", "median"),
        median_comments_per_1k_views=("comments_per_1k_views", "median"),
        median_like_rate=("like_rate", "median"),
        median_likes_per_comment=("likes_per_comment", "median"),
        median_actionable_rate=("actionable_rate", "median"))
    s_ = d.groupby("content_type").agg(positive_pct=("sentiment", lambda x: 100 * (x == "Positive").mean()),
                                       negative_pct=("sentiment", lambda x: 100 * (x == "Negative").mean()))
    return g.join(s_).round(4).reset_index().sort_values("comments", ascending=False)


# ----------------------------------------------------- 3. video / anomaly ----
def video_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("post_id", sort=False)
    v = g.agg(video_title=("video_title", "first"), video_url=("video_url", "first"), platform=("platform", "first"),
              campaign=("campaign", "first"), content_type=("content_type", "first"),
              product_category=("product_category", "first"), eid_campaign=("eid_campaign", "first"),
              views=("views", "first"), video_likes=("video_likes", "first"),
              video_comment_count=("video_comment_count", "first"), days_since_post=("days_since_post", "first"),
              comments=("comment", "size"), total_likes=("likes", "sum"),
              pos_rate=("sentiment", lambda x: (x == "Positive").mean()),
              neg_rate=("sentiment", lambda x: (x == "Negative").mean()),
              avg_score=("score", "mean"), question_rate=("is_question", "mean"),
              actionable_rate=("actionable", "mean"),
              emoji_only_rate=("language_type", lambda x: (x == "Emoji / symbols only").mean())).reset_index()
    v["likes_per_comment"] = v["total_likes"] / v["comments"]
    v["like_rate"] = v["video_likes"] / v["views"]                                   # video likes per view
    v["comments_per_1k_views"] = v["video_comment_count"] / v["views"] * 1000       # all comments on the video
    v["shares"] = np.nan   # not exposed by the YouTube Data API
    return v


def robust_z(x: pd.Series, ref: pd.Series) -> pd.Series:
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826
    if mad == 0:
        mad = ref.std(ddof=0) or 1.0
    return (x - med) / mad


def flag_unusual(v: pd.DataFrame, min_comments: int = 10) -> pd.DataFrame:
    v = v.copy()
    has_views = v["views"].notna().any()
    v["log_views"] = np.log1p(v["views"]) if has_views else np.log1p(v["comments"])
    v["log_likes_pc"] = np.log1p(v["likes_per_comment"])
    if has_views:
        v["log_cpk"] = np.log1p(v["comments_per_1k_views"])
        metrics = ["log_views", "like_rate", "log_cpk", "neg_rate", "question_rate", "actionable_rate", "avg_score"]
    else:
        metrics = ["log_views", "log_likes_pc", "neg_rate", "question_rate", "actionable_rate", "avg_score"]
    elig = v["comments"] >= min_comments
    if has_views:
        elig &= v["views"] >= 1000        # ratios per view are meaningless for tiny videos
    v["eligible"] = elig
    zs = pd.DataFrame(index=v.index, columns=metrics, dtype=float)
    for ct, idx in v[elig].groupby("content_type").groups.items():
        peers = v.loc[idx]
        # compare with the same content type; if too few peers, compare with every eligible post
        ref_df = peers if len(peers) >= 5 else v[elig]
        for m in metrics:
            zs.loc[idx, m] = robust_z(v.loc[idx, m].fillna(ref_df[m].median()), ref_df[m].dropna())
    v["max_abs_z"] = zs.abs().max(axis=1)
    v["z_driver"] = ""
    v.loc[elig, "z_driver"] = zs.loc[elig].abs().idxmax(axis=1)
    v["max_abs_z"] = v["max_abs_z"].fillna(0)
    sub = v.loc[elig, metrics].fillna(v.loc[elig, metrics].median())
    X = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1)
    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=SEED).fit(X)
    v.loc[elig, "isolation_score"] = -iso.score_samples(X)
    v["isolation_flag"] = False
    v.loc[elig, "isolation_flag"] = iso.predict(X) == -1
    v["unusual"] = elig & ((v["max_abs_z"] >= 5) | (v["isolation_flag"] & (v["max_abs_z"] >= 3)))

    label = {"log_views": "views", "like_rate": "video likes per view", "log_cpk": "comments per 1,000 views",
             "log_likes_pc": "likes per comment",
             "neg_rate": "negative-comment share", "question_rate": "question share",
             "actionable_rate": "buying-signal share", "avg_score": "average sentiment"}

    def reason(r):
        if not r["unusual"]:
            return ""
        z = zs.loc[r.name, r["z_driver"]]
        return (f"{label[r['z_driver']]} much {'higher' if z > 0 else 'lower'} than comparable "
                f"'{r['content_type']}' posts (robust z = {z:+.1f})")
    v["why_unusual"] = v.apply(reason, axis=1)
    return v.drop(columns=[c for c in ("log_views", "log_likes_pc", "log_cpk") if c in v])


# ----------------------------------------------------- 4. clustering ----
def cluster_comments(d: pd.DataFrame):
    mask = d["tokens"].str.split().str.len() >= 2
    X_txt = d.loc[mask, "tokens"]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = tf.fit_transform(X_txt)
    svd = TruncatedSVD(n_components=min(40, X.shape[1] - 1), random_state=SEED)
    Z = Normalizer().fit_transform(svd.fit_transform(X))
    scores = {}
    for k in range(4, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        scores[k] = silhouette_score(Z, km.labels_, sample_size=min(5000, len(Z)), random_state=SEED)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=50, random_state=SEED).fit(Z)
    d["cluster"] = -1
    d.loc[mask, "cluster"] = km.labels_
    terms = np.array(tf.get_feature_names_out())
    rows, ex_rows = [], []
    lab = pd.Series(km.labels_, index=X_txt.index)
    for c in range(best_k):
        idx = lab.index[lab == c]
        sub = d.loc[idx]
        pos = np.where(lab.values == c)[0]
        cen = np.asarray(X[pos].mean(axis=0)).ravel()
        top = ", ".join(terms[np.argsort(cen)[::-1][:7]])
        dom = sub["topic"].value_counts()
        act = sub.loc[sub["actionable"], "topic"].value_counts()
        pos_pct = (sub["sentiment"] == "Positive").mean()
        if sub["actionable"].mean() >= 0.20 and len(act):
            name_topic = act.index[0]
        elif (sub["topic"] == "Praise").mean() >= 0.40 or pos_pct >= 0.60:
            name_topic = "Fans / praise"
        else:
            name_topic = "General conversation"
        rows.append(dict(cluster=c, comments=len(sub), share_pct=100 * len(sub) / mask.sum(), top_terms=top,
                         dominant_rule_topic=name_topic, dominant_topic_share_pct=100 * dom.iloc[0] / len(sub),
                         actionable_pct=100 * sub["actionable"].mean(), question_pct=100 * sub["is_question"].mean(),
                         positive_pct=100 * (sub["sentiment"] == "Positive").mean(),
                         negative_pct=100 * (sub["sentiment"] == "Negative").mean(),
                         roman_urdu_pct=100 * (sub["language_type"] == "Roman Urdu").mean(),
                         videos=sub["post_id"].nunique()))
        dist = np.linalg.norm(Z[pos] - km.cluster_centers_[c], axis=1)
        for i in np.argsort(dist)[:4]:
            ex_rows.append(dict(cluster=c, comment=sub.iloc[i]["comment"][:220]))
    cs = pd.DataFrame(rows)
    seg_names = {}
    for r in cs.itertuples():
        tt = []
        for w in r.top_terms.split(", "):
            if not any(w in t or t in w for t in tt):
                tt.append(w)
        seg_names[r.cluster] = f"{r.dominant_rule_topic}: {tt[0]}, {tt[1]}"
    cs["segment"] = cs["cluster"].map(seg_names)
    d["segment"] = d["cluster"].map(seg_names).fillna("Emoji / very short reactions")
    rest = d[~mask]
    extra = dict(cluster=-1, comments=len(rest), share_pct=np.nan, top_terms="(emoji / 1-word comments, not clustered)",
                 dominant_rule_topic=rest["topic"].value_counts().index[0] if len(rest) else "",
                 dominant_topic_share_pct=np.nan, actionable_pct=100 * rest["actionable"].mean(),
                 question_pct=100 * rest["is_question"].mean(), positive_pct=100 * (rest["sentiment"] == "Positive").mean(),
                 negative_pct=100 * (rest["sentiment"] == "Negative").mean(),
                 roman_urdu_pct=100 * (rest["language_type"] == "Roman Urdu").mean(), videos=rest["post_id"].nunique(),
                 segment="Emoji / very short reactions")
    cs = pd.concat([cs, pd.DataFrame([extra])], ignore_index=True)
    nmi = normalized_mutual_info_score(d.loc[mask, "topic"], km.labels_)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    keep_pts = np.random.RandomState(SEED).choice(len(pca), size=min(4000, len(pca)), replace=False)   # plot a sample
    pca = pca[keep_pts]
    cmap = pd.DataFrame({"x": pca[:, 0], "y": pca[:, 1], "segment": d.loc[mask, "segment"].values[keep_pts],
                         "comment": d.loc[mask, "comment"].str[:120].values[keep_pts]})
    ex = pd.DataFrame(ex_rows)
    ex["segment"] = ex["cluster"].map(seg_names)
    return d, cs.round(2), ex, cmap.round(4), dict(best_k=best_k, silhouette=scores, nmi=nmi, clustered=int(mask.sum()))


# ----------------------------------------------------- 5. recommendations ----
def recommendations(d, ct, v, cs, meta) -> pd.DataFrame:
    recs = []
    n = len(d)
    tp = d["topic"].value_counts(normalize=True) * 100

    def add(area, rec, evidence, confidence):
        recs.append(dict(area=area, recommendation=rec, evidence=evidence, confidence=confidence))

    p = tp.get("Price inquiry", 0)
    add("Content", "Put the price (or 'DM/WhatsApp for price') in the first line of every caption and on the product card.",
        f"{p:.1f}% of {n} comments are price enquiries.", "High" if p >= 5 else "Medium")
    disc = tp.get("Discount / sale", 0)
    nsale = int(ct.loc[ct.content_type == "Sale announcement", "videos"].sum())
    add("Promotions", "State the sale end date and discount % clearly in the video itself and pin a comment with the terms.",
        f"{disc:.1f}% of comments ask about or mention discounts/sales; {nsale} of the videos are sale announcements.", "Medium")
    av = tp.get("Availability / location", 0)
    add("Customer service", "Pin a comment on every campaign with city / ship-abroad availability and the store-locator link.",
        f"{av:.1f}% of comments ask whether items are available or ship to a location (UK, USA, other cities).", "Medium")
    dl = tp.get("Delivery / order status", 0)
    add("Customer service", "Reply to order-status comments with a tracking link and a delivery-time promise.",
        f"{dl:.1f}% of comments concern delivery / order status.", "Medium" if dl >= 1 else "Low")
    ru = 100 * (d["language_type"] == "Roman Urdu").mean()
    ur = 100 * (d["language_type"] == "Urdu script").mean()
    add("Customer service", "Staff comment replies in Roman Urdu as well as English, and use bilingual captions.",
        f"{ru:.1f}% of comments are in Roman Urdu and {ur:.1f}% in Urdu script.", "High" if ru >= 10 else "Medium")
    big_ct = ct[(ct["videos"] >= 10) & (ct["content_type"] != "Other")]
    cte = big_ct.sort_values("median_actionable_rate", ascending=False)
    if len(cte):
        t = cte.iloc[0]
        add("Content", f"Test more '{t.content_type}' content: it draws the highest share of buying-signal comments per post.",
            f"Median {100*t.median_actionable_rate:.1f}% buying-signal comments per video over {int(t.videos)} videos "
            f"(all posts: {100*v['actionable_rate'].median():.1f}%).", "Low - small samples; confirm with an A/B test")
    att = big_ct.dropna(subset=["median_comments_per_1k_views"]).sort_values("median_comments_per_1k_views", ascending=False)
    if len(att):
        a_ = att.iloc[0]
        add("Content", f"Use more '{a_.content_type}' content to start conversation: it gets the most comments per 1,000 views.",
            f"Median {a_.median_comments_per_1k_views:.1f} comments per 1,000 views over {int(a_.videos)} videos "
            f"(lowest: {att.iloc[-1].content_type} at {att.iloc[-1].median_comments_per_1k_views:.1f}).", "Medium")
    cn = big_ct.sort_values("negative_pct", ascending=False)
    overall_neg = 100 * (d["sentiment"] == "Negative").mean()
    if len(cn) and cn.iloc[0].negative_pct >= overall_neg + 1.5:      # only flag a clear difference
        w = cn.iloc[0]
        add("Campaign testing", f"Review '{w.content_type}' content: it has the highest share of negative comments.",
            f"{w.negative_pct:.1f}% negative vs {100*(d['sentiment']=='Negative').mean():.1f}% overall.", "Low - small samples")
    un = v[v["unusual"]].sort_values("max_abs_z", ascending=False)
    if len(un):
        r = un.iloc[0]
        add("Campaign testing", f"Investigate '{r.video_title[:70]}' first: {r.why_unusual}.",
            f"{len(un)} of {int(v['eligible'].sum())} eligible posts are flagged as unusual (see the 'Unusual posts' tab).", "Medium")
    comp = d[d["topic"] == "Complaint"]
    add("Customer service", "Prepare a short complaint-response template (apologise, ask for order ID by DM, promise follow-up).",
        f"{100*len(comp)/n:.1f}% of comments are complaints ({len(comp)} comments).", "Medium" if len(comp) >= 20 else "Low")
    big = cs[cs.cluster >= 0].sort_values("actionable_pct", ascending=False).iloc[0]
    add("Audience", f"Treat the '{big.segment}' group as the top lead segment and answer its comments first.",
        f"{big.actionable_pct:.0f}% of its {int(big.comments)} comments are buying-signal topics (price, stock, delivery, size, contact).", "Medium")
    return pd.DataFrame(recs)


# ----------------------------------------------------------------- main ----
def main(include_all=False, include_perfume=False):
    OUT.mkdir(exist_ok=True)
    d = load(include_all, include_perfume)
    d, cs, ex, cmap, meta = cluster_comments(d)
    v = flag_unusual(video_table(d))
    ct = content_summary(d, v)
    recs = recommendations(d, ct, v, cs, meta)
    d.drop(columns=["tokens"]).to_csv(OUT / "comments_enriched.csv", index=False)
    v.to_csv(OUT / "video_features.csv", index=False)
    ct.to_csv(OUT / "content_type_summary.csv", index=False)
    cs.to_csv(OUT / "cluster_summary.csv", index=False)
    ex.to_csv(OUT / "cluster_examples.csv", index=False)
    cmap.to_csv(OUT / "cluster_map.csv", index=False)
    recs.to_csv(OUT / "recommendations.csv", index=False)
    pd.Series({"comments": len(d), "videos": d["post_id"].nunique(), "k": meta["best_k"], "clustered": meta["clustered"],
               "nmi_cluster_vs_rules": round(meta["nmi"], 3),
               **{f"silhouette_k{k}": round(s, 3) for k, s in meta["silhouette"].items()}}
              ).to_csv(OUT / "run_summary.csv", header=["value"])
    print(f"videos={d['post_id'].nunique()} k={meta['best_k']} NMI={meta['nmi']:.3f}")
    print("silhouette:", {k: round(s, 3) for k, s in meta["silhouette"].items()})
    print(f"unusual posts: {int(v['unusual'].sum())} of {int(v['eligible'].sum())} eligible")
    return d, v, ct, cs, recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-all", action="store_true", help="keep every video from Project 1 (incl. non-Khaadi)")
    ap.add_argument("--include-perfume", action="store_true", help="keep Khaadi perfume videos too")
    a = ap.parse_args()
    main(a.include_all, a.include_perfume)
