"""Extra outputs for the Khaadi Insights app: leads, ready replies, brand health, alerts, weekly plan.

Called at the end of pipeline.py (python pipeline.py). Everything is rule based and uses only the
enriched comments / video table, so every number can be traced back to the data.
Writes (into output/):
  leads.csv, lead_summary.csv, reply_templates.csv, health_by_content.csv, health_trend.csv,
  alerts.csv, weekly_plan.csv, insights_meta.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HOT_TOPICS = {"Purchase intent / contact", "Delivery / order status"}
ACTION_TOPICS = {"Price inquiry", "Discount / sale", "Availability / location", "Delivery / order status",
                 "Sizing / fit", "Purchase intent / contact"}
WATCH_TOPICS = ["Complaint", "Delivery / order status", "Price inquiry", "Availability / location", "Discount / sale"]
MAX_LEADS = 250          # per priority group (Hot / Warm)
# self-promotion / reseller spam and anything with a phone number or link is kept out of the lead list
SPAM = re.compile(r"https?://|www\.|\.com\b|\+?\d[\d\s-]{8,}\d|follow me|subscribe|my channel|our channel|wa\.me|"
                  r"whats ?app (?:no|number)|contact (?:us|me) (?:on|at)", re.I)

# Drafts only: text in [BRACKETS] must be filled with real information by Khaadi staff before posting.
TEMPLATES = [
    ("Price inquiry", "Thank you for your interest! The price of this item is Rs. [PRICE]. You can see the full range on our website or at your nearest Khaadi store.",
     "Shukriya! Is item ki qeemat Rs. [PRICE] hai. Poori range hamari website ya qareebi Khaadi store par dekh sakte hain."),
    ("Discount / sale", "The sale runs until [END DATE] at [ONLINE / STORES]. Terms: [TERMS].",
     "Sale [END DATE] tak [ONLINE / STORES] par chal rahi hai. Sharait: [TERMS]."),
    ("Availability / location", "This is available [ONLINE / STORE LIST]. You can find your nearest store here: [LINK].",
     "Ye [ONLINE / STORE LIST] par mojood hai. Apna qareebi store yahan dekhein: [LINK]."),
    ("Delivery / order status", "Sorry for the wait! Please DM your order number and we will check the status for you.",
     "Intezar ke liye maazrat! Apna order number DM kar dein, hum status check kar ke batate hain."),
    ("Sizing / fit", "Our size guide is here: [LINK]. Tell us your usual size and we will help you choose.",
     "Hamari size guide yahan hai: [LINK]. Apna aam size bata dein, hum rehnumai kar denge."),
    ("Purchase intent / contact", "Great choice! You can order here: [LINK], or call us on [PHONE].",
     "Zabardast intekhab! Yahan se order kar sakte hain: [LINK], ya [PHONE] par rabta karein."),
    ("Complaint", "We are sorry about this. Please DM your order ID and details, and our team will follow up within [TIME].",
     "Is par hum maazrat chahte hain. Apna order ID aur tafseel DM kar dein, hamari team [TIME] mein rabta karegi."),
    ("Quality / fabric", "This piece is made of [FABRIC]. Care instructions are on the label and the product page.",
     "Ye [FABRIC] ka bana hai. Dekh-bhaal ki hidayat label aur product page par maujood hain."),
    ("Praise", "Thank you so much, we are glad you like it!",
     "Bohat shukriya, aap ko pasand aaya to hamari khushi hai!"),
]


def health_score(df: pd.DataFrame) -> dict:
    """Transparent score: 50 + (positive% - negative%)/2 - complaint%/2, clipped to 0-100."""
    n = len(df)
    if n == 0:
        return dict(comments=0, positive_pct=np.nan, negative_pct=np.nan, complaint_pct=np.nan, score=np.nan)
    pos = 100 * (df["sentiment"] == "Positive").mean()
    neg = 100 * (df["sentiment"] == "Negative").mean()
    comp = 100 * (df["topic"] == "Complaint").mean()
    return dict(comments=n, positive_pct=round(pos, 1), negative_pct=round(neg, 1), complaint_pct=round(comp, 1),
                score=round(float(np.clip(50 + (pos - neg) / 2 - comp / 2, 0, 100)), 1))


def health_label(score: float) -> str:
    return "Strong" if score >= 70 else "Healthy" if score >= 55 else "Mixed" if score >= 40 else "At risk"


def build_leads(d: pd.DataFrame):
    a = d[d["topic"].isin(ACTION_TOPICS)].copy()
    a = a[~a["comment"].str.contains(SPAM) & a["comment"].str.len().between(8, 300)]
    a["priority"] = np.where(a["topic"].isin(HOT_TOPICS), "Hot", "Warm")
    a["rank"] = a["priority"].map({"Hot": 0, "Warm": 1})
    a["ask"] = (~a["is_question"].astype(bool)).astype(int)          # real questions first
    a = a.sort_values(["rank", "ask", "likes", "published_at"], ascending=[True, True, False, False])
    summary = (a.groupby(["priority", "topic"]).size().reset_index(name="comments")
               .sort_values(["priority", "comments"], ascending=[True, False]))
    top = a.groupby("priority", sort=False).head(MAX_LEADS)
    leads = pd.DataFrame({
        "priority": top["priority"], "topic": top["topic"], "comment": top["comment"].str.replace(r"\s+", " ", regex=True).str[:240],
        "language": top["language_type"], "likes": top["likes"].astype(int),
        "date": top["published_at"].dt.strftime("%Y-%m-%d"), "video": top["video_title"].str[:90], "url": top["video_url"]})
    return leads.reset_index(drop=True), summary.reset_index(drop=True), a


def build_health(d: pd.DataFrame):
    by_ct = pd.DataFrame([dict(content_type=ct, **health_score(g)) for ct, g in d.groupby("content_type")])
    by_ct = by_ct.sort_values("comments", ascending=False).reset_index(drop=True)
    m = d.dropna(subset=["published_at"]).copy()
    m["month"] = m["published_at"].dt.strftime("%Y-%m")
    rows = [dict(month=k, **health_score(g)) for k, g in m.groupby("month") if len(g) >= 100]
    trend = pd.DataFrame(rows).sort_values("month").tail(12).reset_index(drop=True)
    return by_ct, trend


def build_alerts(d: pd.DataFrame, v: pd.DataFrame, leads_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor = d["published_at"].max()
    recent = d[d["published_at"] > anchor - pd.Timedelta(days=30)]
    base = d[(d["published_at"] <= anchor - pd.Timedelta(days=30)) & (d["published_at"] > anchor - pd.Timedelta(days=120))]
    if len(recent) >= 100 and len(base) >= 100:
        for t in WATCH_TOPICS:
            r_n, b_n = int((recent["topic"] == t).sum()), int((base["topic"] == t).sum())
            r_s, b_s = 100 * r_n / len(recent), 100 * b_n / len(base)
            if r_n >= 15 and b_s > 0 and r_s >= 1.5 * b_s:
                rows.append(dict(type="Topic rising", severity="Watch", title=f"'{t}' comments are rising",
                                 detail=f"{r_s:.1f}% of comments in the last 30 days vs {b_s:.1f}% in the 90 days before ({r_n} comments).", url=""))
    overall_neg = 100 * (d["sentiment"] == "Negative").mean()
    neg = v[v["eligible"] & (v["comments"] >= 20) & (100 * v["neg_rate"] >= max(overall_neg + 8, 15))].sort_values("neg_rate", ascending=False).head(5)
    for r in neg.itertuples():
        rows.append(dict(type="Negative comments", severity="Review", title=str(r.video_title)[:90],
                         detail=f"{100*r.neg_rate:.0f}% negative comments vs {overall_neg:.1f}% overall ({int(r.comments)} comments).", url=r.video_url))
    backlog = leads_all.groupby(["video_id", "video_title", "video_url"]).size().reset_index(name="n").sort_values("n", ascending=False).head(5)
    for r in backlog.itertuples():
        if r.n >= 25:
            rows.append(dict(type="Buying signals", severity="Opportunity", title=str(r.video_title)[:90],
                             detail=f"{r.n} buying-signal comments (price, stock, delivery, size, contact) on this video.", url=r.video_url))
    return pd.DataFrame(rows, columns=["type", "severity", "title", "detail", "url"])


def build_plan(d: pd.DataFrame, leads_all: pd.DataFrame, v: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    n = len(d)
    rows = []
    top = leads_all.groupby(["video_title", "video_url"]).size().reset_index(name="n").sort_values("n", ascending=False).head(3)
    if len(top):
        names = "; ".join(f"{r.video_title[:45]} ({r.n})" for r in top.itertuples())
        rows.append(dict(step="Reply to the busiest videos first", action="Answer the buying-signal comments on these videos with the ready replies.",
                         why=f"They hold the most buying-signal comments: {names}.", url=top.iloc[0].video_url))
    tp = d["topic"].value_counts(normalize=True) * 100
    rows.append(dict(step="Pin price and availability", action="Pin a comment with the price range, the stores and where to order.",
                     why=f"{tp.get('Price inquiry', 0):.1f}% of comments ask about price and {tp.get('Availability / location', 0):.1f}% about availability.", url=""))
    rows.append(dict(step="Post the sale terms", action="Add the end date and discount % to the first line of the description of every sale video.",
                     why=f"{tp.get('Discount / sale', 0):.1f}% of comments ask about sales or discounts.", url=""))
    ru = 100 * (d["language_type"] == "Roman Urdu").mean()
    rows.append(dict(step="Answer in Roman Urdu too", action="Reply to Roman Urdu comments in Roman Urdu in one batch.",
                     why=f"{ru:.1f}% of comments are written in Roman Urdu.", url=""))
    big = ct[(ct["videos"] >= 10) & (ct["content_type"] != "Other")].dropna(subset=["median_comments_per_1k_views"])
    if len(big):
        b = big.sort_values("median_comments_per_1k_views", ascending=False).iloc[0]
        rows.append(dict(step="Test one content idea", action=f"Run one '{b.content_type}' post and compare comments per 1,000 views.",
                         why=f"This type gets the most comments per 1,000 views ({b.median_comments_per_1k_views:.1f}). It is a hypothesis to test, not a proven effect.", url=""))
    return pd.DataFrame(rows)


def write(d: pd.DataFrame, v: pd.DataFrame, ct: pd.DataFrame, out: Path):
    leads, lead_summary, leads_all = build_leads(d)
    by_ct, trend = build_health(d)
    alerts = build_alerts(d, v, leads_all)
    plan = build_plan(d, leads_all, v, ct)
    tpl = pd.DataFrame(TEMPLATES, columns=["topic", "english", "roman_urdu"])
    overall = health_score(d)
    leads.to_csv(out / "leads.csv", index=False)
    lead_summary.to_csv(out / "lead_summary.csv", index=False)
    tpl.to_csv(out / "reply_templates.csv", index=False)
    by_ct.to_csv(out / "health_by_content.csv", index=False)
    trend.to_csv(out / "health_trend.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    plan.to_csv(out / "weekly_plan.csv", index=False)
    meta = dict(updated=str(d["collection_date"].max()), health=overall, health_label=health_label(overall["score"]),
                leads_total=int(len(leads_all)), leads_hot=int((leads_all["priority"] == "Hot").sum()),
                leads_shown=int(len(leads)))
    (out / "insights_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"insights: {meta['leads_total']} leads ({meta['leads_hot']} hot), health {overall['score']} ({meta['health_label']}), {len(alerts)} alerts")


if __name__ == "__main__":      # python insights.py  -> rebuild only these files from the existing pipeline outputs
    OUT = Path(__file__).parent / "output"
    _d = pd.read_csv(OUT / "comments_enriched.csv")
    _d["published_at"] = pd.to_datetime(_d["published_at"], errors="coerce", utc=True)
    write(_d, pd.read_csv(OUT / "video_features.csv"), pd.read_csv(OUT / "content_type_summary.csv"), OUT)
