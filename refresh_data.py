"""Incremental refresh of data/comments_project1.csv from the YouTube Data API v3.

Instead of scraping everything again, this script
  1. searches for the newest Khaadi clothing videos (same queries and title filter as Project 1),
  2. updates views / likes / comment counts of every known video,
  3. downloads comments of new videos,
  4. downloads only the newest comments of known videos whose comment count grew,
  5. adds VADER sentiment to the new rows and writes the CSV back.
Then run:  python pipeline.py && python make_web.py

Usage:
    export YOUTUBE_API_KEY=...            # never commit the key
    python refresh_data.py --dry-run      # quota estimate only
    python refresh_data.py

Quota (10,000 units/day): search.list = 100, videos.list = 1, commentThreads.list = 1.
Default settings use roughly 1,300 to 1,800 units per run.
"""
import argparse
import html
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "comments_project1.csv"

QUERIES = [
    "Khaadi sale", "Khaadi new collection 2026", "Khaadi lawn collection", "Khaadi Eid collection",
    "Khaadi pret collection", "Khaadi winter collection", "Khaadi haul", "Khaadi try on",
    "Khaadi unboxing", "Khaadi shopping vlog", "Khaadi 3 piece suit", "Khaadi flat 50% off",
]
BRAND = re.compile(r"khaadi|khaddi", re.I)
CLOTHING = re.compile(r"collection|pret|lawn|suit|dress|kurta|shalwar|kameez|3 ?pc|3 ?piece|2 ?piece|cut piece|"
                      r"sale|off|discount|haul|try.?on|unbox|eid|summer|winter|fabric|clothes|outfit|ready to wear|"
                      r"khaas|casual|signature|stitched|unstitched|trouser|dupatta|abaya|kids wear", re.I)
BLOCK = re.compile(r"perfume|fragrance|attar|deo|body spray|shampoo|soap|hair|oil|henna|mehndi|herbal|ayurved|sketch|"
                   r"paper|saree|sari|meesho|\bipo\b|shares?\b|stock|psx|invest|go public|financials|valuation", re.I)


def is_khaadi_clothing(title: str) -> bool:
    t = html.unescape(title)
    return bool(BRAND.search(t) and CLOTHING.search(t) and not BLOCK.search(t))


def clean(text) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def vader(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t)["compound"] for t in texts]
    labels = ["Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral" for c in scores]
    return labels, scores


def search_new(yt, pages: int):
    found = {}
    for q in QUERIES:
        token = None
        for _ in range(pages):
            try:
                res = yt.search().list(q=q, part="id,snippet", type="video", maxResults=50, pageToken=token,
                                       order="date", regionCode="PK").execute()
            except Exception as e:                      # quota / network: skip this query, keep going
                print(f"  search skip '{q}': {e}")
                break
            for it in res.get("items", []):
                found[it["id"]["videoId"]] = html.unescape(it["snippet"]["title"])
            token = res.get("nextPageToken")
            if not token:
                break
    return found


def video_stats(yt, ids):
    out, ids = {}, list(ids)
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="statistics,snippet", id=",".join(ids[i:i + 50])).execute()
        for it in res.get("items", []):
            st = it.get("statistics", {})
            out[it["id"]] = dict(views=int(st.get("viewCount", 0)), video_likes=int(st.get("likeCount", 0)),
                                 video_comment_count=int(st.get("commentCount", 0)),
                                 video_published_at=it["snippet"]["publishedAt"], channel=it["snippet"]["channelTitle"])
    return out


def get_comments(yt, video_id, title, limit, order, known=None, max_pages=10):
    """Top-level comments. With `known` (set of comment texts) stop as soon as a page contains an already known one."""
    rows, token, pages = [], None, 0
    while len(rows) < limit and pages < max_pages:
        try:
            res = yt.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=token,
                                           textFormat="plainText", order=order).execute()
        except Exception as e:                           # comments disabled etc.
            print(f"  skip {video_id}: {getattr(e, 'reason', e)}")
            break
        pages += 1
        hit_known = False
        for it in res.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            if known is not None and s["textDisplay"] in known:
                hit_known = True
                continue
            rows.append({"video_id": video_id, "video_title": title, "comment": s["textDisplay"],
                         "likes": s["likeCount"], "published_at": s["publishedAt"]})
        token = res.get("nextPageToken")
        if not token or hit_known:
            break
        time.sleep(0.2)
    return rows[:limit]


def refresh(df: pd.DataFrame, yt, a) -> pd.DataFrame:
    known_ids = set(df["video_id"])
    cand = search_new(yt, a.search_pages)
    new = {v: t for v, t in cand.items() if v not in known_ids and is_khaadi_clothing(t)}
    print(f"search: {len(cand)} candidates, {len(new)} new Khaadi clothing videos")

    stats = video_stats(yt, known_ids | set(new))
    for col in ("views", "video_likes", "video_comment_count", "video_published_at", "channel"):
        m = {v: s[col] for v, s in stats.items()}
        df[col] = df["video_id"].map(m).fillna(df[col])

    # new videos: first comments (most relevant)
    new = {v: t for v, t in new.items() if stats.get(v, {}).get("video_comment_count", 0) >= a.min_video_comments}
    rows = []
    for vid, title in list(new.items())[:a.max_new_videos]:
        rows += get_comments(yt, vid, title, a.comments_per_video, "relevance")

    # known videos with more comments than we stored: only the newest comments
    stored = df.groupby("video_id").size()
    grown = []
    for vid, st in stats.items():
        if vid in known_ids:
            grown.append((st["video_comment_count"] - int(stored.get(vid, 0)), vid))
    grown = [g for g in sorted(grown, reverse=True) if g[0] >= a.min_growth][:a.max_refresh_videos]
    titles = df.drop_duplicates("video_id").set_index("video_id")["video_title"]
    seen = set(zip(df["video_id"], df["comment"]))
    for _, vid in grown:
        known = {c for v_, c in seen if v_ == vid}
        rows += get_comments(yt, vid, titles[vid], a.max_newest, "time", known=known, max_pages=3)
    print(f"new videos: {len(new)}; refreshed videos: {len(grown)}; new comments downloaded: {len(rows)}")

    if rows:
        add = pd.DataFrame(rows).drop_duplicates(subset=["video_id", "comment"])
        add = add[[(v, c) not in seen for v, c in zip(add["video_id"], add["comment"])]]
        meta = pd.DataFrame.from_dict(stats, orient="index").reset_index(names="video_id")
        add = add.merge(meta, on="video_id", how="left")
        add["video_url"] = "https://www.youtube.com/watch?v=" + add["video_id"]
        add["platform"] = "YouTube"
        add["clean"] = add["comment"].map(clean)
        add = add[add["clean"].str.len() > 2].copy()
        if len(add):
            add["sentiment"], add["score"] = vader(add["clean"].tolist())
        df = pd.concat([df, add], ignore_index=True)
        print(f"added {len(add)} comments after de-duplication")
    df["collection_date"] = date.today().isoformat()      # data is current as of this run
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-pages", type=int, default=1, help="search pages per query (50 videos each, 100 units per page)")
    ap.add_argument("--max-new-videos", type=int, default=60)
    ap.add_argument("--comments-per-video", type=int, default=200, help="comments downloaded for a new video")
    ap.add_argument("--max-refresh-videos", type=int, default=40, help="known videos to top up per run")
    ap.add_argument("--max-newest", type=int, default=300, help="newest comments downloaded for a known video")
    ap.add_argument("--min-growth", type=int, default=5, help="video must have at least this many unseen comments")
    ap.add_argument("--min-video-comments", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    est = len(QUERIES) * a.search_pages * 100 + 12 + a.max_new_videos * 2 + a.max_refresh_videos * 3
    print(f"Estimated quota: about {est} units (daily limit 10,000)")
    if a.dry_run:
        return
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit("YOUTUBE_API_KEY is not set.")
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", developerKey=key)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    before = len(df)
    df = refresh(df, yt, a)
    df.to_csv(DATA, index=False, encoding="utf-8-sig")
    print(f"Done: {before} -> {len(df)} comments, {df['video_id'].nunique()} videos -> {DATA}")


if __name__ == "__main__":
    main()
