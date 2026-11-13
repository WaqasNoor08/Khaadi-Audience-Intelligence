# Khaadi Audience & Campaign Intelligence Engine

**Project 3** of a Khaadi brand-analytics series (Big Data Analytics, UMT).
- Project 1: YouTube comment scraping + VADER sentiment for Khaadi ([repo](https://github.com/WaqasNoor08/Brand-Sentiment-Analysis-))
- Project 2: Khaadi price-tier clustering + budget split ([repo](https://github.com/WaqasNoor08/Khaadi-Price-Clustering))
- **Project 3 (this repo):** clustering-based audience segments, unusual-post detection and marketing recommendations.

It answers the five questions from the project brief in a small Streamlit app:

| Question | How |
|---|---|
| Which content attracts attention? | Posts grouped into content types (sale announcement, haul/try-on, collection showcase, review, unboxing); medians of comment volume, comment likes, buying-signal share |
| What does the audience want? | Keyword-rule topics (price, discount, availability, delivery, quality, sizing, intent, praise, complaint) + language type (English / Roman Urdu / Urdu script) |
| Which patterns are unusual? | Robust z-scores against posts of the same content type + Isolation Forest |
| Which audience needs should the brand address? | TF-IDF -> SVD -> KMeans on comments (k chosen by silhouette) |
| What should marketing do next? | Rule-based recommendations, each with its evidence and a confidence note; linked to the Project 2 budget plan |

## Run
```bash
pip install -r requirements.txt
python pipeline.py          # Khaadi CLOTHING only -> output/*.csv  (--include-perfume / --include-all to widen)
python make_charts.py       # optional static charts
streamlit run app.py
```

## Data
`data/comments_project1.csv` = the updated Project 1 results: **21,514 public YouTube comments on 571 Khaadi clothing videos** (126 channels), collected 12 Nov 2026 with the YouTube Data API v3,
with VADER sentiment and video statistics (`views`, `video_likes`, `video_comment_count`, `video_published_at`, `video_url`, `collection_date`).
Project 1 keeps only videos whose title has "Khaadi" plus a clothing word; perfume, "Khadi" herbal/stationery and IPO videos are filtered out.
Required columns: `video_title, comment, likes, published_at, score, sentiment` (video statistics are optional; without them the engine falls back to comment counts).
All records are real, none simulated (`data_provenance` column). `data/budget_plan.csv` and `data/cluster_summary.csv` are Project 2 outputs.

## Key results (this data)
- 45.0% positive, 48.5% neutral, 6.6% negative; 37.9% of comments are Roman Urdu.
- ~25% of comments carry a buying signal: discount 7.4%, availability 6.0%, price 5.4%, contact/intent 3.4%.
- Sale announcements are the largest group (282 videos) and have the highest buying-signal share (median 33%); haul / try-on videos get the most comments per 1,000 views (3.2 vs 1.0 for sales).
- The clearest audience segment is "sale kab tak": 1,247 comments, 91% buying-signal, 61% Roman Urdu.
- 24 of 393 eligible posts are flagged unusual.

## Coverage of the project brief
| Requirement | Status |
|---|---|
| Web app, five questions, clustering, anomalies, recommendations | Done |
| One product category (clothing) | Done |
| Table B: language_type, topic, sentiment, purchase_intent | Done |
| Table A: platform, campaign, content_type, likes, comments, views, days_since_post | Done |
| Table A: shares | Not available (the YouTube API does not expose shares) |
| Instagram / TikTok / Facebook | Not done (YouTube only) |
| 2,000-5,000 records | Exceeded (21,514 comments) |
| Provenance, source URL, collection date, real vs simulated | Done |

## Limits (read before quoting numbers)
- YouTube only; comments are sampled (up to 500 per video, by relevance); no shares.
- Some videos feature several brands or are reseller / sale-spam channels, so comments do not always describe Khaadi's own content.
- Cluster separation is weak (silhouette 0.11); segments are exploratory.
- Topic and language detection use keyword rules; ~47% of comments match no topic.
- VADER under-scores Roman Urdu, so "neutral" and "negative" are approximate.
- Recommendations are hypotheses to A/B test, not proven effects.
