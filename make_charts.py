"""Static charts for the report (the dashboard has interactive versions)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "output/"
c = pd.read_csv(O + "comments_enriched.csv"); v = pd.read_csv(O + "video_features.csv")
ct = pd.read_csv(O + "content_type_summary.csv"); cs = pd.read_csv(O + "cluster_summary.csv"); cm = pd.read_csv(O + "cluster_map.csv")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
t = ct.sort_values("comments")
ax[0].barh(t.content_type + " (" + t.videos.astype(str) + " posts)", t.comments, color="#2b6cb0"); ax[0].set_title("Total comments by content type")
t = ct.sort_values("median_comments_per_1k_views")
ax[1].barh(t.content_type, t.median_comments_per_1k_views, color="#dd6b20"); ax[1].set_title("Median comments per 1,000 views")
plt.tight_layout(); plt.savefig(O + "chart_content.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
tp = c.topic.value_counts(normalize=True).mul(100).sort_values()
ax[0].barh(tp.index, tp.values, color="#2f855a"); ax[0].set_title("Comment topics (% of comments)")
lg = c.language_type.value_counts(normalize=True).mul(100).sort_values()
ax[1].barh(lg.index, lg.values, color="#805ad5"); ax[1].set_title("Comment language (%)")
plt.tight_layout(); plt.savefig(O + "chart_topics.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5.2))
for s, g in cm.groupby("segment"):
    ax.scatter(g.x, g.y, s=10, alpha=.6, label=s)
ax.set_title("Comment segments (PCA projection of TF-IDF/SVD space)"); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
ax.legend(fontsize=7, loc="best", markerscale=2)
plt.tight_layout(); plt.savefig(O + "chart_segments.png", dpi=150); plt.close()

e = v[v.eligible]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(e[~e.unusual].views, e[~e.unusual].pos_rate, c="#9aa5b1", label="normal")
ax.scatter(e[e.unusual].views, e[e.unusual].pos_rate, c="#d62728", label="unusual")
ax.set_xscale("log"); ax.set_xlabel("Video views (log)"); ax.set_ylabel("Share positive"); ax.legend(); ax.set_title("Unusual posts")
plt.tight_layout(); plt.savefig(O + "chart_unusual.png", dpi=150); plt.close()
