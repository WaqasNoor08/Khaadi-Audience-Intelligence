"""Builds web/index.html - a static (no server) version of the dashboard for Vercel.

Run after pipeline.py:   python make_web.py
Output: web/index.html (one self-contained file; data is embedded as JSON).
"""
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "output"
WEB = BASE / "web"
GITHUB = "https://github.com/WaqasNoor08/Khaadi-Audience-Intelligence"
STREAMLIT = "https://khaadi-audience-intelligence.streamlit.app/"

ACTION = ["Price inquiry", "Discount / sale", "Availability / location", "Delivery / order status",
          "Sizing / fit", "Purchase intent / contact"]


def build_data():
    c = pd.read_csv(OUT / "comments_enriched.csv")
    v = pd.read_csv(OUT / "video_features.csv")
    ct = pd.read_csv(OUT / "content_type_summary.csv")
    cs = pd.read_csv(OUT / "cluster_summary.csv")
    ex = pd.read_csv(OUT / "cluster_examples.csv")
    cm = pd.read_csv(OUT / "cluster_map.csv")
    recs = pd.read_csv(OUT / "recommendations.csv")
    run = pd.read_csv(OUT / "run_summary.csv", index_col=0)["value"]
    budget = pd.read_csv(BASE / "data" / "budget_plan.csv")

    cube = (c.groupby(["content_type", "language_type", "topic", "sentiment"]).size().reset_index(name="n"))
    eid = (v.groupby("eid_campaign").agg(posts=("post_id", "count"), comments=("comments", "sum"),
                                         median_views=("views", "median"),
                                         cpk=("comments_per_1k_views", "median")).reset_index())
    eid["eid_campaign"] = eid["eid_campaign"].map({True: "Eid campaign", False: "Other"})

    el = v[v["eligible"]]
    vids = [[r.post_id, str(r.video_title)[:90], r.content_type, int(r.views), round(float(r.pos_rate), 3), bool(r.unusual), r.video_url]
            for r in el.itertuples()]
    un = v[v["unusual"]].sort_values("max_abs_z", ascending=False)
    unusual = [dict(id=r.post_id, title=str(r.video_title)[:110], ct=r.content_type, views=int(r.views), days=int(r.days_since_post),
                    comments=int(r.comments), pos=round(float(r.pos_rate), 2), neg=round(float(r.neg_rate), 2),
                    q=round(float(r.question_rate), 2), why=r.why_unusual, url=r.video_url) for r in un.itertuples()]

    segs = cs[cs.cluster >= 0].reset_index(drop=True)
    seg_names = segs["segment"].tolist()
    pts = cm.sample(n=min(1500, len(cm)), random_state=1)
    points = [[round(float(r.x), 3), round(float(r.y), 3), seg_names.index(r.segment)] for r in pts.itertuples() if r.segment in seg_names]
    seg_rows = [dict(name=r.segment, comments=int(r.comments), share=round(float(r.share_pct), 1), act=round(float(r.actionable_pct), 1),
                     pos=round(float(r.positive_pct), 1), neg=round(float(r.negative_pct), 1), ru=round(float(r.roman_urdu_pct), 1),
                     terms=r.top_terms) for r in segs.itertuples()]
    examples = {name: ex[ex.segment == name]["comment"].tolist() for name in seg_names}

    return dict(
        meta=dict(comments=int(len(c)), videos=int(c["post_id"].nunique()), k=int(run["k"]),
                  silhouette=float(run[f"silhouette_k{int(run['k'])}"]), eligible=int(v["eligible"].sum()),
                  unusual=int(v["unusual"].sum()), github=GITHUB, streamlit=STREAMLIT),
        action=ACTION,
        cube=cube.values.tolist(),
        ct=ct.round(4).to_dict(orient="records"),
        eid=eid.round(3).to_dict(orient="records"),
        vids=vids, unusual=unusual, segs=seg_rows, points=points, examples=examples,
        recs=recs.to_dict(orient="records"),
        budget=budget.round(1).to_dict(orient="records"),
    )


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KhaadiPulse · Khaadi Audience &amp; Campaign Intelligence</title>
<meta name="theme-color" content="#0E0B1F">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,600&display=swap" rel="stylesheet">
<meta name="description" content="KhaadiPulse: audience segments, unusual-post detection and marketing ideas from 21,514 public YouTube comments on Khaadi clothing videos.">
<style>
:root{color-scheme:dark;--bg:#0E0B1F;--bg2:#15112B;--card:#1B1636;--ink:#F7F1E6;--muted:#A8A1C4;--line:#2E2753;--accent:#F6B93B;--accent2:#E8467C;--teal:#27C4B0;--chip:#261F4A;--red:#E8467C;--green:#27C4B0}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "DM Sans",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
 background-image:radial-gradient(900px 500px at 85% -5%,rgba(232,70,124,.22),transparent 60%),radial-gradient(800px 480px at -5% 8%,rgba(246,185,59,.16),transparent 60%),radial-gradient(700px 600px at 50% 110%,rgba(39,196,176,.12),transparent 60%);background-attachment:fixed}
.wrap{max-width:1120px;margin:0 auto;padding:0 18px 18px}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
h1,h2,h3,.brand,.big{font-family:"Fraunces",Georgia,"Times New Roman",serif}
h2{font-size:1.5rem;margin:1.3rem 0 .4rem;letter-spacing:-.01em}h3{font-size:1.1rem;margin:1.2rem 0 .3rem}
.sub{color:var(--muted);font-size:.86rem;max-width:78ch}.sub a{color:var(--accent)}
nav.top{display:flex;align-items:center;gap:18px;padding:16px 0;flex-wrap:wrap}
.brand{font-size:1.45rem;font-weight:700;color:var(--ink);text-decoration:none;letter-spacing:-.02em;display:flex;align-items:center;gap:8px}
.brand b{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.brand .dot{width:11px;height:11px;border-radius:50%;background:var(--accent2);box-shadow:0 0 0 4px rgba(232,70,124,.25)}
nav.top .links{margin-left:auto;display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:.9rem}
nav.top .links a{color:var(--muted);text-decoration:none}nav.top .links a:hover{color:var(--ink)}
.btn{display:inline-block;padding:10px 18px;border-radius:999px;font-weight:600;text-decoration:none;font-size:.92rem;border:1px solid var(--line);color:var(--ink);background:transparent}
.btn.pri{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#1a1030;border:0}
.btn:hover{filter:brightness(1.08)}
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:30px;align-items:center;padding:34px 0 30px;position:relative}
.hero h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.06;margin:.2rem 0 .8rem;letter-spacing:-.025em}
.hero h1 em{font-style:italic;background:linear-gradient(90deg,var(--accent),var(--accent2) 60%,#b58bff);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p.lead{color:#d8d2ee;font-size:1.05rem;max-width:52ch}
.cta{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0 6px}
.print{position:absolute;inset:-10px -30px auto auto;width:260px;height:260px;opacity:.5;z-index:-1;
 background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='52' height='52' viewBox='0 0 52 52'%3E%3Cpath d='M26 4l10 14-10 14-10-14z' fill='none' stroke='%23F6B93B' stroke-width='1.5'/%3E%3Ccircle cx='26' cy='18' r='2.6' fill='%23E8467C'/%3E%3Ccircle cx='0' cy='52' r='5' fill='none' stroke='%2327C4B0' stroke-width='1.5'/%3E%3Ccircle cx='52' cy='52' r='5' fill='none' stroke='%2327C4B0' stroke-width='1.5'/%3E%3Ccircle cx='26' cy='44' r='2' fill='%23F6B93B'/%3E%3C/svg%3E");
 -webkit-mask-image:radial-gradient(circle at 70% 30%,#000 25%,transparent 72%);mask-image:radial-gradient(circle at 70% 30%,#000 25%,transparent 72%)}
.mock{background:linear-gradient(160deg,#241d4d,#1a1535);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 24px 60px rgba(0,0,0,.45)}
.mock .who{display:flex;gap:10px;align-items:center;font-size:.82rem;color:var(--muted)}
.mock .av{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,var(--teal),#4DA3FF)}
.mock .q{font-size:1.15rem;margin:10px 0 12px;font-weight:500}
.mock .tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.tag{font-size:.74rem;padding:3px 10px;border-radius:999px;background:var(--chip);border:1px solid var(--line);color:#ddd6f7}
.tag.hot{background:rgba(232,70,124,.18);border-color:rgba(232,70,124,.5);color:#ffb3cf}
.mock .rep{border-left:3px solid var(--accent);background:rgba(246,185,59,.08);padding:9px 12px;border-radius:0 10px 10px 0;font-size:.88rem}
.mock .rep small{display:block;color:var(--accent);margin-bottom:3px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0 26px}
.stat{padding:14px 16px;border-radius:14px;background:var(--bg2);border:1px solid var(--line)}
.stat .big{font-size:1.9rem;font-weight:700;line-height:1.1}
.stat:nth-child(1) .big{color:var(--accent)}.stat:nth-child(2) .big{color:var(--accent2)}.stat:nth-child(3) .big{color:var(--teal)}.stat:nth-child(4) .big{color:#b58bff}
.stat span{color:var(--muted);font-size:.82rem}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:8px 0 28px}
.step{padding:16px;border-radius:14px;background:var(--bg2);border:1px solid var(--line)}
.step:nth-child(1){border-top:3px solid var(--accent)}.step:nth-child(2){border-top:3px solid var(--accent2)}.step:nth-child(3){border-top:3px solid var(--teal)}
.step b{font-family:"Fraunces",Georgia,serif;font-size:1.1rem}.step p{margin:.4rem 0 0;color:var(--muted);font-size:.9rem}
#app{scroll-margin-top:10px;border-top:1px solid var(--line);padding-top:22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.kpi b{display:block;font-size:1.5rem;font-family:"Fraunces",Georgia,serif}.kpi span{color:var(--muted);font-size:.78rem}
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
label{font-size:.8rem;color:var(--muted);display:flex;flex-direction:column;gap:3px}
select{padding:8px 10px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;max-width:100%}
.tabs{display:flex;gap:6px;overflow-x:auto;padding:6px 0;margin-top:10px;position:sticky;top:0;background:rgba(14,11,31,.92);backdrop-filter:blur(6px);z-index:5;border-bottom:1px solid var(--line)}
.tabs button{border:1px solid transparent;background:none;color:var(--muted);padding:8px 14px;font:inherit;font-size:.88rem;white-space:nowrap;border-radius:999px;cursor:pointer}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected=true]{color:#1a1030;background:linear-gradient(90deg,var(--accent),var(--accent2));font-weight:700}
section.tab{display:none}section.tab.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:10px 0}
.bars .row{display:grid;grid-template-columns:minmax(90px,34%) 1fr;gap:10px;align-items:center;margin:7px 0;font-size:.86rem}
.bars .track{background:var(--chip);border-radius:8px;height:24px;position:relative;overflow:hidden}
.bars .fill{background:linear-gradient(90deg,var(--accent2),var(--accent));height:100%;border-radius:8px}
.bars .val{position:absolute;right:8px;top:0;line-height:24px;font-size:.78rem;color:var(--ink);text-shadow:0 1px 2px rgba(0,0,0,.6)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.82rem}
th,td{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}th{color:var(--accent);font-weight:600}
td a{color:#ffd98a}
td.n,th.n{text-align:right;white-space:nowrap}
.warn{background:rgba(246,185,59,.1);color:#ffd98a;border:1px solid rgba(246,185,59,.4);border-radius:10px;padding:9px 12px;font-size:.84rem;margin:8px 0}
.rec{border-left:4px solid var(--accent2)}
.rec:nth-child(3n+2){border-left-color:var(--accent)}.rec:nth-child(3n){border-left-color:var(--teal)}
.rec small{color:var(--muted);display:block;margin-top:5px}
svg{width:100%;height:auto;display:block}
#sc_un svg,#sc_seg svg{max-width:760px;margin:0 auto}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.legend button{border:1px solid var(--line);background:var(--bg2);color:var(--ink);border-radius:999px;padding:4px 11px;font-size:.76rem;cursor:pointer}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.legend button[aria-pressed=true]{outline:2px solid var(--accent)}
.sel{font-size:.86rem}.sel a{color:var(--accent)}
blockquote{margin:6px 0;padding:8px 12px;background:var(--chip);border-left:3px solid var(--teal);border-radius:0 10px 10px 0;font-size:.86rem}
.maker{margin:34px 0 8px;display:flex;gap:18px;align-items:center;padding:20px;border-radius:20px;background:linear-gradient(135deg,rgba(246,185,59,.12),rgba(232,70,124,.14) 55%,rgba(39,196,176,.12));border:1px solid var(--line);flex-wrap:wrap}
.maker .pic{width:92px;height:92px;border-radius:50%;padding:3px;background:conic-gradient(var(--accent),var(--accent2),var(--teal),var(--accent));flex:none}
.maker .pic img{width:100%;height:100%;border-radius:50%;object-fit:cover;display:block;background:var(--bg2);border:3px solid var(--bg)}
.maker .txt{flex:1;min-width:220px}
.maker .name{font-family:"Fraunces",Georgia,serif;font-size:1.35rem;font-weight:700}
.maker .made{color:var(--muted);font-size:.82rem}
.maker .links{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
footer{color:var(--muted);font-size:.78rem;margin:14px 0 16px;text-align:center}
@media (max-width:820px){.hero{grid-template-columns:1fr}.print{display:none}nav.top .links{margin-left:0}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>
</head>
<body>
<div class="wrap">
<nav class="top">
<a class="brand" href="#"><span class="dot"></span>Khaadi<b>Pulse</b></a>
<div class="links"><a href="#how">How it works</a><a class="gh" href="#" target="_blank" rel="noopener">GitHub</a><a class="st" href="#" target="_blank" rel="noopener">Streamlit version</a><a class="btn pri" href="#app">Open the dashboard</a></div>
</nav>

<header class="hero">
<div class="print" aria-hidden="true"></div>
<div>
<h1>The shoppers under Khaadi's videos are <em>telling you</em> what to launch next.</h1>
<p class="lead">KhaadiPulse reads thousands of public YouTube comments on Khaadi clothing videos. It shows which content gets attention, what buyers ask for, which posts look unusual and what marketing should try next.</p>
<div class="cta"><a class="btn pri" href="#app">Open the dashboard</a><a class="btn st" href="#" target="_blank" rel="noopener">Try the Streamlit app</a></div>
<p class="sub">Project 3 &middot; Big Data Analytics &middot; an observational sample of public comments, not Khaadi's own analytics.</p>
</div>
<div class="mock" aria-label="Example of a buyer comment and the idea it produces">
<div class="who"><span class="av"></span>Real comment under a Khaadi sale video</div>
<div class="q">&ldquo;Ye kb tk he&rdquo;</div>
<div class="tags"><span class="tag hot">Buying signal</span><span class="tag">Discount / sale</span><span class="tag">Roman Urdu</span></div>
<div class="rep"><small>Idea to test</small>Pin the sale end date in the first line of the description and the pinned comment.</div>
</div>
</header>

<div class="stats">
<div class="stat"><div class="big" id="hs1">-</div><span>public comments read</span></div>
<div class="stat"><div class="big" id="hs2">-</div><span>Khaadi clothing videos</span></div>
<div class="stat"><div class="big" id="hs3">-</div><span>comments with a buying signal</span></div>
<div class="stat"><div class="big" id="hs4">-</div><span>comments in Roman Urdu</span></div>
</div>

<h2 id="how">How it works</h2>
<div class="steps">
<div class="step"><b>We read the comments</b><p>Public YouTube comments on Khaadi clothing videos, collected with the YouTube Data API.</p></div>
<div class="step"><b>We group what people want</b><p>Keyword topics, language and clustering turn thousands of comments into a few clear audience needs.</p></div>
<div class="step"><b>Marketing gets next steps</b><p>Every recommendation shows its evidence and how sure we are, so it can be tested.</p></div>
</div>

<div id="app">
<h2 style="margin-top:0">Dashboard</h2>
<div class="filters">
<label>Content type<select id="fct"></select></label>
<label>Comment language<select id="fl"></select></label>
</div>
<div class="kpis" id="kpis"></div>

<div class="tabs" role="tablist" id="tabs"></div>

<section class="tab" id="t0">
<h2>Which content attracts attention?</h2>
<p class="sub">Attention uses video views, video likes and comments (YouTube does not expose shares). Medians are used because a few very large videos would dominate any average. This tab does not use the filters above.</p>
<label>Metric<select id="metric"></select></label>
<div class="card bars" id="b_ct"></div>
<div id="w_ct"></div>
<div class="tablewrap card"><table id="tb_ct"></table></div>
<h3>Eid vs other campaigns</h3>
<div class="tablewrap card"><table id="tb_eid"></table></div>
</section>

<section class="tab" id="t1">
<h2>What does the audience want?</h2>
<p class="sub" id="n_filter"></p>
<h3>Comment topics</h3><div class="card bars" id="b_topic"></div>
<h3>Comment language</h3><div class="card bars" id="b_lang"></div>
<h3>Sentiment (VADER)</h3><div class="card bars" id="b_sent"></div>
<p class="sub">Topics come from transparent keyword rules (see pipeline.py), so about half of comments fit no rule. VADER under-scores Roman Urdu.</p>
</section>

<section class="tab" id="t2">
<h2>Which marketing patterns are unusual?</h2>
<p class="sub">Each post is compared with posts of the same content type (robust z-score on views, video likes per view, comments per 1,000 views, negative, question and buying-signal share, and sentiment) and with an Isolation Forest. Posts need at least 10 collected comments and 1,000 views. Flags are leads to investigate, not proof of a problem.</p>
<div class="card"><div id="sc_un"></div><div class="sel" id="sel_un">Tap a dot to see the video.</div></div>
<h3 id="h_un"></h3>
<div class="tablewrap card"><table id="tb_un"></table></div>
</section>

<section class="tab" id="t3">
<h2>Which audience needs should the brand address?</h2>
<p class="sub" id="seg_note"></p>
<div class="tablewrap card"><table id="tb_seg"></table></div>
<div class="card"><div class="legend" id="lg_seg"></div><div id="sc_seg"></div></div>
<label>Example comments<select id="exsel"></select></label>
<div id="ex_list"></div>
</section>

<section class="tab" id="t4">
<h2>What should the marketing team do next?</h2>
<div id="recs"></div>
<h3>Link with Project 2 (price tiers &amp; budget)</h3>
<div class="tablewrap card"><table id="tb_bud"></table></div>
<p class="sub" id="p2note"></p>
</section>

<section class="tab" id="t5">
<h2>About &amp; limits</h2>
<p><b>Data.</b> 21,514 public YouTube comments collected on 12 Nov 2026 with the YouTube Data API v3 from 571 Khaadi <i>clothing</i> videos (perfume, "Khadi" herbal/stationery and IPO videos are filtered out). Video views, likes and publish dates come from the same API. All records are real; none are simulated.</p>
<p><b>Limits.</b> One platform (YouTube); shares are not available; some videos feature several brands (Khaadi only appears in the title); topic rules and language detection are keyword based and Roman Urdu spelling varies; VADER under-scores Roman Urdu; cluster separation is weak, so segments are exploratory. Recommendations are hypotheses to A/B test, not proven effects.</p>
<p><b>This page</b> is a static snapshot built by <code>make_web.py</code> from the pipeline outputs, so it works on Vercel with no server. The Streamlit version has the full filters and comment-level tables.</p>
</section>

</div><!-- /#app -->

<section class="maker" aria-label="About the maker">
<div class="pic"><img src="https://github.com/WaqasNoor08.png?size=240" alt="Waqas Noor" width="92" height="92" loading="lazy" referrerpolicy="no-referrer"></div>
<div class="txt">
<div class="made">Made by</div>
<div class="name">Waqas Noor</div>
<div class="made">F2023376280 &middot; Section A2 &middot; UMT Lahore</div>
<div class="links"><a class="btn" href="https://github.com/WaqasNoor08" target="_blank" rel="noopener">GitHub profile</a><a class="btn gh" href="#" target="_blank" rel="noopener">Project code</a><a class="btn st" href="#" target="_blank" rel="noopener">Streamlit app</a></div>
</div>
</section>
<footer>&copy; 2026 KhaadiPulse &middot; Made by Waqas Noor &middot; Student project, not affiliated with Khaadi.</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var $=function(id){return document.getElementById(id)};
var esc=function(s){return String(s).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})};
var nf=function(x,d){return Number(x).toLocaleString("en-US",{maximumFractionDigits:d==null?0:d,minimumFractionDigits:d==null?0:d})};
var PAL=["#F6B93B","#E8467C","#27C4B0","#8E7CFF","#FF8A5B","#7BD88F","#4DA3FF","#F28BB8","#C9B6FF"];
Array.prototype.forEach.call(document.querySelectorAll(".gh"),function(a){a.href=D.meta.github});Array.prototype.forEach.call(document.querySelectorAll(".st"),function(a){a.href=D.meta.streamlit});

/* ---------- filters + KPIs ---------- */
var cts=Array.from(new Set(D.cube.map(function(r){return r[0]}))).sort();
var langs=Array.from(new Set(D.cube.map(function(r){return r[1]}))).sort();
function fillSel(el,list){el.innerHTML='<option value="">All</option>'+list.map(function(x){return'<option>'+esc(x)+'</option>'}).join("")}
fillSel($("fct"),cts);fillSel($("fl"),langs);
function filtered(){var a=$("fct").value,b=$("fl").value;return D.cube.filter(function(r){return(!a||r[0]===a)&&(!b||r[1]===b)})}
function tally(rows,idx){var m={},tot=0;rows.forEach(function(r){m[r[idx]]=(m[r[idx]]||0)+r[4];tot+=r[4]});return{m:m,tot:tot}}
function bars(el,map,tot,fmtPct){
  var items=Object.keys(map).map(function(k){return[k,map[k]]}).sort(function(a,b){return b[1]-a[1]});
  var max=items.length?items[0][1]:1;
  el.innerHTML=items.map(function(it){
    var pct=tot?100*it[1]/tot:0;
    return'<div class="row"><div>'+esc(it[0])+'</div><div class="track"><div class="fill" style="width:'+(100*it[1]/max)+'%"></div><span class="val">'+nf(it[1])+' &middot; '+nf(pct,1)+'%</span></div></div>'}).join("")||"No comments match these filters.";
}
function renderKpis(){
  var rows=filtered(),tot=0,pos=0,act=0,ru=0;
  rows.forEach(function(r){tot+=r[4];if(r[3]==="Positive")pos+=r[4];if(D.action.indexOf(r[2])>=0)act+=r[4];if(r[1]==="Roman Urdu")ru+=r[4]});
  var p=function(x){return tot?nf(100*x/tot,1)+"%":"-"};
  $("kpis").innerHTML=[["Comments",nf(tot)],["Videos in dataset",nf(D.meta.videos)],["Positive",p(pos)],["Buying-signal comments",p(act)],["Roman Urdu",p(ru)]]
   .map(function(k){return'<div class="kpi"><b>'+k[1]+'</b><span>'+k[0]+'</span></div>'}).join("");
}
function renderAudience(){
  var rows=filtered(),t=tally(rows,2),l=tally(rows,1),s=tally(rows,3);
  $("n_filter").textContent="Filters: "+($("fct").value||"all content types")+" / "+($("fl").value||"all languages")+" ("+nf(t.tot)+" comments).";
  bars($("b_topic"),t.m,t.tot);bars($("b_lang"),l.m,l.tot);bars($("b_sent"),s.m,s.tot);
}
function refresh(){renderKpis();renderAudience()}
$("fct").onchange=refresh;$("fl").onchange=refresh;

/* ---------- tab 1: content ---------- */
var METRICS=[["median_comments_per_1k_views","Median comments per 1,000 views",2],["median_views","Median views",0],["median_like_rate","Median video likes per view",3],
 ["comments","Comments collected",0],["median_likes_per_comment","Median likes per comment",2],["median_actionable_rate","Median buying-signal share",3],["positive_pct","Positive comments (%)",1]];
$("metric").innerHTML=METRICS.map(function(m,i){return'<option value="'+i+'">'+esc(m[1])+'</option>'}).join("");
function renderContent(){
  var m=METRICS[$("metric").value],rows=D.ct.slice().sort(function(a,b){return b[m[0]]-a[m[0]]}),max=rows[0][m[0]]||1;
  $("b_ct").innerHTML=rows.map(function(r){
    return'<div class="row"><div>'+esc(r.content_type)+' <span class="sub">('+r.videos+' videos)</span></div><div class="track"><div class="fill" style="width:'+(100*r[m[0]]/max)+'%"></div><span class="val">'+nf(r[m[0]],m[2])+'</span></div></div>'}).join("");
  var small=D.ct.filter(function(r){return r.videos<10}).map(function(r){return r.content_type});
  $("w_ct").innerHTML=small.length?'<div class="warn">Fewer than 10 videos, treat as anecdotal (small sample): '+esc(small.join(", "))+'</div>':"";
  var head='<tr><th>Content type</th><th class="n">Videos</th><th class="n">Comments</th><th class="n">Median views</th><th class="n">Comments / 1k views</th><th class="n">Likes / view</th><th class="n">Buying-signal</th><th class="n">Positive %</th></tr>';
  $("tb_ct").innerHTML=head+D.ct.map(function(r){return'<tr><td>'+esc(r.content_type)+'</td><td class="n">'+r.videos+'</td><td class="n">'+nf(r.comments)+'</td><td class="n">'+nf(r.median_views)+'</td><td class="n">'+nf(r.median_comments_per_1k_views,2)+'</td><td class="n">'+nf(r.median_like_rate,3)+'</td><td class="n">'+nf(100*r.median_actionable_rate,0)+'%</td><td class="n">'+nf(r.positive_pct,1)+'</td></tr>'}).join("");
  $("tb_eid").innerHTML='<tr><th>Group</th><th class="n">Videos</th><th class="n">Comments</th><th class="n">Median views</th><th class="n">Comments / 1k views</th></tr>'+D.eid.map(function(r){return'<tr><td>'+esc(r.eid_campaign)+'</td><td class="n">'+r.posts+'</td><td class="n">'+nf(r.comments)+'</td><td class="n">'+nf(r.median_views)+'</td><td class="n">'+nf(r.cpk,2)+'</td></tr>'}).join("");
}
$("metric").onchange=renderContent;

/* ---------- tab 3: unusual ---------- */
function renderUnusual(){
  var W=640,H=340,L=46,R=12,T=10,B=40,vs=D.vids;
  var lx=vs.map(function(v){return Math.log10(v[3])}),mn=Math.floor(Math.min.apply(null,lx)),mx=Math.ceil(Math.max.apply(null,lx));
  var X=function(v){return L+(Math.log10(v)-mn)/(mx-mn)*(W-L-R)},Y=function(p){return T+(1-p)*(H-T-B)};
  var s='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Views versus share of positive comments">';
  for(var i=0;i<=4;i++){var y=Y(i/4);s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="var(--line)"/><text x="'+(L-6)+'" y="'+(y+4)+'" font-size="11" text-anchor="end" fill="var(--muted)">'+(i*25)+'%</text>'}
  for(var e=mn;e<=mx;e++){var x=X(Math.pow(10,e));s+='<line x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+(H-B)+'" stroke="var(--line)"/><text x="'+x+'" y="'+(H-B+16)+'" font-size="11" text-anchor="middle" fill="var(--muted)">'+nf(Math.pow(10,e))+'</text>'}
  s+='<text x="'+((W+L)/2)+'" y="'+(H-6)+'" font-size="12" text-anchor="middle" fill="var(--muted)">Video views (log scale)</text><text transform="translate(12,'+(H/2)+') rotate(-90)" font-size="12" text-anchor="middle" fill="var(--muted)">Share positive</text>';
  vs.filter(function(v){return!v[5]}).forEach(function(v,i){s+='<circle data-i="'+vs.indexOf(v)+'" cx="'+X(v[3])+'" cy="'+Y(v[4])+'" r="4" fill="#8a83b5" fill-opacity=".7"/>'});
  vs.filter(function(v){return v[5]}).forEach(function(v){s+='<circle data-i="'+vs.indexOf(v)+'" cx="'+X(v[3])+'" cy="'+Y(v[4])+'" r="6" fill="#E8467C" stroke="#0E0B1F" stroke-width="1.5"/>'});
  s+='</svg><div class="legend"><span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#8a83b5;margin-right:5px"></i>normal</span><span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#E8467C;margin-right:5px"></i>unusual</span></div>';
  $("sc_un").innerHTML=s;
  $("sc_un").onclick=function(ev){var t=ev.target;if(t.tagName!=="circle")return;var v=vs[+t.getAttribute("data-i")];
    $("sel_un").innerHTML='<b>'+esc(v[1])+'</b><br>'+esc(v[2])+' &middot; '+nf(v[3])+' views &middot; '+nf(100*v[4],0)+'% positive'+(v[5]?' &middot; <b>unusual</b>':'')+' &middot; <a href="'+esc(v[6])+'" target="_blank" rel="noopener">open video</a>'};
  $("h_un").textContent=D.meta.unusual+" unusual posts of "+D.meta.eligible+" checked";
  $("tb_un").innerHTML='<tr><th>Video</th><th>Type</th><th class="n">Views</th><th class="n">Comments</th><th class="n">Pos</th><th class="n">Neg</th><th>Why flagged</th></tr>'+D.unusual.map(function(u){
    return'<tr><td><a href="'+esc(u.url)+'" target="_blank" rel="noopener">'+esc(u.title)+'</a></td><td>'+esc(u.ct)+'</td><td class="n">'+nf(u.views)+'</td><td class="n">'+u.comments+'</td><td class="n">'+nf(100*u.pos)+'%</td><td class="n">'+nf(100*u.neg)+'%</td><td>'+esc(u.why)+'</td></tr>'}).join("");
}

/* ---------- tab 4: segments ---------- */
var activeSeg=-1;
function renderSegments(){
  $("seg_note").textContent="TF-IDF + SVD + KMeans on comments with at least two content words, k = "+D.meta.k+" chosen by silhouette. Silhouette is low ("+nf(D.meta.silhouette,2)+"): short mixed-language comments do not separate cleanly, so treat segments as exploratory.";
  $("tb_seg").innerHTML='<tr><th>Segment</th><th class="n">Comments</th><th class="n">Share %</th><th class="n">Buying-signal %</th><th class="n">Positive %</th><th class="n">Negative %</th><th class="n">Roman Urdu %</th><th>Top terms</th></tr>'+D.segs.map(function(r,i){
    return'<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:'+PAL[i%PAL.length]+';margin-right:6px"></i>'+esc(r.name)+'</td><td class="n">'+nf(r.comments)+'</td><td class="n">'+r.share+'</td><td class="n">'+r.act+'</td><td class="n">'+r.pos+'</td><td class="n">'+r.neg+'</td><td class="n">'+r.ru+'</td><td>'+esc(r.terms)+'</td></tr>'}).join("");
  $("lg_seg").innerHTML=D.segs.map(function(r,i){return'<button data-s="'+i+'" aria-pressed="false"><i style="background:'+PAL[i%PAL.length]+'"></i>'+esc(r.name)+'</button>'}).join("");
  $("lg_seg").onclick=function(ev){var b=ev.target.closest("button");if(!b)return;var i=+b.getAttribute("data-s");activeSeg=(activeSeg===i)?-1:i;
    Array.prototype.forEach.call($("lg_seg").children,function(x){x.setAttribute("aria-pressed",String(+x.getAttribute("data-s")===activeSeg))});drawSeg()};
  $("exsel").innerHTML=D.segs.map(function(r,i){return'<option value="'+i+'">'+esc(r.name)+'</option>'}).join("");
  $("exsel").onchange=drawEx;drawSeg();drawEx();
}
function drawSeg(){
  var W=640,H=380,P=D.points,xs=P.map(function(p){return p[0]}),ys=P.map(function(p){return p[1]});
  var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
  var X=function(v){return 10+(v-x0)/(x1-x0)*(W-20)},Y=function(v){return 10+(1-(v-y0)/(y1-y0))*(H-20)};
  $("sc_seg").innerHTML='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="PCA projection of comment segments">'+P.map(function(p){
    var on=activeSeg<0||activeSeg===p[2];return'<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[1]).toFixed(1)+'" r="'+(on?3:2)+'" fill="'+PAL[p[2]%PAL.length]+'" fill-opacity="'+(on?.65:.08)+'"/>'}).join("")+'</svg><div class="sub">PCA projection of a 1,500-comment sample. Tap a segment name above to highlight it.</div>';
}
function drawEx(){var n=D.segs[+$("exsel").value].name;$("ex_list").innerHTML=(D.examples[n]||[]).map(function(t){return'<blockquote>'+esc(t)+'</blockquote>'}).join("")}

/* ---------- tab 5: recommendations ---------- */
function renderRecs(){
  $("recs").innerHTML=D.recs.map(function(r){return'<div class="card rec"><b>'+esc(r.area)+'</b> &ndash; '+esc(r.recommendation)+'<small>Evidence: '+esc(r.evidence)+' &middot; Confidence: '+esc(r.confidence)+'</small></div>'}).join("");
  $("tb_bud").innerHTML='<tr><th>Price tier</th><th class="n">Min PKR</th><th class="n">Max PKR</th><th class="n">Budget share %</th><th class="n">Budget PKR</th></tr>'+D.budget.map(function(b){return'<tr><td>'+esc(b.price_tier)+'</td><td class="n">'+nf(b.price_min)+'</td><td class="n">'+nf(b.price_max)+'</td><td class="n">'+nf(b.budget_share_pct,1)+'</td><td class="n">'+nf(b.budget_pkr)+'</td></tr>'}).join("");
  var t=tally(D.cube,2),p=t.m["Price inquiry"]||0,d=t.m["Discount / sale"]||0;
  $("p2note").textContent="Project 2 found no significant sentiment difference between price tiers. Here, "+nf(100*p/t.tot,1)+"% of comments ask for prices and "+nf(100*d/t.tot,1)+"% ask about discounts, so price visibility and sale clarity matter in every tier.";
}

/* ---------- tabs ---------- */
var NAMES=["1 · Content","2 · What the audience wants","3 · Unusual posts","4 · Audience segments","5 · What to do next","About / limits"];
$("tabs").innerHTML=NAMES.map(function(n,i){return'<button role="tab" data-t="'+i+'" aria-selected="'+(i===0)+'">'+n+'</button>'}).join("");
function show(i){Array.prototype.forEach.call($("tabs").children,function(b){b.setAttribute("aria-selected",String(+b.getAttribute("data-t")===i))});
  for(var k=0;k<6;k++)$("t"+k).className="tab"+(k===i?" on":"");history.replaceState(null,"","#t"+i)}
$("tabs").onclick=function(ev){var b=ev.target.closest("button");if(b)show(+b.getAttribute("data-t"))};

(function(){var t=tally(D.cube,2),l=tally(D.cube,1),a=0;D.cube.forEach(function(r){if(D.action.indexOf(r[2])>=0)a+=r[4]});
  $("hs1").textContent=nf(D.meta.comments);$("hs2").textContent=nf(D.meta.videos);$("hs3").textContent=nf(100*a/t.tot,0)+"%";$("hs4").textContent=nf(100*(l.m["Roman Urdu"]||0)/l.tot,0)+"%"})();
renderContent();refresh();renderUnusual();renderSegments();renderRecs();
var h=/^#t([0-5])$/.exec(location.hash);show(h?+h[1]:0);
})();
</script>
</body>
</html>
"""


def main():
    WEB.mkdir(exist_ok=True)
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")          # keep the JSON safe inside <script>
    html = TEMPLATE.replace("__DATA__", data)
    (WEB / "index.html").write_text(html, encoding="utf-8")
    print(f"web/index.html written ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
