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
<title>Khaadi Audience &amp; Campaign Intelligence Engine</title>
<meta name="description" content="Clustering-based audience and campaign intelligence for Khaadi clothing, built from 21,514 public YouTube comments.">
<style>
:root{--bg:#fafaf8;--card:#fff;--ink:#1d2433;--muted:#667085;--line:#e4e7ec;--accent:#2b6cb0;--accent2:#dd6b20;--bar:#2b6cb0;--red:#d64545;--green:#2e9e5b;--chip:#eef2f7}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--card:#171c26;--ink:#e8ecf3;--muted:#98a2b3;--line:#2a3140;--accent:#63a4f0;--accent2:#f0a060;--bar:#63a4f0;--chip:#232b3a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.15rem;margin:1.2rem 0 .4rem}h3{font-size:1rem;margin:1rem 0 .3rem}
.sub{color:var(--muted);font-size:.85rem}.sub a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:1.4rem}.kpi span{color:var(--muted);font-size:.78rem}
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
label{font-size:.8rem;color:var(--muted);display:flex;flex-direction:column;gap:2px}
select{padding:7px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;max-width:100%}
.tabs{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;border-bottom:1px solid var(--line);margin-top:10px;position:sticky;top:0;background:var(--bg);z-index:5}
.tabs button{border:0;background:none;color:var(--muted);padding:9px 10px;font:inherit;font-size:.88rem;white-space:nowrap;border-bottom:2px solid transparent;cursor:pointer}
.tabs button[aria-selected=true]{color:var(--accent);border-color:var(--accent);font-weight:600}
section.tab{display:none}section.tab.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
.bars .row{display:grid;grid-template-columns:minmax(90px,36%) 1fr;gap:8px;align-items:center;margin:5px 0;font-size:.85rem}
.bars .track{background:var(--chip);border-radius:6px;height:22px;position:relative;overflow:hidden}
.bars .fill{background:var(--bar);height:100%;border-radius:6px}
.bars .val{position:absolute;right:6px;top:0;line-height:22px;font-size:.78rem;color:var(--ink)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.82rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;white-space:nowrap}
.warn{background:#fff4e5;color:#7a4a00;border:1px solid #f3cf9a;border-radius:8px;padding:8px 10px;font-size:.84rem;margin:8px 0}
@media (prefers-color-scheme:dark){.warn{background:#3a2c14;color:#f3d29a;border-color:#6a4c1c}}
.rec{border-left:4px solid var(--accent);}
.rec small{color:var(--muted);display:block;margin-top:4px}
svg{width:100%;height:auto;display:block}
#sc_un svg,#sc_seg svg{max-width:760px;margin:0 auto}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.legend button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:14px;padding:3px 9px;font-size:.75rem;cursor:pointer}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.legend button[aria-pressed=true]{outline:2px solid var(--accent)}
.sel{font-size:.85rem}.sel a{color:var(--accent)}
footer{color:var(--muted);font-size:.78rem;margin:26px 0 10px}
blockquote{margin:6px 0;padding:6px 10px;background:var(--chip);border-radius:8px;font-size:.85rem}
</style>
</head>
<body>
<div class="wrap">
<h1>Khaadi Audience &amp; Campaign Intelligence Engine</h1>
<div class="sub">Project 3 &middot; Big Data Analytics &middot; builds on Project 1 (comment sentiment) and Project 2 (price-tier budget). Public YouTube comments on Khaadi clothing videos: an observational sample, not Khaadi's own analytics.
&middot; <a id="lg" href="#">GitHub</a> &middot; <a id="ls" href="#">Interactive Streamlit version</a></div>

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

<footer>Waqas Noor &middot; F2023376280 &middot; Section A2 &middot; UMT Lahore</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var $=function(id){return document.getElementById(id)};
var esc=function(s){return String(s).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})};
var nf=function(x,d){return Number(x).toLocaleString("en-US",{maximumFractionDigits:d==null?0:d,minimumFractionDigits:d==null?0:d})};
var PAL=["#2b6cb0","#dd6b20","#2f855a","#805ad5","#d53f8c","#b7791f","#319795","#718096","#e53e3e"];
$("lg").href=D.meta.github;$("ls").href=D.meta.streamlit;

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
  vs.filter(function(v){return!v[5]}).forEach(function(v,i){s+='<circle data-i="'+vs.indexOf(v)+'" cx="'+X(v[3])+'" cy="'+Y(v[4])+'" r="4" fill="#98a2b3" fill-opacity=".7"/>'});
  vs.filter(function(v){return v[5]}).forEach(function(v){s+='<circle data-i="'+vs.indexOf(v)+'" cx="'+X(v[3])+'" cy="'+Y(v[4])+'" r="6" fill="#d64545" stroke="#fff" stroke-width="1"/>'});
  s+='</svg><div class="legend"><span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#98a2b3;margin-right:5px"></i>normal</span><span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#d64545;margin-right:5px"></i>unusual</span></div>';
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
