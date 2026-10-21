"""Self-contained report presentation. Used only by opal.report."""

import re


def render(values):
    # One pass: record text containing a placeholder must remain literal text.
    return re.sub(r"@([a-z_]+)@", lambda m: values[m[1]], PAGE)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
@refresh@<title>OPAL run report</title>
<style>
:root {color-scheme:light dark; --bg:#fff; --ink:#202124; --muted:#63676d; --rule:#d9dce0; --accent:#1f4e8c; --code:#f5f6f7; --add:#e8f2ea; --del:#f8e9e7}
@media(prefers-color-scheme:dark) {:root {--bg:#17191c; --ink:#eaedf0; --muted:#b0b5bc; --rule:#3c4249; --accent:#94baf0; --code:#202429; --add:#223d2c; --del:#482a29}}
* {box-sizing:border-box}
body {margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
main {max-width:1040px; margin:auto; padding:30px 22px 65px; overflow-wrap:anywhere}
header {display:flex; justify-content:space-between; align-items:baseline; gap:12px}
h1 {font-size:1.65rem; line-height:1.25; margin:0; font-weight:650}
h2 {font-size:1.15rem; margin:1.7rem 0 .6rem; font-weight:650}
h3 {font-size:1rem; margin:1.2rem 0 .3rem}
p {margin:.55rem 0}
.goal {color:var(--muted); margin:.4rem 0 1.2rem}
#overview p {margin:.65rem 0; max-width:92ch}
a {color:var(--accent); text-underline-offset:3px}
button {font:inherit; color:var(--accent); background:none; border:0; padding:0; cursor:pointer; text-decoration:underline; text-underline-offset:3px}
button[aria-pressed=true] {color:var(--ink); text-decoration:none}
nav {white-space:nowrap; font-size:.85rem}
[data-lang=zh] {display:none}
html[lang=zh-CN] [data-lang=en] {display:none}
html[lang=zh-CN] [data-lang=zh] {display:revert}
figure {margin:.5rem 0 1.4rem}
svg {display:block; width:100%; height:auto; overflow:visible}
svg text {font:12px ui-sans-serif,system-ui,sans-serif; fill:var(--muted)}
.grid {stroke:var(--rule); stroke-width:.6}
.loss {fill:none; stroke:var(--ink); stroke-width:1.5; stroke-linejoin:round}
.loss.alt {stroke:var(--accent)}
.mark {cursor:pointer; outline-offset:3px}
.mark line {stroke:var(--accent); stroke-width:1; opacity:.5}
.mark circle {stroke:var(--accent); stroke-width:1.5; fill:var(--bg)}
.mark text {fill:var(--accent); font-size:11px; font-weight:650; pointer-events:none}
.mark:hover circle,.mark:focus circle {stroke-width:3}
.kept {stroke:var(--muted); opacity:.3}
#readout,figcaption,.muted {font-size:.85rem; color:var(--muted)}
#readout {min-height:3.2em; margin:.4rem 0}
#readout a {margin-left:.5em; white-space:nowrap}
details {border-top:1px solid var(--rule); padding:.85rem 0}
details details {margin-top:.5rem; padding:.65rem 0}
summary {cursor:pointer; font-weight:550}
summary .muted {font-weight:400}
pre {white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.55 ui-monospace,Menlo,monospace; background:var(--code); padding:12px; margin:.5rem 0}
pre.diff .a {background:var(--add); display:block}
pre.diff .r {background:var(--del); display:block}
.flag b {color:var(--ink)}
table {width:100%; border-collapse:collapse; font-size:.86rem; line-height:1.5}
th {text-align:left; font-weight:600; border-bottom:1px solid var(--ink); padding:.55rem .45rem; vertical-align:bottom}
td {vertical-align:top; border-bottom:1px solid var(--rule); padding:.7rem .45rem}
td.number,td.when {white-space:nowrap}
td.what {width:35%} td.why {width:32%} td.tags {width:14%; color:var(--muted); font-size:.8rem}
dt {font-weight:550; margin-top:.7rem} dd {margin:.1rem 0 .5rem; font-size:.9rem}
noscript {display:block; color:var(--muted)}
@media(max-width:640px) {
 main {padding:22px 16px 50px} h1 {font-size:1.4rem} body {font-size:15px}
 table thead {display:none} table tr {display:grid; grid-template-columns:2.2em 1fr; padding:.55rem 0; border-bottom:1px solid var(--rule)}
 table td {display:block; border:0; padding:.15rem 0; width:auto!important}
 td.number {grid-column:1; grid-row:1 / span 4} td.when,td.what,td.why,td.tags {grid-column:2; white-space:normal}
 td[data-label]::before {content:attr(data-label) ": "; color:var(--muted)}
 #readout {min-height:4.5em}
}
</style></head><body><main>
<header><h1><span data-lang="en">OPAL run report</span><span data-lang="zh">OPAL 运行报告</span></h1>
</header>
<p class="goal">@goal@</p>
<section id="overview" aria-label="Run overview">@overview@</section>
<h2><span data-lang="en">Training loss</span><span data-lang="zh">训练损失</span></h2>
<figure>
<svg id="chart" role="group" aria-label="Training loss and applied decisions"></svg>
<p id="readout" role="status" aria-live="polite"></p>
<figcaption>@caption@</figcaption>
<noscript>Enable JavaScript to draw the loss and analysis plots. The overview, changes and full records below are available without it.</noscript>
</figure>
@table@
<details id="analysis"><summary><span data-lang="en">Analysis plots</span><span data-lang="zh">分析图表</span></summary><div id="small"></div></details>
<details id="decision-details"><summary><span data-lang="en">Decision messages and answers</span><span data-lang="zh">决策消息与回答</span></summary>@decisions@</details>
<details id="code-details"><summary><span data-lang="en">Code diffs</span><span data-lang="zh">代码改动</span></summary>@diffs@</details>
<details id="facts"><summary><span data-lang="en">Run facts</span><span data-lang="zh">运行记录</span></summary>@facts@</details>
</main>
<script>
'use strict';
const L = @losses@, D = @ticks@, S = @small@, T = @total@;
const ns = 'http://www.w3.org/2000/svg';
let lang = 'en', selected = null;
const tr = (en, zh) => lang === 'zh' ? zh : en;
const fmt = v => Math.abs(v) > 0 && (Math.abs(v) < .001 || Math.abs(v) >= 1e5) ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
function readout(d) {
 const out = document.getElementById('readout'); out.replaceChildren();
 if (!d) {out.textContent = tr('Hover, focus or tap a number for the change and the agent’s reason.', '悬停、聚焦或轻触编号，查看改动与 agent 的理由。'); return;}
 out.textContent = tr('Decision ', '决策 ') + d.k + ': ' + d.label;
 const link = document.createElement('a'); link.href = '#d' + d.k; link.textContent = tr('Read full decision', '查看完整决策'); out.append(link);
}
function draw(svg, series, label, main) {
 svg.replaceChildren();
 const W = Math.max(280, svg.clientWidth || 860), H = main ? 300 : 205;
 svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
 const m = {l:59, r:18, t:main ? 64 : 26, b:42};
 const el = (t, a, txt, parent=svg) => {const e=document.createElementNS(ns,t); for(const k in a)e.setAttribute(k,a[k]); if(txt!=null)e.textContent=txt; parent.appendChild(e); return e;};
 const pts = series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[0]) && Number.isFinite(p[1]));
 if (!pts.length) {el('text',{x:8,y:36},tr('No training loss recorded.', '未记录训练损失。')); return;}
 // Use the full recorded range; do not clip spikes or replace non-finite losses with zero.
 let x1=1, y0=Infinity, y1=-Infinity;
 for(const p of pts) {x1=Math.max(x1,p[0]); y0=Math.min(y0,p[1]); y1=Math.max(y1,p[1]);}
 if(main)for(const d of D)for(const x of [d.requested,d.applied])if(Number.isFinite(x))x1=Math.max(x1,x);
 x1=Math.max(x1,T || 0);
 if(y0===y1) {const pad=Math.abs(y0)*.05 || .5; y0-=pad; y1+=pad;}
 const X=v=>m.l+v/x1*(W-m.l-m.r), Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
 for(let i=0;i<=3;i++) {const v=y0+(y1-y0)*i/3; el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:'grid'}); el('text',{x:m.l-7,y:Y(v)+4,'text-anchor':'end'},fmt(v));}
 const nx=W<500?2:4;
 for(let i=0;i<=nx;i++) {const v=Math.round(x1*i/nx); el('text',{x:X(v),y:H-23,'text-anchor':'middle'},v.toLocaleString());}
 el('text',{x:m.l,y:15},label);
 el('text',{x:(m.l+W-m.r)/2,y:H-3,'text-anchor':'middle'},tr('Update','更新步'));
 for(let i=0;i<series.length;i++) {
  let path='', pen=false;
  for(const p of series[i].points) {
   if(!Number.isFinite(p[0]) || !Number.isFinite(p[1])) {pen=false; continue;}
   path+=(pen?'L':'M')+X(p[0]).toFixed(2)+','+Y(p[1]).toFixed(2); pen=true;
  }
  el('path',{class:i?'loss alt':'loss',d:path});
  if(pts.length===1)el('circle',{cx:X(pts[0][0]),cy:Y(pts[0][1]),r:3,fill:'var(--ink)'});
  if(series.length>1) {const last=series[i].points.filter(p=>Number.isFinite(p[1])).at(-1); if(last)el('text',{x:W-m.r,y:Y(last[1])-5,'text-anchor':'end'},series[i].name);}
 }
 if(!main)return;
 let previous=-Infinity, lane=0;
 for(const d of D) {
  const u=d.applied ?? d.requested; if(!Number.isFinite(u))continue;
  const x=X(u);
  if(!d.changed) {
   if(d.status==='kept') {const tick=el('line',{x1:x,x2:x,y1:H-m.b-7,y2:H-m.b,class:'kept'}); el('title',{},tr('Decision ','决策 ')+d.k+': '+d.label,tick);}
   continue;
  }
  lane=x-previous<28?(lane+1)%2:0; previous=x;
  const y=32+lane*24;
  const near=pts.reduce((a,p)=>Math.abs(p[0]-u)<Math.abs(a[0]-u)?p:a,pts[0]);
  const g=el('g',{class:'mark',tabindex:0,role:'button','aria-label':tr('Decision ','决策 ')+d.k+': '+d.label});
  el('title',{},d.label,g);
  el('line',{x1:x,x2:x,y1:y+10,y2:Y(near[1])},null,g);
  el('circle',{cx:x,cy:y,r:11},null,g); el('text',{x:x,y:y+4,'text-anchor':'middle'},d.k,g);
  const show=()=>{selected=d; readout(d);};
  g.addEventListener('mouseenter',show); g.addEventListener('focus',show); g.addEventListener('click',show);
  g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ') {e.preventDefault(); show();}});
 }
}
function charts() {
 draw(document.getElementById('chart'),[{name:'loss',points:L}],tr('Training loss ↓','训练损失 ↓'),true);
 const box=document.getElementById('small'); box.replaceChildren();
 if(document.getElementById('analysis').open) {
  if(!S.length) {const p=document.createElement('p'); p.textContent=tr('No additional metric series recorded.','未记录其他指标序列。'); box.append(p);}
  for(const s of S) {const svg=document.createElementNS(ns,'svg'); svg.setAttribute('role','img'); svg.setAttribute('aria-label',s.label); box.append(svg); draw(svg,s.series,s.label,false);}
 }
 readout(selected);
}
function language(value) {
 lang=value; document.documentElement.lang=value==='zh'?'zh-CN':'en';
 charts();
}
function reveal(id) {const target=document.getElementById(id); if(!target)return; let p=target; while(p) {if(p.tagName==='DETAILS')p.open=true; p=p.parentElement;} target.scrollIntoView({block:'start'});}
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#d"]'); if(a) {e.preventDefault(); reveal(a.getAttribute('href').slice(1));}});
document.getElementById('analysis').addEventListener('toggle',charts);
let resize; window.addEventListener('resize',()=>{clearTimeout(resize); resize=setTimeout(charts,100);});
language(lang);
if(location.hash)reveal(decodeURIComponent(location.hash.slice(1)));
</script></body></html>"""
