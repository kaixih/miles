"use strict";
const REPORT=JSON.parse(document.getElementById("report-data").textContent);
const experiment=REPORT.inputs.experiment,runs=REPORT.inputs.comparison?.runs||[];
const D=REPORT.derived,V=D.validation,P=D.performance,N=D.numerical,profiles=D.profiles;
const palette={rubin:"#087f96",gb300:"#d4773e"},platforms=["rubin","gb300"];
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const e=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(v,d=1)=>finite(v)?v.toLocaleString("en-US",{maximumFractionDigits:d}):"Pending";
const pct=v=>finite(v)?`${(v*100).toFixed(1)}%`:"Pending";
const name=p=>p==="rubin"?"Rubin ES":"GB300";
const run=p=>runs.find(r=>r.label===p);
const mean=(p,k)=>P.paired.statistics?.[p]?.[k]?.mean_seconds;
const pending=(message="Awaiting measurements from this new TRTLLM experiment.")=>`<div class="pending-box"><strong>PENDING / UNMEASURED</strong><p>${e(message)}</p></div>`;
const rateText=k=>{const v=P.ratios[k]?.rubin_time_reduction;return finite(v)?`${pct(Math.abs(v))} ${v>=0?"less":"more"} time on Rubin`:"Comparison pending";};
const slides=[];
function slide(title,kicker,body,footnote="",dark=false){
 const id=slides.length+1;slides.push({id,title});
 document.getElementById("deck").insertAdjacentHTML("beforeend",`<section class="slide${dark?" dark":""}" id="slide-${id}" aria-label="${e(title)}"><p class="kicker">${e(kicker)}</p>${dark?"":`<h2>${e(title)}</h2>`}${body}<div class="footnote">${e(footnote)}</div><div class="page-number">${id}</div></section>`);
}
const coverStatus=REPORT.status==="FINAL"?"FULL RUN + MATCHED PROFILES":REPORT.status==="READY_FOR_REVIEW"?"COMPLETE / READY FOR REVIEW":"PENDING / NEW EXPERIMENT";
const progress=platforms.map(p=>`${name(p)}: ${V.runs[p]?.completed_rollouts||0}/50 rollouts`).join(" · ");
slide("Miles with TRTLLM BF16","Rubin ES vs GB300 · Preliminary results",`
 <h1>Miles with<br>TRTLLM <span>BF16</span></h1>
 <p class="cover-subtitle">Qwen3-30B-A3B · GSM8K<br>Correctness, whole-run performance, prefill and decode.</p>
 <div class="cover-bottom"><div><p class="status ${V.complete?"complete":"pending"}">${e(coverStatus)}</p><p class="small">${e(progress)}</p></div><p class="small">One four-GPU node per platform.<br>Decode CUDA Graph ON throughout.</p></div>`,"",true);

const version=(p,k)=>run(p)?.metadata?.versions?.[k]||`planned ${experiment.planned_te_versions[p]}`;
slide("One recipe, two recorded systems","Experiment",`
 <div class="columns content"><div class="numbered">
 <article><div><h3>Qwen3-30B-A3B standard</h3><p>GSM8K with the same prompt set and strict answer scorer. Fifty rollouts; four optimizer updates per rollout.</p></div></article>
 <article><div><h3>Matched batches and token limits</h3><p>256 prompts × 8 responses; training batch 512. Prompt limit 512; response limit 1,024 tokens.</p></div></article>
 <article><div><h3>Matched inference configuration</h3><p>Four TP1 engines; BF16 TRTLLM MoE, Triton attention. Decode graph ON; prefill graph OFF.</p></div></article>
 </div><div class="stack-card"><h3>Software context</h3>
 <p><strong>Rubin ES</strong><br>Rubin container adaptation<br>TE ${e(version("rubin","transformer-engine"))}</p>
 <p><strong>GB300</strong><br>Upstream Miles base + refit fixes<br>TE ${e(version("gb300","transformer-engine"))}</p>
 <p class="small muted">Both include the TRTLLM weight-refit fixes. Training uses TP1 / EP4 and a 4,096-token budget per GPU.</p>
 <div class="callout">Results compare these complete systems. Library versions differ.</div></div></div>`,"Same training settings; final weights-only checkpoint. Profiling runs separately after training.");

slide("Reward and held-out evaluation","Correctness",`
 <div class="chart-columns content"><div><h3>Training rollout reward</h3><div id="reward-chart" class="chart half"></div></div><div><h3>Held-out GSM8K accuracy</h3><div id="eval-chart" class="chart half"></div></div></div>
 <p class="chart-note">Training reward uses all 2,048 rollout samples. Evaluation uses the same fixed 256 held-out problems.</p>`,"Single runs with stochastic generation; compare the observed learning trajectories, without assuming exact equality.");

slide("Output behavior alongside reward","Correctness",`
 <div class="chart-columns content"><div><h3>Response length</h3><div id="length-chart" class="chart half"></div></div><div><h3>Truncation rate</h3><div id="truncation-chart" class="chart half"></div></div></div>
 <p class="chart-note">These curves show whether reward changes coincide with shorter responses or the 1,024-token output cap.</p>`,"Actual per-rollout observations; no smoothing or interpolation across missing data.");

const numericComplete=V.complete&&platforms.every(p=>N[p]?.all_gradients_finite_positive);
slide("Numerical behavior through all updates","Correctness",`
 <div class="chart-columns content"><div><h3>Training / rollout log-prob difference</h3><div id="logprob-chart" class="chart half"></div></div><div><h3>Gradient norm</h3><div id="gradient-chart" class="chart half"></div></div></div>
 <p class="chart-note">${numericComplete?"Both runs completed 200 updates with finite losses, positive finite gradient norms and normal outcomes on all four ranks.":"Full-update validation is pending; the plots show only recorded observations."}</p>`,"Finite gradients and learning curves are checks for this workload, not a general proof of correctness.");

slide("Whole-step and generation performance","Performance",`
 <div class="chart-columns content"><div><h3>Miles step timer</h3><div id="time-chart" class="chart compact-chart"></div></div><div><h3>Generation</h3><div id="generation-chart" class="chart compact-chart"></div></div></div>
 <div class="metric-strip"><div><span>Mean whole step</span><strong>${fmt(mean("gb300","step"))} → ${fmt(mean("rubin","step"))} s</strong><small>GB300 → Rubin · ${e(rateText("step"))}</small></div><div><span>Mean generation</span><strong>${fmt(mean("gb300","rollout"))} → ${fmt(mean("rubin","rollout"))} s</strong><small>GB300 → Rubin · ${e(rateText("rollout"))}</small></div></div>`,"Steady training rounds; startup and checkpoint-related rounds excluded on both systems. Main runs are unprofiled.");

const actorRatio=P.ratios.actor_train?.gb300_over_rubin;
const actorText=finite(actorRatio)?`Actor update: Rubin takes ${fmt(mean("rubin","actor_train"))} s versus ${fmt(mean("gb300","actor_train"))} s on GB300 (${fmt(actorRatio,2)}× GB300/Rubin time ratio).`:"Actor-update comparison is pending.";
const tokenRate=p=>P.weighted_output_tokens_per_gpu_generation_second[p];
slide("Keep the full performance breakdown visible","Performance",`
 <div id="stage-chart" class="chart stage-chart"></div>
 <p class="stage-finding">${e(actorText)}</p>
 <p class="small muted">Generation throughput: ${fmt(tokenRate("gb300"),0)} → ${fmt(tokenRate("rubin"),0)} output tokens/GPU/s (GB300 → Rubin).</p>`,"Independent recorded stage timers. Their sum is not a wall-time partition of the Miles step; faster generation need not mean a faster whole step.");

function profileSlide(platform,stage){
 const evidence=profiles.runs[platform],item=evidence?.profiles?.[stage],row=item?.selected_forward;
 const heading=`${name(platform)} · ${stage==="prefill"?"prefill":"decode"}`;
 let body;
 if(!item) body=`<div class="new-profile-frame" data-profile-platform="${platform}" data-profile-stage="${stage}">${pending("A fresh TRTLLM trace and screenshot will appear after the separate capture finishes.")}</div>`;
 else {
  const kernel=row.kernel_intervals,fields=row.fields;
  const scope=stage==="prefill"?`${fmt(fields.c_sq??fields.toks,0)} input tokens`:`${fmt(fields.g_sk,0)} KV tokens across the batch`;
  const proof=stage==="decode"?`${row.graph_kernel_correlation.linked_graph_node_kernel_count} graph-node kernels linked to the replay`:`Eager prefill; graph launches absent`;
  body=`<p class="profile-caption">Batch 128 · ${e(scope)} · GPU window ${fmt(row.gpu_duration_ms,3)} ms · kernel coverage ${fmt(kernel.union_ms,3)} ms</p>
   <div class="new-profile-frame" data-profile-platform="${platform}" data-profile-stage="${stage}"><a href="${e(item.attachments.image.url)}" target="_blank"><img src="${e(item.attachments.image.url)}" alt="${e(heading)} actual GPU trace"></a></div>
   <p class="profile-proof">${e(proof)} · <a href="${e(item.attachments.trace.url)}" download>Source trace</a></p>`;
 }
 slide(heading,"Separate initial-policy profiling",body,"Decode graph ON; prefill graph OFF. Instrumented GPU windows include profiler overhead; kernel coverage is not GPU utilization.");
}
profileSlide("rubin","prefill");profileSlide("gb300","prefill");
profileSlide("rubin","decode");profileSlide("gb300","decode");

function profileTable(){
 if(!profiles.complete)return pending("Both platforms' prefill and decode captures are required for this comparison.");
 const rows=["prefill","decode"].map(stage=>{
  const a=profiles.runs.rubin.profiles[stage].selected_forward,b=profiles.runs.gb300.profiles[stage].selected_forward;
  return `<tr><td>${stage==="prefill"?"Prefill":"Decode graph replay"}</td><td>${fmt(a.gpu_duration_ms,3)} ms</td><td>${fmt(b.gpu_duration_ms,3)} ms</td><td>${profiles.matched?`${fmt(b.gpu_duration_ms/a.gpu_duration_ms,2)}×`:"Unmatched"}</td></tr>`;
 }).join("");
 return `<table class="table"><thead><tr><th>GPU window</th><th>Rubin ES</th><th>GB300</th><th>GB300 / Rubin</th></tr></thead><tbody>${rows}</tbody></table>`;
}
const finalReward=platforms.map(p=>`${name(p)} ${pct(N[p]?.last10_reward_mean)}`).join(" · ");
slide("What the measured evidence supports","Results and scope",`
 <div class="content">${profileTable()}</div>
 <div class="conclusion-grid"><div><h3>Correctness evidence</h3><p>${V.complete?`Both 50-rollout runs passed the update checks. Last ten training rewards: ${e(finalReward)}.`:"Awaiting both complete learning trajectories and all update checks."}</p></div><div><h3>Whole-run result</h3><p>${V.complete?`${e(rateText("step"))}. Generation: ${e(rateText("rollout"))}.` : "Whole-step, generation and actor comparisons remain preliminary until both runs finish."}</p></div></div>
 <p class="small muted">${profiles.matched?"Profiles use matched frozen requests and token counts. Their short initial-policy windows do not directly explain the full-run speedup.":e(profiles.reason)}</p>
 <p class="source-link"><a href="report-data.json" download>All measurements and provenance</a> · <a href="asset-manifest.json" download>Artifact hashes</a></p>`,"Software stacks differ. One run per system and short profiles support observation, not a hardware-only causal claim.");

const plotJobs=[];
function draw(id,traces,{xTitle="Rollout ID",yTitle="",percent=false,layout={}}={}){
 const el=document.getElementById(id),valid=traces.filter(t=>t.y.some(finite));
 if(!valid.length){el.innerHTML=pending();return;}
 const options={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Inter,Segoe UI,sans-serif",size:22,color:"#183043"},margin:{l:83,r:20,t:35,b:70},legend:{orientation:"h",x:0,y:1.15,font:{size:21}},showlegend:true,hovermode:"closest",xaxis:{title:{text:xTitle,font:{size:22}},gridcolor:"#e3e9ed",automargin:true},yaxis:{title:{text:yTitle,font:{size:22}},gridcolor:"#e3e9ed",automargin:true,...(percent?{tickformat:".0%",range:[0,1.04]}:{rangemode:"tozero"})},...layout};
 plotJobs.push(Plotly.newPlot(el,valid,options,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["select2d","lasso2d"]}));
}
const lineStyle=p=>({name:name(p),type:"scatter",mode:"lines+markers",connectgaps:false,line:{color:palette[p],width:3},marker:{size:5}});
function curves(key,timing=false){return runs.map(r=>{
 const map=new Map(r.rows.map(row=>[row.rollout_id,row]));
 const ids=map.size?Array.from({length:Math.max(...map.keys())+1},(_,i)=>i):[];
 const eligible=new Set(P.paired.rollout_ids);
 return {...lineStyle(r.label),x:ids,y:ids.map(id=>(!timing||eligible.has(id))&&finite(map.get(id)?.common?.[key])?map.get(id).common[key]:null)};
});}
draw("reward-chart",curves("training_reward_mean"),{yTitle:"Training reward",percent:true});
draw("length-chart",curves("response_length_mean_tokens"),{yTitle:"Mean output tokens"});
draw("truncation-chart",curves("truncated_ratio"),{yTitle:"Fraction",percent:true});
draw("time-chart",curves("step_seconds",true),{yTitle:"Seconds"});
draw("generation-chart",curves("rollout_seconds",true),{yTitle:"Seconds"});
draw("eval-chart",runs.map(r=>{const es=r.rows.flatMap(row=>row.eval.map(v=>({...v,rollout_id:row.rollout_id}))).filter(v=>finite(v.metrics["eval/gsm8k"]));return {...lineStyle(r.label),mode:"markers",x:es.map(v=>v.rollout_id),y:es.map(v=>v.metrics["eval/gsm8k"]),customdata:es.map(v=>v.weight_phase),marker:{color:palette[r.label],size:11,symbol:"diamond"},hovertemplate:"Rollout %{x}<br>%{y:.2%}<br>%{customdata}<extra>%{fullData.name}</extra>"};}),{yTitle:"Held-out accuracy",percent:true});
for(const [id,key,title]of [["gradient-chart","train/grad_norm","Gradient norm"],["logprob-chart","train/train_rollout_logprob_abs_diff","Mean absolute difference"]]){
 draw(id,runs.map(r=>{const values=new Map(r.rows.flatMap(row=>row.train_steps).map(s=>[s.logged_id,s.metrics[key]]));const ids=values.size?Array.from({length:Math.max(...values.keys())+1},(_,i)=>i):[];return {...lineStyle(r.label),x:ids,y:ids.map(i=>finite(values.get(i))?values.get(i):null)};}),{xTitle:"Optimizer update ID",yTitle:title});
}
const stageNames=[["rollout","Generation"],["actor_train","Actor update"],["log_probs","Old log prob"],["ref_log_probs","Reference"],["update_weights","Weight sync"]];
draw("stage-chart",runs.map(r=>({name:name(r.label),type:"bar",x:stageNames.map(x=>x[1]),y:stageNames.map(x=>mean(r.label,x[0])??null),marker:{color:palette[r.label]}})),{xTitle:"",yTitle:"Mean seconds",layout:{barmode:"group"}});
