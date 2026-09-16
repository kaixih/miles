/* Separate CUDA Graph experiment. Measured points come only from bound new runs. */
"use strict";
const REPORT=JSON.parse(document.getElementById("report-data").textContent);
const INPUT=REPORT.inputs, experiment=INPUT.experiment, requested=experiment.requested;
const runs=INPUT.comparison?.runs||[], profiles=INPUT.profiles||{}, diagnostics=INPUT.diagnostics||{};
const paired=REPORT.derived.paired_timing;
const actor=REPORT.derived.actor_profile||{},actorInput=INPUT.actor_profile||{};
const actorAvailable=actor.status==="available"&&(actor.runs||[]).length>0;
const actorMicro=actor.capture_window==="single_forward_backward_microbatch";
const palette=["#087f96","#d4773e"];
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const e=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number=(x,d=1)=>finite(x)?x.toLocaleString("en-US",{maximumFractionDigits:d}):"Pending";
const pct=x=>finite(x)?`${(x*100).toFixed(1)}%`:"Pending";
const name=label=>/^rubin/i.test(label)?"Rubin":/^gb300/i.test(label)?"GB300":label;
const niceName=run=>run.metadata?.display_name||name(run.label);
const forLabel=label=>runs.find(run=>run.label===label);
const updates=run=>(run?.rows||[]).reduce((total,row)=>total+(row.train_steps||[]).length,0);
const complete=run=>run&&run.metadata?.status==="SUCCEEDED"&&run.partial===false&&
  (run.completed_training_rollouts||[]).length===requested.expected_rollouts&&
  updates(run)===requested.expected_rollouts*requested.optimizer_updates_per_rollout;
const pending=(title="PENDING / UNMEASURED",body="No new measurements have been supplied.")=>`<div class="pending-box"><strong>${e(title)}</strong><p>${e(body)}</p></div>`;
const slides=[];
function slide(title,kicker,body,footnote="",dark=false){
  const id=slides.length+1;slides.push({id,title});
  document.getElementById("deck").insertAdjacentHTML("beforeend",`<section id="slide-${id}" class="slide${dark?" dark":""}" aria-label="Slide ${id}: ${e(title)}"><div class="kicker">${e(kicker)}</div>${dark?"":`<h2>${e(title)}</h2>`}${body}<div class="footnote">${footnote}</div><div class="page-number">${String(id).padStart(2,"0")}</div></section>`);
}
const allComplete=runs.length===2&&runs.every(complete);
const capturedReplayVerified=requested.platforms.length===2&&requested.platforms.every(label=>
  (profiles.graph_evidence||[]).some(proof=>proof.run_label===label&&proof.verified===true&&finite(proof.decode_graph_replays)&&proof.decode_graph_replays>0));
slide("Qwen3 with decode CUDA Graph","New experiment / Rubin and GB300",`
  <h1>Qwen3 with<br>decode CUDA Graph</h1>
  <p class="cover-subtitle">One node and four GPUs on each platform.<br>New learning curves and measured graph replay.</p>
  <div class="cover-bottom"><div><p class="status ${allComplete?"complete":runs.length?"partial":"pending"}">${allComplete?"Both main runs complete":runs.length?"New evidence / completion under review":"PENDING / UNMEASURED"}</p><p class="small">${capturedReplayVerified?"Decode replay verified in captured windows.<br>Full-run coverage is not inferred.":"Decode ON and prefill OFF are requested settings.<br>Observed capture and replay need separate evidence."}</p></div><p class="small">${e(experiment.experiment_id)}<br>${e(INPUT.comparison?.collected_at||"No new metrics snapshot")}</p></div>`,"The previous eager report remains a separate, preserved artifact.",true);

slide("The comparison changes decode execution","01 / Intended experiment",`
  <div class="columns content"><div class="rule"><h3>Common new recipe</h3><p>Qwen3-30B-A3B on GSM8K.<br>50 rollouts, 200 optimizer updates.<br>4,096 training tokens per GPU.</p><p class="callout">Decode CUDA Graph ON.<br>Prefill CUDA Graph OFF.</p></div><div class="rule"><h3>Interpretation boundaries</h3><p>Keep each platform’s existing image and kernel choices.</p><p>GB300’s old-to-new change includes decode graph mode, training and log-prob budgets (8,192 → 4,096 tokens/GPU), and the save schedule.</p></div></div>`,"These are planned settings. Actual recipes and replay evidence appear on the following slides. Software differences remain part of the system comparison.");

slide("New runs and completion evidence","02 / Actual run state",`
  <div class="columns content">${requested.platforms.map((label,i)=>{
    const run=forLabel(label),m=run?.metadata||{};
    const health=(INPUT.run_health?.runs||[]).find(item=>item.label===label&&item.run_id===m.run_id);
    return `<div class="rule"><div class="run-status"><h3 class="run-name ${i?"orange":""}">${e(name(label))}</h3><span class="status ${complete(run)?"complete":/FAILED|STOPPED/.test(m.status)?"failed":run?"partial":"pending"}">${e(run?(m.status||"Status unknown"):"PENDING")}</span></div><div class="stats-row"><div><div class="big-number">${run?(run.completed_training_rollouts||[]).length:"—"}</div><div class="number-label">complete rollouts / 50</div></div><div><div class="big-number">${run?updates(run):"—"}</div><div class="number-label">recorded updates / 200</div></div></div><p class="small">${e(m.hardware?.name||"Hardware identity pending")}<br>${finite(m.hardware?.memory_mib_per_gpu)?`${number(m.hardware.memory_mib_per_gpu,0)} MiB/GPU`:"Device memory unrecorded"}${m.hardware?.engineering_sample?" / Engineering sample":""}</p><p class="hash">${e(m.run_id||"Run identity not yet bound")}</p>${health?.issue?`<p class="run-issue">${e(health.issue)}</p>`:""}</div>`;
  }).join("")}</div>`,"Main training, graph verification and diagnostic trace export have independent completion states.");

const recipeFields=[
  ["Model","model",requested.model],["Dataset / reward","dataset_reward",requested.dataset_reward],
  ["Prompts / samples","batch_description",requested.batch],["Response / prompt cap","length_description",requested.response_prompt_caps],
  ["Sampling","sampling_description",requested.sampling],["Training tokens / GPU","max_training_tokens_per_gpu",requested.max_training_tokens_per_gpu],
  ["Log-prob tokens / GPU","max_logprob_tokens_per_gpu",requested.max_logprob_tokens_per_gpu]
];
slide("Recorded recipe versus the requested recipe","03 / Configuration",`
  <table class="table compact content"><thead><tr><th>Setting</th><th>Rubin</th><th>GB300</th></tr></thead><tbody>${recipeFields.map(([title,key,intent])=>`<tr><td>${e(title)}</td>${requested.platforms.map(label=>{const actual=forLabel(label)?.metadata?.recipe?.[key];return `<td>${e(actual??`Planned: ${intent}`)}</td>`;}).join("")}</tr>`).join("")}</tbody></table>
  <p class="chart-note">Planned inference: four TP1 engines, Triton attention/MoE, Torch BF16 GEMM. Actual argv and source hashes remain downloadable.</p>`,"Recipe changes must remain visible. A common training budget does not make different software stacks a hardware-only experiment.");

const graphProof=label=>(profiles.graph_evidence||[]).find(item=>item.run_label===label);
slide("Capture success and replay need evidence","04 / CUDA Graph verification",`
  <p class="subtitle">An enable flag records intent. Runtime observations establish which forwards replayed.</p>
  <table class="table content"><thead><tr><th>Observed evidence</th><th>Rubin</th><th>GB300</th></tr></thead><tbody>${[
    ["Verified decode replay",p=>p?.verified===true?`${number(p.decode_graph_replays,0)} / ${number(p.decode_forwards,0)} forwards`:"Pending verification"],
    ["Decode fallback / unknown",p=>p?.verified===true?`${number(p.decode_fallbacks,0)} / ${number(p.decode_unknown,0)}`:"Pending"],
    ["Prefill forwards / graph replays",p=>p?.verified===true?`${number(p.prefill_forwards,0)} / ${number(p.prefill_graph_replays,0)}`:"Pending"],
    ["Observation scope",p=>p?.scope||"No bounded observation supplied"],
    ["Capture result",p=>p?.capture_status||"Pending"]
  ].map(([label,render])=>`<tr><td>${e(label)}</td>${requested.platforms.map(platform=>`<td>${e(render(graphProof(platform)))}</td>`).join("")}</tr>`).join("")}</tbody></table>`,"Counters describe their recorded scope, not automatically every rollout. Preserve fallback reasons and exact batch/context evidence.");

slide("Training rollout reward","05 / New learning curve",`<p class="subtitle">Only samples generated by the new bound runs appear here.</p><div id="reward-chart" class="chart chart-wide"></div>`,"IDs0–49 mean50 rollouts. Each reward precedes that rollout’s updates. Held-out evaluation appears separately.");
slide("Truncation and response length","06 / Generation behavior",`<div class="chart-columns content"><div><h3>Truncated responses</h3><div id="truncation-chart" class="chart half"></div></div><div><h3>Mean response length</h3><div id="length-chart" class="chart half"></div></div></div>`,"Retain raw sampling counts and status definitions. Curves contain only observations from the new bound runs.");
slide("Gradient signal and held-out evaluation","07 / Learning evidence",`<div class="chart-columns content"><div><h3>Optimizer gradient norm</h3><div id="gradient-chart" class="chart half"></div></div><div><h3>Fixed GSM8K test accuracy</h3><div id="eval-chart" class="chart half"></div></div></div>`,"Finite nonzero gradients support learning activity. One-question differences on256 stochastic evaluations do not establish quality superiority.");
slide("System timing after warmup","08 / Unprofiled performance",`<div class="chart-columns content"><div><h3>Miles step timer</h3><div id="time-chart" class="chart half"></div></div><div><h3>Generation throughput</h3><div id="throughput-chart" class="chart half"></div></div></div><p class="chart-note">Excludes rollout0, incomplete/profiled work and recorded save/I/O intervals. Missing observations stay as gaps.</p>`,"Throughput = retained output tokens / generation seconds / GPUs. Record graph replay and fallback coverage before interpreting the curves.");
slide("Stage timing on the same rollout IDs","09 / Paired system measurements",`<p class="subtitle">${paired.count?`Shared eligible observations: N=${paired.count}`:"PENDING / no shared eligible observations"}</p><div id="stage-chart" class="chart short"></div><p class="chart-note">${paired.count?`Shared IDs: ${e(paired.rollout_ids.join(", "))}`:"Both new runs must supply complete, explicitly unprofiled timing evidence."}</p>`,"Nested stage timers are not additive. A measured difference alone does not identify its hardware, software or host cause.");

function diagnosticRow(label){
  const pair=(diagnostics.pairs||[]).find(item=>item.platform===label);
  const verified=pair?.verified===true;
  const state=verified?"available":pair?.status==="unavailable"?"unavailable":"pending";
  const reason=pair?.status_reason||(verified?"Verified matched OFF/ON pair.":"Awaiting complete matched OFF/ON evidence.");
  return `<tr data-diagnostic-platform="${e(label)}" data-diagnostic-status="${state}" data-diagnostic-verified="${verified}"><td>${e(name(label))}</td><td data-mode="off">${verified?`${number(pair.off?.generation_seconds_mean,3)} s`:"—"}</td><td data-mode="on">${verified?`${number(pair.on?.generation_seconds_mean,3)} s`:"—"}</td><td><strong>${verified?"Verified pair":state==="unavailable"?"Unavailable":"Pending"}</strong><br><span class="small">${e(reason)}</span></td></tr>`;
}
slide("Separate generation-only OFF / ON diagnostic","10 / Attribution",`
  <p class="subtitle">Same initial policy and fixed request/context evidence within each platform.</p>
  <table class="table content"><thead><tr><th>Platform</th><th>OFF mean request</th><th>ON mean request</th><th>Evidence state</th></tr></thead><tbody>${requested.platforms.map(diagnosticRow).join("")}</tbody></table>
  <p class="chart-note">Request time includes prefill + decode, queueing and response delivery. One TP1 engine; input/output token counts and cache policy remain in the raw receipts.</p>`,"Full HTTP generation time is separate from trace-only decode spans and the 50-rollout learning run. No end-to-end Miles speedup is inferred.");

let profileSequence=11;
for(const platform of requested.platforms){
  for(const stage of ["prefill","decode"]){
    // The builder rejects duplicate platform/stage records before rendering.
    const item=(profiles.profiles||[]).find(p=>p.run_label===platform&&p.stage===stage&&p.verified===true);
    const attachment=item?.attachments?.image;
    const graphMode=value=>value===true?"ON":value===false?"OFF":"unknown";
    const condition=item?`Diagnostic condition: decode graph ${graphMode(item.decode_graph)}; prefill graph ${graphMode(item.prefill_graph)}.`:"Diagnostic condition pending.";
    slide(`${name(platform)} ${stage} trace`,`${profileSequence++} / Actual profile evidence`,`
      <p class="subtitle">${e(item?.title||`PENDING / no verified ${stage} capture`)}</p>
      <div class="profile-layout" data-profile-platform="${e(platform)}" data-profile-stage="${stage}" data-profile-verified="${!!item}" data-profile-trace-sha="${e(item?.source_trace_sha256||"")}"><div class="profile-frame">${attachment?.status==="available"?`<img src="${e(attachment.url)}" alt="${e(item.title)}">`:pending(`${stage==="prefill"?"Prefill":"Decode"} trace image pending`,"Only a rendering or screenshot of this stage’s new verified trace belongs here.")}</div><div class="profile-copy"><h3>Recorded scope</h3><p class="profile-condition" style="font-size:20px;line-height:1.4;margin-bottom:14px">${e(condition)}</p><p>${e(item?.scope||`${stage==="prefill"?"EXTEND/prefill":"DECODE"} counts, batch sizes, context and graph replay are unmeasured.`)}</p>${item?.observations?.length?`<ul>${item.observations.slice(0,3).map(value=>`<li>${e(value)}</li>`).join("")}</ul>`:""}${item?.attachments?.trace?.status==="available"?`<a href="${e(item.attachments.trace.url)}" download>Download ${stage} source trace</a>`:""}</div></div>`,e(item?.caption||"No kernel or causal conclusion is inferred while this stage’s capture is pending. Profiler overhead must remain visible."));
  }
}

const prior=experiment.previous_report,priorHref=prior?.href&&!/[:\\]/.test(prior.href)?prior.href:null;
const finalStageLabels={rollout:"generation",actor_train:"actor update",log_probs:"old log prob",ref_log_probs:"reference",update_weights:"weight sync"};
const generationMeans=requested.platforms.map(label=>paired.statistics?.[label]?.rollout);
const measuredFinal=allComplete&&paired.count>0&&generationMeans.every(value=>value?.paired_metric_available===true&&finite(value.mean_seconds));
const finalGap=paired.largest_observed_stage_gap;
const gapMeans=finalGap?requested.platforms.map(label=>paired.statistics?.[label]?.[finalGap.stage]?.mean_seconds):[];
const gapMeasured=measuredFinal&&gapMeans.length===2&&gapMeans.every(finite);
const gapDifference=gapMeasured?gapMeans[1]-gapMeans[0]:null;
const gapLonger=gapMeasured?name(requested.platforms[gapDifference>=0?1:0]):null;
const finalFindings=measuredFinal?`
  <h3>Measured system takeaways</h3>
  <p class="small">Mean generation: ${requested.platforms.map((label,i)=>`${e(name(label))} ${number(generationMeans[i].mean_seconds,1)} s`).join("; ")}.<br>Same ${paired.count} eligible rollout IDs.</p>
  ${gapMeasured?`<p class="small">Largest stage gap: ${e(finalStageLabels[finalGap.stage]||finalGap.stage)}, ${e(gapLonger)} ${number(Math.abs(gapDifference),1)} s longer per rollout.</p>`:'<p class="small">Stage gap awaits complete paired timing evidence.</p>'}
  <p class="small">System timers; nested stages are not additive. Software differs, and GPU causality remains unresolved.</p>`:
  `<h3>${allComplete?"Main runs complete":"Conclusions require measurements"}</h3><p class="small">${allComplete?"Paired timing remains pending. Graph verification and diagnostic export have independent evidence states.":"New run completion and performance remain under review. Pending sections carry no measured conclusion."}</p><p class="small">${e(experiment.runtime_note)}</p>`;
function actorFinalPanel(){
  const records=actorInput.runs||[];
  const verified=records.filter(r=>r.verified===true);
  const findings=actor.findings||[];
  const proof=REPORT.provenance.find(p=>p.section==="actor_profile");
  return `<p class="subtitle actor-final-subtitle">${actor.matching_status==="matched_workload"?"Matched workload; post-warmup state equality is not assumed.":actorMicro?"Diagnostic / unmatched · one forward/backward window per available platform.":"Diagnostic / unmatched · missing paired evidence remains pending."}</p>
  <div class="actor-timelines">${requested.platforms.map(label=>{
    const record=verified.find(r=>r.run_label===label), image=record?.attachments?.image;
    return `<figure class="actor-timeline" data-actor-timeline-platform="${e(label)}" data-actor-verified="${!!record}" data-actor-trace-sha="${e(record?.source_trace_sha256||"")}" data-actor-ranks="${e(record?.rank_ids.join(",")||"")}" data-actor-updates="${e(record?.samples.map(s=>s.update_id).join(",")||"")}" data-actor-microbatch="${record?.microbatch_index??""}"><h3>${e(name(label))} · ${record?actorMicro?"one microbatch":`rank ${e(record.rank_ids.join(","))}, update ${e(record.samples.map(s=>s.update_id).join(","))}`:"capture pending"}</h3><div class="actor-timeline-frame">${image?.status==="available"?`<a href="${e(image.url)}" target="_blank"><img src="${e(image.url)}" alt="${e(name(label))} actual ${actorMicro?"forward/backward microbatch":"actor-update"} trace; click for full resolution"></a>`:pending(record?"Timeline image pending":"Actor capture pending",record?"Verified source trace retained; no image has been supplied.":"No verified actor window is available for this platform.")}</div><figcaption>${record?`<a href="${e(record.attachments.trace.url)}" download>Source trace</a> <span class="hash">${e(record.source_trace_sha256.slice(0,12))}</span> · ${(record.verified_receipts||[]).map(r=>`<a href="${e(r.url)}" download>audit</a>`).join(" · ")}`:"No measured actor breakdown or timing ratio is inferred."}</figcaption></figure>`;
  }).join("")}</div>
  <div class="actor-audited-findings">${findings.length?findings.map((f,i)=>`<p data-actor-finding="${i}">${e(f.text)} <a class="actor-finding-receipt" href="${e(actorInput.findings[i].verified_receipts[0].url)}" download>Evidence</a></p>`).join(""):'<p>No audited actor interpretation supplied; the images establish the recorded scope only.</p>'}</div>
  ${actor.interpretation_limits?`<p class="actor-final-limits">${e(actor.interpretation_limits)}</p>`:""}
  <p class="actor-final-evidence"><a href="report-data.json" download>Full recipes and evidence</a> · <a href="asset-manifest.json" download>Checksums</a> · actor input <span class="hash">${e(proof?.sha256?.slice(0,12)||"unknown")}</span>${priorHref?` · <a href="${e(priorHref)}">Preserved eager report</a>`:""}</p>`;
}
if(actorAvailable){
  slide(actorMicro?"One actor microbatch":"Actor windows, evidence and limits","15 / Actor diagnostic evidence",actorFinalPanel(),actorMicro?"Rank 0; profiler enabled; optimizer/final gradient sync excluded. Kernel coverage is not GPU utilization.":"Instrumented actor updates are separate from the main stage timers. Ranges can overlap; rank 0 does not represent every EP rank or isolate a hardware cause.");
}else{
slide("Evidence, limits and the preserved baseline","15 / Provenance",`
  <div class="columns content"><div><h3>New experiment inputs</h3>${REPORT.provenance.map(p=>`<div class="input-evidence-row"><span>${e(p.section)}</span><span class="hash">${p.status==="loaded"?e(p.sha256.slice(0,12)):"PENDING"}</span></div>`).join("")}<p class="small" style="margin-top:24px"><a href="report-data.json" download>Complete recipes, raw observations and hashes</a><br><a href="asset-manifest.json" download>Local artifact checksums</a></p></div><div>${finalFindings}${priorHref?`<p class="small"><a href="${e(priorHref)}">Original eager report: preserved separately</a></p>`:""}<p class="small muted">${e(prior?.scope_note||"")}</p></div></div>`,"The previous report remains separate. Downloadable evidence preserves exact run identity and does not merge attempts.");
}

const plotJobs=[];
function draw(id,traces,{yTitle="",xTitle="Rollout ID",percent=false,layout={}}={}){
  const el=document.getElementById(id),usable=traces.filter(t=>t.y.some(finite));
  if(!usable.length){el.innerHTML=pending();return;}
  const config={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Inter,Segoe UI,sans-serif",size:23,color:"#183043"},margin:{l:90,r:20,t:35,b:80},showlegend:true,legend:{orientation:"h",x:0,y:1.15,font:{size:21}},hovermode:"closest",xaxis:{title:{text:xTitle,font:{size:22}},gridcolor:"#e3e9ed",automargin:true},yaxis:{title:{text:yTitle,font:{size:22}},gridcolor:"#e3e9ed",automargin:true,...(percent?{tickformat:".0%",range:[0,1.04]}:{rangemode:"tozero"})},...layout};
  plotJobs.push(Plotly.newPlot(el,usable,config,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["select2d","lasso2d"]}));
}
function eligible(run,row){return row.rollout_id!==0&&row.training_stage_complete===true&&row.profiled===false&&row.unprofiled_timing_eligible===true&&(run.completed_training_rollouts||[]).includes(row.rollout_id)&&!(run.metadata?.exclude_timing_rollouts||[]).includes(row.rollout_id);}
function lines(key,steady=false){return runs.map((run,i)=>{
  const map=new Map((run.rows||[]).map(row=>[row.rollout_id,row]));
  const ids=[...map.keys()];const rows=steady&&ids.length?Array.from({length:Math.max(...ids)+1},(_,id)=>map.get(id)||{rollout_id:id}):(run.rows||[]);
  return {name:niceName(run),type:"scatter",mode:"lines+markers",connectgaps:false,x:rows.map(r=>r.rollout_id),y:rows.map(r=>(!steady||eligible(run,r))&&finite(r.common?.[key])?r.common[key]:null),line:{color:palette[i%2],width:3},marker:{size:7},hovertemplate:"Rollout %{x}<br>%{y:.5g}<extra>%{fullData.name}</extra>"};
});}
draw("reward-chart",lines("training_reward_mean"),{percent:true,yTitle:"Training reward"});
draw("truncation-chart",lines("truncated_ratio"),{percent:true,yTitle:"Fraction"});
draw("length-chart",lines("response_length_mean_tokens"),{yTitle:"Output tokens"});
draw("time-chart",lines("step_seconds",true),{yTitle:"Seconds"});
draw("throughput-chart",lines("output_tokens_per_gpu_generation_second",true),{yTitle:"Output tokens / GPU / s"});
draw("gradient-chart",runs.map((run,i)=>{const steps=(run.rows||[]).flatMap(r=>r.train_steps||[]);return {name:niceName(run),type:"scatter",mode:"lines+markers",x:steps.map(s=>s.logged_id),y:steps.map(s=>s.metrics?.["train/grad_norm"]??null),connectgaps:false,line:{color:palette[i%2],width:3},marker:{size:5}};}),{xTitle:"Optimizer update ID",yTitle:"Gradient norm"});
draw("eval-chart",runs.map((run,i)=>{const events=(run.rows||[]).flatMap(r=>(r.eval||[]).map(event=>({...event,rollout_id:r.rollout_id}))).filter(event=>finite(event.metrics?.["eval/gsm8k"]));return {name:niceName(run),type:"scatter",mode:"markers",x:events.map(ev=>ev.rollout_id),y:events.map(ev=>ev.metrics["eval/gsm8k"]),customdata:events.map(ev=>ev.weight_phase||"unknown"),marker:{color:palette[i%2],symbol:"diamond",size:11},hovertemplate:"Rollout %{x}<br>%{y:.2%}<br>Weight phase: %{customdata}<extra>%{fullData.name}</extra>"};}),{percent:true,yTitle:"Held-out accuracy"});
const stageNames=[["rollout","Generation"],["actor_train","Actor update"],["log_probs","Old log prob"],["ref_log_probs","Reference"],["update_weights","Weight sync"]];
draw("stage-chart",runs.map((run,i)=>({name:niceName(run),type:"bar",x:stageNames.map(([,label])=>label),y:stageNames.map(([key])=>paired.statistics[run.label]?.[key]?.paired_metric_available?paired.statistics[run.label][key].mean_seconds:null),marker:{color:palette[i%2]}})),{xTitle:"",yTitle:"Mean seconds / shared cohort",layout:{barmode:"group"}});
