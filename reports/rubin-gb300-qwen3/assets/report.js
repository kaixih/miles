/* Offline evidence deck. Only supplied records create data points. */
"use strict";
const REPORT = JSON.parse(document.getElementById("report-data").textContent);
const INPUT = REPORT.inputs;
const comparison = INPUT.comparison || {};
const runs = comparison.runs || [];
const build = INPUT.build || {};
const profiles = INPUT.profiles || {};
const palette = ["#087f96", "#d4773e", "#726198", "#528269"];
const e = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite = value => typeof value === "number" && Number.isFinite(value);
const number = (value, digits=1) => finite(value) ? value.toLocaleString("en-US", {maximumFractionDigits:digits, minimumFractionDigits:digits}) : "Pending";
const pct = value => finite(value) ? `${number(value*100)}%` : "Pending";
const textValue = value => value == null ? "Not recorded" : typeof value === "object" ? JSON.stringify(value) : String(value);
const pending = (title="Evidence pending", message="This snapshot contains no measurements for this section.") => `<div class="pending-box"><strong>${e(title)}</strong><p>${e(message)}</p></div>`;
const last = (run, key) => [...(run.rows || [])].reverse().find(row => finite(row.common?.[key]))?.common?.[key];
const niceName = run => run.metadata?.display_name || (/rubin/i.test(run.label) ? "Rubin" : /gb300/i.test(run.label) ? "GB300" : run.label);
const healthFor = run => run?.metadata?.run_health
  ? (typeof run.metadata.run_health==="object" ? run.metadata.run_health : {run_health:run.metadata.run_health,issue:run.metadata.issue})
  : (INPUT.run_health?.runs||[]).find(r=>r.label===run?.label && r.run_id===run?.metadata?.run_id) || null;
const activeIssues=runs.filter(run=>healthFor(run)?.issue);
function status(run) {
  if (!run) return {label:"Pending", css:"pending"};
  const s = run.metadata?.status;
  if (["FAILED", "STOPPED", "CANCELED", "TIMED_OUT"].includes(s)) return {label:s.replaceAll("_", " "), css:"failed"};
  if (healthFor(run)?.issue) return {label:({recovery_pending:"Recovery pending",training_pending_after_recovery:"Training pending",partial_inference_recovery:"Partial recovery",workers_registered_workload_pending:"Workload pending"}[healthFor(run).run_health]||"Investigating"), css:"pending"};
  if (run.partial === false && s === "SUCCEEDED") return {label:"Complete", css:"complete"};
  return {label: s === "SUCCEEDED" ? "Terminal / incomplete evidence" : s || "Partial snapshot", css:"partial"};
}
function table(headers, rows, cls="") {
  return `<table class="table ${cls}"><thead><tr>${headers.map(v=>`<th>${e(v)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
const slides = [];
function slide(title, kicker, html, foot="", cls="") {
  const id = slides.length + 1;
  slides.push({title, id});
  document.getElementById("deck").insertAdjacentHTML("beforeend", `<section class="slide ${cls}" id="slide-${id}" aria-label="Slide ${id}: ${e(title)}" aria-hidden="true"><div class="kicker">${e(kicker)}</div>${cls.includes("cover") ? "" : `<h2>${e(title)}</h2>`}${html}<div class="footnote">${e(foot)}</div><div class="page-number">${String(id).padStart(2,"0")}</div></section>`);
}

// 1. Cover: no performance conclusion appears before results exist.
const finished = runs.length >= 2 && runs.every(run => !run.partial && run.metadata?.status === "SUCCEEDED");
slide("Qwen3 on Rubin & GB300", "Experiment report / September 2026", `
  <h1>Qwen3 on Rubin<br><span>&amp;</span> GB300</h1>
  <p class="cover-subtitle">A four-GPU Miles / SGLang comparison<br>Learning behavior, performance, and kernel evidence</p>
  <div class="cover-bottom"><div><p class="status ${activeIssues.length ? "pending" : finished ? "complete" : "partial"}">${activeIssues.length ? activeIssues.map(run=>`${niceName(run)}: ${status(run).label.toLowerCase()}`).join(" / ") : finished ? "Both runs complete" : runs.length ? "Experiment in progress" : "Awaiting experiment evidence"}</p><p class="small muted">Qwen3-30B-A3B / GSM8K / GRPO</p></div><p class="small">Rubin uses the preserved CUDA 13.4 build.<br>GB300 uses the upstream Miles container.</p></div>`, `Metrics snapshot ${(comparison.collected_at||REPORT.generated_at).replace("T"," ").slice(0,19)} UTC${INPUT.run_health?.observed_at?`; operational note ${INPUT.run_health.observed_at.replace("T"," ").slice(0,19)} UTC`:""}`, "dark cover");

// 2. System status: missing second run stays visibly pending.
const displayRuns = runs.length ? [...runs.slice(0,2)] : [null,null];
while (displayRuns.length < 2) displayRuns.push(null);
slide("One node and four GPUs per system", "01 / Comparison scope", `
  <p class="subtitle">The experiment compares complete software stacks on the same learning task.</p>
  <div class="columns content">${displayRuns.map((run,i)=>{
    const state=status(run), meta=run?.metadata||{};
    const name=run ? niceName(run) : i===0 ? "Rubin" : "GB300";
    return `<div class="rule"><div class="run-status"><h3 class="run-name ${i===1?"orange":""}">${e(name)}</h3><span class="status ${state.css}">${e(state.label)}</span></div>
      <div class="stats-row"><div class="metric"><div class="big-number">${run ? e((run.completed_training_rollouts||[]).length) : "—"}</div><div class="number-label">complete training rollouts${meta.expected_rollouts ? ` / ${e(meta.expected_rollouts)}`:""}</div></div><div class="metric"><div class="big-number">${pct(run && last(run,"training_reward_mean"))}</div><div class="number-label">latest training reward</div></div></div>
      <div class="run-details hardware-details"><div class="hardware-name">${e(meta.hardware?.name || "Hardware not recorded")}</div><div>${finite(meta.hardware?.memory_mib_per_gpu) ? `${number(meta.hardware.memory_mib_per_gpu,0)} MiB per GPU` : "GPU memory not recorded"}${meta.hardware?.engineering_sample === true ? " · Engineering sample" : ""}</div><div>${e(meta.gpus ?? "Unrecorded")} GPUs · ${e(meta.optimizer_steps_per_rollout ?? "Unrecorded")} optimizer updates / rollout</div>${healthFor(run)?.issue ? `<p class="run-issue">${e(healthFor(run).issue)} Ray: ${e(meta.status||"unknown")}.</p>` : ""}</div></div>`;
  }).join("")}</div><div class="callout">${runs.some(r=>r.metadata?.hardware?.engineering_sample === true) ? "Rubin is an engineering sample; its measured capacity and software stack define this comparison." : "Library versions and kernel choices can differ. The results describe each recorded system configuration."}</div>`, "Hardware comes from recorded device metadata. A partial snapshot never counts as a completed run.");

// Build schema supports components/changes/validation while retaining the full original input.
const components = build.components || build.stack || [];
const componentRows = Array.isArray(components) ? components : Object.entries(components).map(([name,value])=>({name,...(typeof value==="object" ? value : {version:value})}));
const component = pattern => componentRows.find(c=>pattern.test(c.name||c.component||""));
const baseComponent=component(/CUDA \/ Python/), sglangComponent=component(/SGLang Python/),
  teComponent=component(/Transformer Engine/), faComponent=component(/FlashAttention 2/),
  apexComponent=component(/^Apex$/), megatronComponent=component(/^Megatron/),
  bridgeComponent=component(/^mbridge/), flaComponent=component(/Gated DeltaNet/);
const matchVersion=(record,pattern)=>record?.version?.match(pattern)?.[1]||"Not recorded";
const shortPin=record=>record?.source_ref?.match(/[a-f0-9]{40}/)?.[0].slice(0,8)||"Not recorded";
const groupedBuildRows = [
  ["CUDA / Torch",baseComponent ? `${matchVersion(baseComponent,/CUDA ([^;]+)/)} / ${matchVersion(baseComponent,/torch ([^;]+)/)}`:null,"Inherited cu134 ARM64 base"],
  ["SGLang",sglangComponent ? shortPin(sglangComponent):null,"Miles source with retained native rollout libraries"],
  ["Transformer Engine",teComponent?.version,"Source build for SM107a and the selected Torch ABI"],
  ["FA2 / Apex",faComponent&&apexComponent ? `${faComponent.version} / ${shortPin(apexComponent)}`:null,"Native SM107 rebuilds with C++20"],
  ["Megatron-LM",megatronComponent ? shortPin(megatronComponent):null,"Preserved Miles fork"],
  ["mbridge / TMS",bridgeComponent ? `${matchVersion(bridgeComponent,/mbridge ([^;]+)/)} / ${matchVersion(bridgeComponent,/torch-memory-saver ([^;]+)/)}`:null,"Pinned mbridge, rebuilt TMS CUDA library"],
  ["FLA / Triton",flaComponent ? `${matchVersion(flaComponent,/fla-core ([^;]+)/)} / ${matchVersion(flaComponent,/triton ([^;]+)/)}`:null,"FLA serves Qwen3.5 GDN, outside standard Qwen3"],
].filter(row=>row[1]!=null);
slide("Rubin container and native components", "02 / Build process", `
  <p class="subtitle">The preserved image records the stack that actually ran on SM107.</p>
  ${groupedBuildRows.length ? table(["Component","Version / source pin","Build decision"],groupedBuildRows.map(row=>row.map(e)),"compact") : `<div style="height:450px">${pending("Build manifest pending","The report will show recorded versions and required changes once the build JSON is supplied.")}</div>`}
  <p class="chart-note">${groupedBuildRows.length ? "Stages: cu134 base → preserve native deps / ABI → rebuild TE / FA2 / Apex / TMS → pinned Python stack → GPU checks → GitLab digest." : "Build stages await the supplied build manifest."}</p>`, "Conceptual build stages, not an exact execution timeline. Full source pins and checks remain in the build evidence JSON.");

const changeSource = build.compatibility_fixes || build.fixes || build.required_changes || build.changes || [];
const changes = Array.isArray(changeSource) ? changeSource : Object.entries(changeSource).map(([title,value])=>({title,description:textValue(value)}));
const nativeFix=changes.find(c=>/native base/.test(c.title||""));
const d256Fix=changes.find(c=>/d256/.test(c.title||""));
const fa4Fix=changes.find(c=>/FA4/.test(c.title||""));
const selectedFixes=[nativeFix&&{title:"Preserved native dependencies",detail:nativeFix.detail,scope:"TE, Apex, FA2 and TMS rebuilt against the selected Torch environment."},
  d256Fix&&{title:"A narrow Qwen3.5 attention exception",detail:"Qwen3.5 cuDNN attention backward failed; FA2 required a narrow TE eligibility exception for SM107 BF16/d256/dropout0.",scope:"Python dispatch only. Standard Qwen3 uses d128 and does not need this exception."},
  fa4Fix&&{title:"FA4 package compatibility",detail:fa4Fix.detail,scope:"Selected FA2 paths passed. Other FA4 configurations remain untested."}].filter(Boolean);
slide("Compatibility fixes and their limits", "03 / Validation boundaries", selectedFixes.length ? `
  <div class="numbered">${selectedFixes.map(c=>`<article><div><h3>${e(c.title)}</h3><p>${e(c.detail)}</p><p class="small muted">${e(c.scope)}</p></div></article>`).join("")}</div>` : `<p class="subtitle">Each workaround needs a recorded trigger, a narrow scope, and an actual check.</p><div style="height:430px">${pending("Compatibility evidence pending","No undocumented fix or unverified kernel claim is added to the deck.")}</div>`, "A successful smoke test establishes only the tested scope. It does not validate all kernels or model families.");

function field(meta, names) {
  for (const name of names) {
    for (const source of [meta.recipe,meta]) if (source && source[name] != null) return textValue(source[name]);
  }
  return "Not recorded";
}
const recipeFields = [["Model",["model","model_name"]],["Dataset / reward",["dataset_reward","dataset"]],
  ["Prompts × samples",["batch_description","rollout_batch_size"]],["Response / prompt cap",["length_description","rollout_max_response_len","max_response_tokens"]],
  ["Learning rate",["learning_rate","lr"]],["Sampling",["sampling","sampling_description"]],["Group filtering",["group_filter","dynamic_sampling_filter_path","filter"]]];
function runtimeConditions(run) {
  const meta=run.metadata||{}, recipe=meta.recipe||{}, kernels=meta.kernels||{}, graph=meta.graph||{};
  if (!recipe.parallelism || !finite(recipe.max_training_tokens_per_gpu) || !kernels.sglang_attention || !kernels.sglang_moe || !kernels.sglang_bf16_gemm || typeof graph.rollout_cuda_graph!=="boolean" || typeof graph.rollout_piecewise_cuda_graph!=="boolean") return null;
  const backend=name=>({triton:"Triton",torch:"Torch"}[name]||name);
  const attention=kernels.sglang_attention===kernels.sglang_moe ? `${backend(kernels.sglang_attention)} attention + MoE` : `${backend(kernels.sglang_attention)} attention / ${backend(kernels.sglang_moe)} MoE`;
  const graphs=graph.rollout_cuda_graph===false&&graph.rollout_piecewise_cuda_graph===false ? "rollout CUDA graphs off" : `rollout CUDA graphs ${graph.rollout_cuda_graph?"on":"off"} / piecewise ${graph.rollout_piecewise_cuda_graph?"on":"off"}`;
  return `${recipe.parallelism} · ${number(recipe.max_training_tokens_per_gpu,0)} train tokens/GPU · ${attention} · ${backend(kernels.sglang_bf16_gemm)} GEMM · ${graphs}`;
}
const conditionRows=runs.map(run=>({name:niceName(run),text:runtimeConditions(run)})).filter(row=>row.text);
const conditionNote=conditionRows.length===runs.length && new Set(conditionRows.map(r=>r.text)).size===1
  ? `${runs.length>1?"Shared conditions":"Recorded conditions"}: ${conditionRows[0].text}.`
  : conditionRows.map(r=>`${r.name}: ${r.text}.`).join(" ");
slide("Qwen3 experiment recipe", "04 / Learning setup", `
  <p class="subtitle">The current comparison uses Qwen3-30B-A3B. Earlier Qwen3.5 tests are separate experiments.</p>
  ${table(["Setting",...displayRuns.map((r,i)=>r?niceName(r):i===0?"Rubin":"GB300")],recipeFields.map(([label,names])=>[e(label),...displayRuns.map(run=>e(field(run?.metadata||{},names)))]),"compact")}
  <p class="chart-note">${e(conditionNote || comparison.recipe_note || "Reward parsing, sampling, batch size, and output length belong to the comparison contract.")}</p>`, "Missing recipe values remain explicit. The historical VeRL run appears only in the appendix.");

slide("Training rollout reward", "05 / Learning curve", `
  <p class="subtitle">Observed reward from the retained training samples at each policy version</p>
  <div id="reward-chart" class="chart chart-wide"></div>`, "Rollout 0 samples the initial policy. Reward at each rollout precedes that rollout’s optimizer updates. This is not held-out accuracy.");

slide("Truncation and response length", "06 / Generation behavior", `
  <p class="subtitle">Output length changes the amount of work and can affect the reward curve.</p>
  <div class="chart-columns"><div><h3 class="chart-title">Engine-reported truncation</h3><div id="truncation-chart" class="chart half"></div></div><div><h3 class="chart-title">Mean output length</h3><div id="length-chart" class="chart half"></div></div></div>`, "Metrics describe retained samples. Dynamic filtering can select a different population from all generated responses.");

const unknownEvalPhase=runs.flatMap(run=>(run.rows||[]).flatMap(row=>(row.eval||[]).filter(event=>event.weight_phase==="unknown"||!event.weight_phase).map(()=>`${niceName(run)}: recorded weight phase unknown${(run.rows||[]).every(row=>!(row.train_steps||[]).length)?"; 0 observed optimizer updates":""}.`)));
const evalProgress=runs.flatMap(run=>{
  const events=(run.rows||[]).flatMap(row=>(row.eval||[]).map(event=>({...event,rollout_id:row.rollout_id}))).filter(event=>finite(event.metrics?.["eval/gsm8k"]));
  const first=events[0], latest=events.at(-1);
  if (events.length<2 || first.weight_phase!=="before_this_update" || latest.weight_phase!=="after_this_update") return [];
  const completed=(run.completed_training_rollouts||[]).filter(id=>id<=latest.rollout_id).length;
  return [`${niceName(run)} eval: ${pct(first.metrics["eval/gsm8k"])} → ${pct(latest.metrics["eval/gsm8k"])} after ${completed} completed rollouts.`];
});
slide("Gradient signal and evaluation", "07 / Learning evidence", `
  <p class="subtitle">Optimizer updates and evaluation events retain their own step and policy alignment.</p>
  <div class="chart-columns"><div><h3 class="chart-title">Optimizer gradient norm</h3><div id="gradient-chart" class="chart half"></div></div><div><h3 class="chart-title">Recorded evaluation</h3><div id="eval-chart" class="chart half"></div></div></div>${evalProgress.length||unknownEvalPhase.length?`<p class="chart-note">${e([...evalProgress,...new Set(unknownEvalPhase)].slice(0,2).join(" "))}</p>`:""}`, "A GRPO loss near zero can still produce nonzero gradients. Evaluation is labeled by its recorded metric and policy phase.");

const coldStartSteps=runs.map(run=>({name:niceName(run),seconds:(run.rows||[]).find(r=>r.rollout_id===0)?.common?.step_seconds})).filter(r=>finite(r.seconds));
const coldStartNote=coldStartSteps.length ? `Recorded rollout 0 step: ${coldStartSteps.map(r=>`${r.name} ${number(r.seconds,1)} s`).join("; ")}.` : "Cold-start step totals remain in the evidence JSON.";
slide("Runtime and generation throughput", "08 / Performance curves", `
  <p class="subtitle">Step timing excludes rollout 0 and uses only completed, eligible unprofiled observations.</p>
  <div class="chart-columns"><div><h3 class="chart-title">Miles step timer · after warmup</h3><div id="time-chart" class="chart half"></div></div><div><h3 class="chart-title">Generation throughput · raw</h3><div id="throughput-chart" class="chart half"></div></div></div>
  <p class="chart-note">Step = train wait + train. Generation keeps rollout 0; open circles mark warmup, diamonds mark profiling.</p>`, `${coldStartNote} Throughput = retained output tokens / generation seconds / GPUs. All raw timings remain downloadable.`);

slide("Time in each measured stage", "09 / Unprofiled timing", `
  <p class="subtitle">Means use only completed, explicitly unprofiled rollouts after the configured warmup exclusion.</p>
  <div id="stage-chart" class="chart short"></div>
  <div class="callout">Stage timers overlap. Their sum is not a wall-time breakdown.</div>`, "Unknown profiling coverage excludes a run from this summary. The exact count and median remain in the evidence JSON.");

function profileSlide(label, index) {
  const records=profiles.profiles || [];
  const item=records.find(p=>(p.run_label||p.title||"").toLowerCase().includes(label.toLowerCase()));
  const attachment=item?.attachments?.image;
  const observations=(item?.observations||[]).slice(0,3);
  slide(`${label} profiling evidence`, `${String(index).padStart(2,"0")} / Kernel timeline`, `
    <p class="subtitle">${e(item?.title || "The timeline will appear when a local capture and its scope are available.")}</p>
    <div class="profile-layout"><div class="profile-frame">${attachment?.status==="available" ? `<img src="${e(attachment.url)}" alt="${e(item.title||label+" profiler capture")}">` : pending("Profiler capture pending", "No synthetic timeline or invented Nsight screenshot substitutes for measured evidence.")}</div>
    <div class="profile-copy"><h3>${e(item?.tool || "Capture scope")}</h3><p>${e(textValue(item?.scope ?? "No capture supplied"))}</p>
      ${observations.length?`<ul>${observations.map(x=>`<li>${e(typeof x==="object"?x.text||x.observation||JSON.stringify(x):x)}</li>`).join("")}</ul>`:`<p class="muted">Gap observations remain pending until the trace supports them.</p>`}
      ${item?.attachments?.trace?.status==="available" ? `<p><a href="${e(item.attachments.trace.url)}" download>Download original trace</a></p>`:""}
      ${attachment?.status==="available"?`<p><a href="${e(attachment.url)}" target="_blank" rel="noopener">Open full-size evidence image</a></p>`:""}
    </div></div>`, item?.caption || "Capture overhead belongs to the profiled sample. Steady-state timing uses separate observations.");
}
profileSlide("Rubin",10);
profileSlide("GB300",11);

const findings=[];
for (const run of runs.slice(0,2)) {
  const valid=(run.rows||[]).filter(r=>finite(r.common?.training_reward_mean));
  findings.push([niceName(run),valid.length ? `${valid.length} reward observations. Initial ${pct(valid[0].common.training_reward_mean)}, latest ${pct(valid[valid.length-1].common.training_reward_mean)}. ${status(run).label}.` : `No training reward measurements in this snapshot.${healthFor(run)?.issue?` ${healthFor(run).issue}`:""}`]);
}
if (!findings.length) findings.push(["Learning","The current Miles runs have no supplied result records yet."]);
const profileSummary = profiles.summary || profiles.gap_summary;
findings.push(["Measured gap", profileSummary ? textValue(profileSummary) : "A causal GPU-speed conclusion is pending comparable timings and profile evidence."]);
findings.push(["Scope","This compares the recorded Rubin and upstream GB300 software stacks. Version and kernel differences remain part of the result."]);
slide("Findings at this snapshot", "12 / Conclusions", `
  <ul class="observations">${findings.slice(0,4).map(([label,body])=>`<li><strong>${e(label)}</strong><span class="body">${e(body)}</span></li>`).join("")}</ul>`, "Only supplied metrics and profile observations support these statements. No projected final reward or extrapolated speedup.");

function normalizeHistorical(source) {
  if (!source) return null;
  if (source.rows) return source;
  if (!source.steps) return null;
  return {metadata:source.configuration, source_log_sha256:source.source_log_sha256, rows:source.steps.map(s=>({rollout_id:s.rollout_step-1, common:{training_reward_mean:s.reward_mean, capacity_clip_ratio_not_engine_truncation:s.response_length_clip_ratio}}))};
}
const historical=normalizeHistorical(INPUT.historical);
slide("Historical VeRL reference", "Appendix A / Earlier GB300 experiment", `
  <p class="subtitle">The earlier “AMD vs NVIDIA” experiment supplies a learning reference. The new GB300 baseline uses Miles.</p>
  <div class="chart-columns"><div><h3 class="chart-title">Training reward</h3><div id="historical-reward-chart" class="chart half"></div></div><div><h3 class="chart-title">Response capacity clip ratio</h3><div id="historical-truncation-chart" class="chart half"></div></div></div>`, "VeRL step 1 aligns with Miles rollout 0. Capacity clipping differs from engine truncation. Whole-step VeRL throughput differs from Miles generation throughput.");

const hashPrefix=value=>value ? `${String(value).slice(0,24)}…` : "Pending";
const reproduction=build.reproduction||{};
const runEvidence=runs.slice(0,2).map(run=>{
  const image=textValue(run.metadata?.image || "Image digest not recorded");
  const imageHash=image.match(/@sha256:([a-f0-9]{64})/i)?.[1];
  const imageName=imageHash ? image.split("@")[0].split("/").slice(-2).join("/") : image;
  return `<div class="evidence-row"><h3>${e(niceName(run))}</h3><p>${image.includes("gitlab")?"GitLab · ":""}${e(imageName)}</p><p class="hash" title="${e(image)}">Image: ${e(hashPrefix(imageHash))}</p><p class="hash" title="${e(run.source_log_sha256)}">Log: ${e(hashPrefix(run.source_log_sha256))}</p></div>`;
}).join("");
slide("Source pins and reproduction evidence", "Appendix B / Provenance", `
  <div class="evidence-columns provenance-content content"><div><h3>Experiment records</h3>${runEvidence || `<p class="muted">Current run provenance is pending.</p>`}<p class="small evidence-download"><a href="report-data.json" download>Full recipes, raw metrics, and SHA256 hashes</a></p></div>
  <div><h3>Input integrity · SHA256 prefixes</h3>${REPORT.provenance.filter(p=>p.status==="loaded").map(p=>`<div class="input-evidence-row"><span>${e(p.section)} JSON</span><span class="hash" title="${e(p.sha256)}">${e(hashPrefix(p.sha256))}</span></div>`).join("") || `<p class="muted">No evidence inputs supplied.</p>`}
  <div class="build-entrypoints"><h3>Build entry points</h3><p>${e(reproduction.repository||"Repository pending")} · ${e(reproduction.branch||"branch pending")}</p><p class="hash">${e(reproduction.dockerfile||"Dockerfile not recorded")}</p><p class="hash">${e(reproduction.build_helper||"Build helper not recorded")}</p></div>
  <p class="small"><a href="reproduction.txt" download>Build paths &amp; exact GitLab pull command</a><br><a href="asset-manifest.json" download>All local asset checksums</a></p></div></div>`, "SHA256 prefixes identify the records shown; complete hashes and immutable image references remain in the downloadable evidence.");

// Plots keep missing values as gaps. The chart library is a pinned local asset.
const plotJobs=[];
function draw(id, traces, options={}) {
  const element=document.getElementById(id);
  const usable=traces.filter(t=>t.y.some(finite));
  if (!usable.length) { element.innerHTML=pending(options.emptyTitle||"Measurements pending",options.emptyMessage||"No recorded points for this metric in the supplied evidence.");return; }
  if (!window.Plotly) { element.innerHTML=pending("Local chart library unavailable","The underlying measurements remain in report-data.json.");return; }
  const layout={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",size:23,color:"#183043"},
    margin:{l:90,r:20,t:22,b:85},autosize:true,showlegend:true,
    legend:{orientation:"h",x:0,y:1.15,font:{size:21}},hovermode:"closest",
    xaxis:{title:{text:options.xTitle??"Rollout ID",font:{size:22}},gridcolor:"#e3e9ed",zeroline:false,tickfont:{size:21},automargin:true},
    yaxis:{title:{text:options.yTitle||"",font:{size:22}},gridcolor:"#e3e9ed",zeroline:false,tickfont:{size:21},automargin:true,...(options.percent?{tickformat:".0%",range:[0,1.04]}:{rangemode:"tozero"})},
    ...options.layout};
  const job=Plotly.newPlot(element,usable,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["select2d","lasso2d"],toImageButtonOptions:{format:"png",scale:2,filename:id}});
  plotJobs.push(job);
}
function timingEligible(run,row) {
  return row.rollout_id!==0 && row.training_stage_complete===true && row.profiled===false &&
    row.unprofiled_timing_eligible===true && !(run.metadata?.exclude_timing_rollouts||[]).includes(row.rollout_id);
}
function lines(key, {steady=false, markWarmup=false}={}) {
  return runs.map((run,i)=>{
    const rows=(run.rows||[]).filter(row=>!steady || timingEligible(run,row));
    return {name:niceName(run),type:"scatter",mode:"lines+markers",connectgaps:false,
      x:rows.map(r=>r.rollout_id),y:rows.map(r=>finite(r.common?.[key])?r.common[key]:null),
      line:{color:palette[i%palette.length],width:3},marker:{size:7,color:palette[i%palette.length],symbol:rows.map(r=>r.profiled?"diamond":markWarmup&&r.rollout_id===0?"circle-open":"circle")},
      customdata:rows.map(r=>`${markWarmup&&r.rollout_id===0?"Warmup / rollout 0 · ":""}${r.profiled==null?"Profiling coverage unknown":r.profiled?"Profiled":"Unprofiled"}`),
      hovertemplate:"Rollout %{x}<br>%{y:.5g}<br>%{customdata}<extra>%{fullData.name}</extra>"};
  });
}
draw("reward-chart",lines("training_reward_mean"),{percent:true,yTitle:"Reward"});
draw("truncation-chart",lines("truncated_ratio"),{percent:true,yTitle:"Fraction"});
draw("length-chart",lines("response_length_mean_tokens"),{yTitle:"Output tokens"});
draw("time-chart",lines("step_seconds",{steady:true}),{yTitle:"Seconds",emptyMessage:"No completed, eligible unprofiled step after rollout 0 is available yet."});
draw("throughput-chart",lines("output_tokens_per_gpu_generation_second",{markWarmup:true}),{yTitle:"Output tokens / GPU / s"});
draw("gradient-chart",runs.map((run,i)=>{
  const events=(run.rows||[]).flatMap(r=>r.train_steps||[]);
  return {name:niceName(run),type:"scatter",mode:"lines+markers",connectgaps:false,x:events.map(x=>x.logged_id),y:events.map(x=>finite(x.metrics?.["train/grad_norm"])?x.metrics["train/grad_norm"]:null),line:{color:palette[i],width:3},marker:{size:6}};
}),{xTitle:"Optimizer update ID",yTitle:"Gradient norm"});
const evalTraces=[];
for (const [i,run] of runs.entries()) {
  const events=(run.rows||[]).flatMap(r=>(r.eval||[]).map(event=>({...event,rollout_id:r.rollout_id})));
  // Miles emits each dataset score at eval/<dataset>; nested paths are diagnostics.
  const keys=[...new Set(events.flatMap(event=>Object.keys(event.metrics||{}).filter(key=>/^eval\/[^/]+$/.test(key) && !/-(none_reward_ratio|truncated_ratio)$/.test(key))))];
  for (const key of keys.slice(0,2)) {
    evalTraces.push({name:`${niceName(run)} ${key.replace(/^eval\//,"")}`,type:"scatter",mode:"markers",x:events.map(x=>x.rollout_id),y:events.map(x=>finite(x.metrics[key])?x.metrics[key]:null),marker:{size:11,color:palette[i],symbol:"diamond"},customdata:events.map(x=>x.weight_phase||"unknown"),hovertemplate:"Rollout %{x}<br>%{y:.5g}<br>%{customdata}<extra>%{fullData.name}</extra>"});
  }
}
draw("eval-chart",evalTraces,{yTitle:"Recorded evaluation metric",emptyTitle:"Evaluation evidence pending",emptyMessage:"Training reward stays separate. No held-out score is inferred from the training curve."});
const stageNames=[["rollout","Generation"],["actor_train","Actor update"],["log_probs","Old log prob"],["ref_log_probs","Reference"],["update_weights","Weight sync"]];
draw("stage-chart",runs.map((run,i)=>({name:niceName(run),type:"bar",marker:{color:palette[i]},x:stageNames.map(([,name])=>name),
  y:stageNames.map(([key])=>run.unprofiled_stage_statistics?.[key]?.mean_seconds ?? null),
  customdata:stageNames.map(([key])=>[run.unprofiled_stage_statistics?.[key]?.count ?? 0,run.unprofiled_stage_statistics?.[key]?.median_seconds ?? null]),
  hovertemplate:"%{x}<br>Mean %{y:.3f} s<br>n=%{customdata[0]}<br>Median %{customdata[1]:.3f} s<extra>%{fullData.name}</extra>"})),{xTitle:"",yTitle:"Mean seconds",layout:{barmode:"group"},emptyMessage:"Eligible timings require explicit profiling coverage and completed training. No unprofiled timing is assumed."});
for (const [id,key] of [["historical-reward-chart","training_reward_mean"],["historical-truncation-chart","capacity_clip_ratio_not_engine_truncation"]]) {
  draw(id,historical?[{name:"Historical GB300 / VeRL",type:"scatter",mode:"lines+markers",x:historical.rows.map(r=>r.rollout_id),y:historical.rows.map(r=>r.common?.[key]??null),line:{color:"#726198",width:3},marker:{size:5}}]:[],{percent:true,yTitle:id.includes("reward")?"Reward":"Capacity clip fraction",emptyMessage:"No historical baseline was supplied. It will remain separate from current Miles results."});
}

// Presentation navigation is independent of data completeness.
let current=Math.max(0,Math.min(slides.length-1,Number(location.hash.replace("#slide-",""))-1||0));
function resizeDeck() {
  const area=document.getElementById("stage");
  document.documentElement.style.setProperty("--scale",String(Math.min(area.clientWidth/1600,area.clientHeight/900)));
}
function show(index) {
  current=Math.max(0,Math.min(slides.length-1,index));
  document.querySelectorAll(".slide").forEach((s,i)=>{s.classList.toggle("active",i===current);s.setAttribute("aria-hidden",String(i!==current));});
  document.getElementById("counter").textContent=`${current+1} / ${slides.length}`;
  document.getElementById("prev").disabled=current===0;
  document.getElementById("next").disabled=current===slides.length-1;
  history.replaceState(null,"",`#slide-${current+1}`);
  document.querySelectorAll("#overview-grid button").forEach((b,i)=>b.classList.toggle("current",i===current));
  if(window.Plotly) document.querySelectorAll(".slide.active .js-plotly-plot").forEach(el=>Plotly.Plots.resize(el));
}
const overview=document.getElementById("overview");
function toggleOverview(force) { overview.hidden=force===undefined?!overview.hidden:!force;if(!overview.hidden) document.querySelectorAll("#overview-grid button")[current].focus(); }
document.getElementById("overview-grid").innerHTML=slides.map(s=>`<button data-slide="${s.id-1}"><small>${String(s.id).padStart(2,"0")} / ${slides.length}</small>${e(s.title)}</button>`).join("");
document.querySelectorAll("#overview-grid button").forEach(b=>b.addEventListener("click",()=>{show(Number(b.dataset.slide));toggleOverview(false);}));
document.getElementById("prev").onclick=()=>show(current-1);
document.getElementById("next").onclick=()=>show(current+1);
document.getElementById("overview-button").onclick=()=>toggleOverview();
document.getElementById("close-overview").onclick=()=>toggleOverview(false);
async function fullScreen() { try { if(document.fullscreenElement) await document.exitFullscreen();else await document.documentElement.requestFullscreen(); } catch (_) {} }
document.getElementById("fullscreen").onclick=fullScreen;
document.addEventListener("fullscreenchange",()=>{document.body.classList.toggle("fullscreen",!!document.fullscreenElement);resizeDeck();});
document.addEventListener("keydown",event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||/INPUT|TEXTAREA|SELECT/.test(event.target.tagName))return;
  if(event.key==="Escape"){toggleOverview(false);return;}
  if(event.key.toLowerCase()==="o"){event.preventDefault();toggleOverview();return;}
  if(!overview.hidden)return;
  if(["ArrowRight","PageDown"," "].includes(event.key)){event.preventDefault();show(current+1);}
  else if(["ArrowLeft","PageUp"].includes(event.key)){event.preventDefault();show(current-1);}
  else if(event.key==="Home"){event.preventDefault();show(0);}
  else if(event.key==="End"){event.preventDefault();show(slides.length-1);}
  else if(event.key.toLowerCase()==="f"){event.preventDefault();fullScreen();}
});
window.addEventListener("resize",resizeDeck);
window.addEventListener("hashchange",()=>show(Number(location.hash.replace("#slide-",""))-1||0));
resizeDeck();show(current);
Promise.allSettled(plotJobs).then(()=>{window.REPORT_READY=true;resizeDeck();});
window.REPORT_SLIDES=slides;
