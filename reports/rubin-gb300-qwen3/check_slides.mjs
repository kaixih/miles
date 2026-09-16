// Local browser QA. Does not change inputs or access the network.
import {createRequire} from 'node:module';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
const require=createRequire('/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json');
const {chromium}=require('playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const site=path.resolve(process.argv[2]||path.join(root,'site'));
const output=path.resolve(process.argv[3]||path.join(root,'qa'));
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const page=await browser.newPage({viewport:{width:1600,height:958},deviceScaleFactor:1});
const errors=[],network=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',request=>{if(/^https?:/.test(request.url()))network.push(request.url());});
await page.goto(pathToFileURL(path.join(site,'index.html')).href);
await page.waitForFunction(()=>window.REPORT_READY===true);
const count=await page.locator('.slide').count();
const checks=[];
for(let i=0;i<count;i++) {
  await page.keyboard.press(i===0?'Home':'ArrowRight');
  await page.waitForTimeout(40);
  const result=await page.evaluate(()=>{
    const slide=document.querySelector('.slide.active'),bounds=slide.getBoundingClientRect();
    const scale=bounds.width/1600;
    const overflow=[...slide.querySelectorAll('h1,h2,h3,p,table,li,pre,.chart,.profile-layout,.observations')].filter(e=>{
      if(e.closest('.js-plotly-plot'))return false;
      const r=e.getBoundingClientRect();
      return r.right>bounds.right-45*scale || r.left<bounds.left+45*scale || r.bottom>bounds.top+822*scale;
    }).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,100)}));
    return {id:slide.id,title:slide.getAttribute('aria-label'),overflow};
  });
  checks.push(result);
  await page.locator('.slide.active').screenshot({path:path.join(output,`slide-${String(i+1).padStart(2,'0')}.png`)});
}
await page.keyboard.press('Home');
await page.keyboard.press('Space');
const space=await page.locator('.slide.active').getAttribute('id');
await page.keyboard.press('End');
const end=await page.locator('.slide.active').getAttribute('id');
await page.keyboard.press('o');
const overview=await page.locator('#overview').isVisible();
await page.keyboard.press('Escape');
await page.setViewportSize({width:390,height:844});
await page.keyboard.press('Home');
await page.locator('.slide.active').screenshot({path:path.join(output,'mobile-cover.png')});
const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
const evidenceChecks=await page.evaluate(()=>{
  const report=JSON.parse(document.getElementById('report-data').textContent);
  const runs=report.inputs.comparison?.runs||[], failures=[];
  const hardwareText=document.getElementById('slide-2').textContent;
  const recipeSlide=document.getElementById('slide-5');
  const budgetRow=[...recipeSlide.querySelectorAll('tbody tr')].find(row=>row.cells[0]?.textContent==='Training tokens / GPU');
  const budgets=runs.slice(0,2).map(run=>run.metadata?.recipe?.max_training_tokens_per_gpu ?? run.metadata?.max_training_tokens_per_gpu);
  for (const [i,budget] of budgets.entries()) {
    if (Number.isFinite(budget) && budgetRow?.cells[i+1]?.textContent!==String(budget)) failures.push(`${runs[i].label}: training-token budget missing or incorrect in recipe table`);
  }
  const budgetMismatch=new Set(budgets.filter(Number.isFinite)).size>1;
  const recipeWarning=recipeSlide.textContent.includes('not matched recipes');
  const performanceWarning=document.getElementById('slide-9').textContent.includes('No hardware-only speedup inference');
  if (recipeWarning!==budgetMismatch || performanceWarning!==budgetMismatch) failures.push('Training-budget mismatch warnings do not match the recorded recipe values');
  for (const run of runs.slice(0,2)) {
    const hardware=run.metadata?.hardware;
    if (hardware?.name && !hardwareText.includes(hardware.name)) failures.push(`${run.label}: device name missing`);
    if (typeof hardware?.memory_mib_per_gpu==='number' && !hardwareText.includes(hardware.memory_mib_per_gpu.toLocaleString('en-US'))) failures.push(`${run.label}: device memory missing`);
    if (hardware?.engineering_sample===true && !hardwareText.includes('Engineering sample')) failures.push(`${run.label}: engineering sample qualification missing`);
  }
  const timeTraces=document.getElementById('time-chart').data||[];
  const throughputTraces=document.getElementById('throughput-chart').data||[];
  const evalTraces=document.getElementById('eval-chart').data||[];
  if (evalTraces.some(trace=>/none_reward_ratio|num_training_samples|truncated_ratio/.test(trace.name))) failures.push('Evaluation diagnostics must not appear as score traces');
  for (const run of runs) {
    const name=run.metadata?.display_name||(/rubin/i.test(run.label)?'Rubin':/gb300/i.test(run.label)?'GB300':run.label);
    const events=(run.rows||[]).flatMap(row=>(row.eval||[]).map(event=>({...event,rollout_id:row.rollout_id}))).filter(event=>Number.isFinite(event.metrics?.['eval/gsm8k']));
    const actual=evalTraces.find(trace=>trace.name===`${name} gsm8k`);
    if (events.length && (JSON.stringify(actual?.x)!==JSON.stringify(events.map(e=>e.rollout_id)) || JSON.stringify(actual?.y)!==JSON.stringify(events.map(e=>e.metrics['eval/gsm8k'])) || JSON.stringify(actual?.customdata)!==JSON.stringify(events.map(e=>e.weight_phase||'unknown')))) failures.push(`${run.label}: evaluation score or policy phase mismatch`);
    if (events.some(e=>e.weight_phase==='unknown') && !document.getElementById('slide-8').textContent.includes(`${name}: recorded weight phase unknown`)) failures.push(`${run.label}: unknown evaluation phase is hidden`);
    const health=run.metadata?.run_health ? (typeof run.metadata.run_health==='object'?run.metadata.run_health:{issue:run.metadata.issue}) : (report.inputs.run_health?.runs||[]).find(record=>record.label===run.label&&record.run_id===run.metadata?.run_id);
    if (health?.issue && !hardwareText.includes(health.issue)) failures.push(`${run.label}: known run issue is hidden`);
  }
  const timingSelections=(traces,key,description)=>runs.map(run=>{
    const name=run.metadata?.display_name||(/rubin/i.test(run.label)?'Rubin':/gb300/i.test(run.label)?'GB300':run.label);
    const expected=(run.rows||[]).filter(row=>row.rollout_id!==0 && row.training_stage_complete===true && row.profiled===false && row.unprofiled_timing_eligible===true && !(run.metadata?.exclude_timing_rollouts||[]).includes(row.rollout_id));
    const actual=traces.find(trace=>trace.name===name);
    const finiteExpected=expected.filter(row=>Number.isFinite(row.common?.[key]));
    const expectedX=finiteExpected.map(row=>row.rollout_id), expectedY=finiteExpected.map(row=>row.common[key]);
    const finiteActual=(actual?.x||[]).map((id,index)=>({id,value:actual.y[index]})).filter(point=>Number.isFinite(point.value));
    const actualX=finiteActual.map(point=>point.id), actualY=finiteActual.map(point=>point.value);
    const recordedIds=(run.rows||[]).map(row=>row.rollout_id).filter(id=>Number.isInteger(id)&&id>=0);
    const slotIds=recordedIds.length?Array.from({length:Math.max(...recordedIds)+1},(_,id)=>id):[];
    const excludedIds=slotIds.filter(id=>!expectedX.includes(id));
    if (expectedY.some(Number.isFinite)) {
      if (JSON.stringify(actualX)!==JSON.stringify(expectedX) || JSON.stringify(actualY)!==JSON.stringify(expectedY)) failures.push(`${run.label}: ${description} selection or numeric value mismatch`);
      if (JSON.stringify(actual?.x)!==JSON.stringify(slotIds) || excludedIds.some(id=>actual.y[actual.x.indexOf(id)]!==null) || actual?.connectgaps!==false) failures.push(`${run.label}: ${description} excluded or missing observations must be explicit null gaps`);
    } else if (actual) failures.push(`${run.label}: unexpected ${description} trace`);
    return {run:run.label,plotted_rollouts:actualX,expected_rollouts:expectedX,null_gap_rollouts:actual?excludedIds:[],rollout_zero_excluded:!actualX.includes(0)};
  });
  const missingRows=timingRows({rows:[{rollout_id:1},{rollout_id:3}]});
  if (JSON.stringify(missingRows.map(row=>row.rollout_id))!=='[0,1,2,3]' || missingRows[0].timing_missing!==true || missingRows[2].timing_missing!==true || missingRows[1].timing_missing) failures.push('Timing slot regression: missing rollout IDs must remain gaps');
  const stepSelections=timingSelections(timeTraces,'step_seconds','steady step');
  const throughputSelections=timingSelections(throughputTraces,'output_tokens_per_gpu_generation_second','steady generation throughput');
  const paired=report.derived?.paired_timing;
  const pairKeys=['rollout','actor_train','log_probs','ref_log_probs','update_weights'];
  const pairLabels=['Generation','Actor update','Old log prob','Reference','Weight sync'];
  const eligibleForPair=run=>(run.rows||[]).filter(row=>row.rollout_id!==0 && row.training_stage_complete===true && row.profiled===false && row.unprofiled_timing_eligible===true && (run.completed_training_rollouts||[]).includes(row.rollout_id) && !(run.metadata?.exclude_timing_rollouts||[]).includes(row.rollout_id));
  const pairRows=runs.map(eligibleForPair);
  const expectedPairIds=runs.length===2 ? pairRows[0].map(row=>row.rollout_id).filter(id=>pairRows[1].some(row=>row.rollout_id===id)).sort((a,b)=>a-b) : [];
  const stageTraces=document.getElementById('stage-chart').data||[];
  if (JSON.stringify(paired?.rollout_ids)!==JSON.stringify(expectedPairIds) || paired?.count!==expectedPairIds.length) failures.push('Paired stage cohort differs from the independent input intersection');
  const median=values=>{const sorted=[...values].sort((a,b)=>a-b), middle=Math.floor(sorted.length/2);return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;};
  const near=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(b));
  const expectedStages=runs.map((run,index)=>Object.fromEntries(pairKeys.map(key=>{
    const values=expectedPairIds.map(id=>pairRows[index].find(row=>row.rollout_id===id)?.common?.[`${key}_seconds`]);
    return [key,values.length&&values.every(Number.isFinite)?{mean:values.reduce((a,b)=>a+b,0)/values.length,median:median(values),count:values.length}:null];
  })));
  for (const [index,run] of runs.entries()) {
    const name=run.metadata?.display_name||(/rubin/i.test(run.label)?'Rubin':/gb300/i.test(run.label)?'GB300':run.label);
    const trace=stageTraces.find(trace=>trace.name===name);
    const expectedY=pairKeys.map(key=>expectedStages.length===2&&expectedStages.every(stats=>stats[key])?expectedStages[index][key].mean:null);
    if (!expectedY.some(Number.isFinite)) {if(trace)failures.push(`${run.label}: paired bars must be pending`);continue;}
    if(JSON.stringify(trace?.x)!==JSON.stringify(pairLabels))failures.push(`${run.label}: paired stage labels differ`);
    pairKeys.forEach((key,i)=>{
      const expected=expectedY[i], actual=trace?.y[i], statistic=paired?.statistics?.[run.label]?.[key];
      if(expected===null){if(actual!==null)failures.push(`${run.label}: missing paired stage was filled`);return;}
      if(!near(actual,expected)||!near(statistic?.mean_seconds,expected)||!near(statistic?.median_seconds,expectedStages[index][key].median)||statistic?.count!==expectedPairIds.length||trace?.customdata[i]?.[0]!==expectedPairIds.length) failures.push(`${run.label}: paired ${key} bar/statistic differs from same-ID inputs`);
    });
  }
  if(expectedPairIds.length && !document.getElementById('paired-cohort').textContent.includes(expectedPairIds.join(', '))) failures.push('Shared stage cohort IDs are not visible');
  const decode=report.inputs.profiles?.decode_comparison;
  const verifiedDecode=decode?.schema==='sglang-same-nominal-decode-bs128-v1'&&['rubin','gb300'].every(label=>decode.provenance?.[label]?.verified===true);
  const decodeChart=document.getElementById('decode-gap-chart');
  const decodeBars=decodeChart?.data||[];
  if(Boolean(decodeChart)!==verifiedDecode)failures.push('Optional decode slide requires verified actual metrics');
  if(document.querySelectorAll('.slide').length!==(verifiedDecode?16:15))failures.push('Dynamic slide count is incorrect');
  let decodeChecks=null;
  if(verifiedDecode){
    const labels=['rubin','gb300'];
    const expected=labels.map(label=>{
      const elapsed=decode.forward_statistics[label].gpu_annotation_span_ms.values;
      const union=decode.forward_statistics[label].kernel_union_ms.values;
      return {label,n:elapsed.length,elapsed_ms:elapsed.reduce((a,b)=>a+b,0)/elapsed.length,
        union_ms:union.reduce((a,b)=>a+b,0)/union.length,
        remainder_ms:elapsed.reduce((sum,v,i)=>sum+v-union[i],0)/elapsed.length,
        trace_sha256:decode.provenance[label].trace_sha256};
    });
    if(decodeBars.length!==2)failures.push('Decode comparison must contain kernel-union and remainder bars');
    expected.forEach((row,i)=>{
      if(!near(decodeBars[0]?.y[i],row.union_ms)||!near(decodeBars[1]?.y[i],row.remainder_ms)||!near(decodeBars[0]?.y[i]+decodeBars[1]?.y[i],row.elapsed_ms))failures.push(`${row.label}: decode stack differs from raw forward intervals`);
      if(decodeBars[0]?.customdata[i]?.[0]!==row.n||decodeBars[0]?.customdata[i]?.[2]!==row.trace_sha256)failures.push(`${row.label}: decode N or source SHA mismatch`);
    });
    const scope=document.getElementById('decode-gap-scope')?.textContent||'';
    if(!scope.includes(`Rubin N=${expected[0].n}`)||!scope.includes(`GB300 N=${expected[1].n}`)||!scope.includes('not proven matched'))failures.push('Decode subset counts or workload caveat are hidden');
    const rejected=[];
    const missing=structuredClone(decode);missing.provenance.gb300.verified=false;
    if(window.decodeGapEvidence(missing)!==null)failures.push('Unverified decode evidence was accepted');else rejected.push('unverified');
    for(const [name,change] of [
      ['wrong_mean',d=>{d.forward_statistics.rubin.kernel_union_ms.mean+=1;}],
      ['union_exceeds_span',d=>{const m=d.forward_statistics.gb300.kernel_union_ms;m.values[0]=d.forward_statistics.gb300.gpu_annotation_span_ms.values[0]+1;m.mean=m.values[0];}],
      ['mismatched_count',d=>{d.selection.counts.gb300+=1;}],
      ['invalid_source_sha',d=>{d.provenance.gb300.trace_sha256='not-a-sha';}]
    ]){
      const invalid=structuredClone(decode);change(invalid);
      try{window.decodeGapEvidence(invalid);failures.push(`Invalid decode evidence accepted: ${name}`);}catch(_){rejected.push(name);}
    }
    decodeChecks={counts:decode.selection.counts,expected,bar_values:decodeBars.map(t=>({name:t.name,x:t.x,y:t.y})),negative_cases_rejected:rejected};
  }
  return {failures,decode_gap:decodeChecks,paired_stage_cohort:{rollout_ids:expectedPairIds,count:expectedPairIds.length,status:paired?.status,bar_values:stageTraces.map(trace=>({name:trace.name,x:trace.x,y:trace.y,customdata:trace.customdata}))},training_budget_comparison:{tokens_per_gpu:budgets,mismatch:budgetMismatch,recipe_warning:recipeWarning,performance_warning:performanceWarning},step_selections:stepSelections,generation_throughput_selections:throughputSelections,evaluation_series:evalTraces.map(trace=>trace.name),hardware_visible:!failures.some(x=>/device|engineering/.test(x))};
});
const report={slides:count,errors,external_requests:network,checks,evidence_checks:evidenceChecks,keyboard:{space,end,overview},mobile_horizontal_overflow:mobileOverflow};
await writeFile(path.join(output,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
await browser.close();
if(errors.length||network.length||checks.some(x=>x.overflow.length)||mobileOverflow||evidenceChecks.failures.length)process.exitCode=1;
