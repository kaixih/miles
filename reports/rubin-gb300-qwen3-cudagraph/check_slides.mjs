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
  const input=report.inputs,runs=input.comparison?.runs||[],failures=[];
  const oldIds=new Set(input.experiment.previous_run_ids||[]);
  const bindings=input.experiment.run_bindings||[];
  const chartIds=['reward-chart','truncation-chart','length-chart','gradient-chart','eval-chart','time-chart','throughput-chart','stage-chart'];
  if(!runs.length){
    if(!document.getElementById('slide-1').textContent.includes('PENDING / UNMEASURED'))failures.push('Pending cover missing');
    for(const id of chartIds){
      const element=document.getElementById(id);
      if((element.data||[]).length||!element.textContent.includes('PENDING / UNMEASURED'))failures.push(`${id}: missing-input chart must be empty and pending`);
    }
  }
  const timing=[];
  for(const run of runs){
    const runId=run.metadata?.run_id;
    if(oldIds.has(runId)||!bindings.some(b=>b.label===run.label&&b.run_id===runId))failures.push('Unbound or old run appeared');
    const label=run.metadata?.display_name||(/^rubin/i.test(run.label)?'Rubin':/^gb300/i.test(run.label)?'GB300':run.label);
    for(const [chart,key]of [['reward-chart','training_reward_mean'],['truncation-chart','truncated_ratio'],['length-chart','response_length_mean_tokens']]){
      const actual=(document.getElementById(chart).data||[]).find(t=>t.name===label);
      const expected=(run.rows||[]).map(row=>Number.isFinite(row.common?.[key])?row.common[key]:null);
      if(expected.some(Number.isFinite)&&(JSON.stringify(actual?.y)!==JSON.stringify(expected)||JSON.stringify(actual?.x)!==JSON.stringify((run.rows||[]).map(row=>row.rollout_id))))failures.push(`${chart}: changed source values`);
    }
    for(const [chart,key]of [['time-chart','step_seconds'],['throughput-chart','output_tokens_per_gpu_generation_second']]){
      const actual=(document.getElementById(chart).data||[]).find(t=>t.name===label);
      const accepted=(run.rows||[]).filter(r=>r.rollout_id!==0&&r.training_stage_complete===true&&r.profiled===false&&r.unprofiled_timing_eligible===true&&(run.completed_training_rollouts||[]).includes(r.rollout_id)&&!(run.metadata?.exclude_timing_rollouts||[]).includes(r.rollout_id)&&Number.isFinite(r.common?.[key]));
      const points=(actual?.x||[]).map((id,i)=>({id,y:actual.y[i]})).filter(p=>Number.isFinite(p.y));
      if(JSON.stringify(points)!==JSON.stringify(accepted.map(r=>({id:r.rollout_id,y:r.common[key]}))))failures.push(`${chart}: selection mismatch`);
      if(actual){
        const end=Math.max(...(run.rows||[]).map(r=>r.rollout_id));
        const slots=Array.from({length:end+1},(_,i)=>i);
        if(JSON.stringify(actual.x)!==JSON.stringify(slots)||actual.connectgaps!==false||slots.some(id=>!accepted.some(r=>r.rollout_id===id)&&actual.y[id]!==null))failures.push(`${chart}: excluded/missing timing slots are not gaps`);
      }
      timing.push({run:run.label,chart,finite_rollout_ids:points.map(p=>p.id)});
    }
  }
  const proofSlide=document.getElementById('slide-5').textContent;
  if(!(input.profiles?.graph_evidence||[]).some(p=>p.verified===true)&&!proofSlide.includes('Pending verification'))failures.push('Requested graph mode was presented as verified');
  const traces=(input.profiles?.profiles||[]).filter(p=>p.verified===true);
  if(!traces.length&&document.querySelectorAll('.profile-frame img').length)failures.push('Unverified or old profile image rendered');
  const diagnosticPairs=(input.diagnostics?.pairs||[]).filter(pair=>pair.verified===true);
  const diagnosticSlide=document.getElementById('slide-11').textContent;
  if(!diagnosticSlide.includes('prefill + decode')||/mean decode/i.test(diagnosticSlide))failures.push('HTTP generation duration is mislabeled as decode-only');
  for(const pair of diagnosticPairs){
    for(const mode of ['off','on']){
      const condition=pair[mode], values=condition?.generation_seconds||[];
      const mean=values.reduce((a,b)=>a+b,0)/values.length;
      if(condition?.timing_scope!=='http_request_prefill_decode_queue_response'||!values.length||values.some(v=>!Number.isFinite(v)||v<=0)||Math.abs(condition.generation_seconds_mean-mean)>1e-9||condition.sample_count!==values.length||'decode_ms' in condition)failures.push('HTTP generation raw/derived diagnostic mismatch');
    }
  }
  return {failures,run_count:runs.length,timing,paired_ids:report.derived.paired_timing.rollout_ids,verified_profile_count:traces.length,verified_generation_pairs:diagnosticPairs.length};
});
const report={slides:count,errors,external_requests:network,checks,evidence_checks:evidenceChecks,keyboard:{space,end,overview},mobile_horizontal_overflow:mobileOverflow};
await writeFile(path.join(output,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
await browser.close();
if(errors.length||network.length||checks.some(x=>x.overflow.length)||mobileOverflow||evidenceChecks.failures.length)process.exitCode=1;
