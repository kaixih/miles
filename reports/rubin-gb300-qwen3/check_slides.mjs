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
  for (const run of runs.slice(0,2)) {
    const hardware=run.metadata?.hardware;
    if (hardware?.name && !hardwareText.includes(hardware.name)) failures.push(`${run.label}: device name missing`);
    if (typeof hardware?.memory_mib_per_gpu==='number' && !hardwareText.includes(hardware.memory_mib_per_gpu.toLocaleString('en-US'))) failures.push(`${run.label}: device memory missing`);
    if (hardware?.engineering_sample===true && !hardwareText.includes('Engineering sample')) failures.push(`${run.label}: engineering sample qualification missing`);
  }
  const timeTraces=document.getElementById('time-chart').data||[];
  const evalTraces=document.getElementById('eval-chart').data||[];
  if (evalTraces.some(trace=>/none_reward_ratio|num_training_samples|truncated_ratio/.test(trace.name))) failures.push('Evaluation diagnostics must not appear as score traces');
  const stepSelections=runs.map(run=>{
    const name=run.metadata?.display_name||(/rubin/i.test(run.label)?'Rubin':/gb300/i.test(run.label)?'GB300':run.label);
    const expected=(run.rows||[]).filter(row=>row.rollout_id!==0 && row.training_stage_complete===true && row.profiled===false && row.unprofiled_timing_eligible===true && !(run.metadata?.exclude_timing_rollouts||[]).includes(row.rollout_id));
    const actual=timeTraces.find(trace=>trace.name===name);
    const expectedX=expected.map(row=>row.rollout_id), expectedY=expected.map(row=>Number.isFinite(row.common?.step_seconds)?row.common.step_seconds:null);
    if (expectedY.some(Number.isFinite)) {
      if (JSON.stringify(actual?.x)!==JSON.stringify(expectedX) || JSON.stringify(actual?.y)!==JSON.stringify(expectedY)) failures.push(`${run.label}: steady step selection or numeric value mismatch`);
    } else if (actual) failures.push(`${run.label}: unexpected steady step trace`);
    return {run:run.label,plotted_rollouts:actual?.x||[],expected_rollouts:expectedX,rollout_zero_excluded:!(actual?.x||[]).includes(0)};
  });
  return {failures,step_selections:stepSelections,evaluation_series:evalTraces.map(trace=>trace.name),hardware_visible:!failures.some(x=>/device|engineering/.test(x))};
});
const report={slides:count,errors,external_requests:network,checks,evidence_checks:evidenceChecks,keyboard:{space,end,overview},mobile_horizontal_overflow:mobileOverflow};
await writeFile(path.join(output,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
await browser.close();
if(errors.length||network.length||checks.some(x=>x.overflow.length)||mobileOverflow||evidenceChecks.failures.length)process.exitCode=1;
