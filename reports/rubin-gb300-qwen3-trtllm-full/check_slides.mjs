// Local-only visual and source-value QA for the NEW12-slide TRTLLM report.
import {createRequire} from 'node:module';
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const require=createRequire('/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json');
const {chromium}=require('playwright');
const root=path.dirname(fileURLToPath(import.meta.url));
const site=path.resolve(process.argv[2]||path.join(root,'site'));
const output=path.resolve(process.argv[3]||path.join(root,'qa-pending'));
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const page=await browser.newPage({viewport:{width:1600,height:958},deviceScaleFactor:1});
const errors=[],network=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
await page.goto(pathToFileURL(path.join(site,'index.html')).href);
await page.waitForFunction(()=>window.REPORT_READY===true);
const count=await page.locator('.slide').count(),checks=[];
for(let i=0;i<count;i++){
 await page.keyboard.press(i===0?'Home':'ArrowRight');
 await page.waitForTimeout(60);
 checks.push(await page.evaluate(()=>{
  const slide=document.querySelector('.slide.active'),b=slide.getBoundingClientRect(),scale=b.width/1600;
  const overflow=[...slide.querySelectorAll('h1,h2,h3,p,table,li,.chart,.new-profile-frame,.metric-strip,.conclusion-grid')].filter(el=>{
   if(el.closest('.js-plotly-plot'))return false;
   const r=el.getBoundingClientRect();return r.left<b.left+45*scale||r.right>b.right-45*scale||r.bottom>b.top+822*scale;
  }).map(el=>({tag:el.tagName,class:el.className,text:el.textContent.slice(0,100)}));
  return {id:slide.id,title:slide.getAttribute('aria-label'),overflow};
 }));
 await page.locator('.slide.active').screenshot({path:path.join(output,`slide-${String(i+1).padStart(2,'0')}.png`)});
}
const evidence=await page.evaluate(()=>{
 const report=JSON.parse(document.getElementById('report-data').textContent),failures=[];
 const runs=report.inputs.comparison?.runs||[],p=report.derived.performance;
 if(report.experiment_id!=='qwen3-trtllm-full-v1')failures.push('Wrong experiment');
 if(document.querySelectorAll('.slide').length!==12)failures.push('Expected12 slides');
 const bindings=report.inputs.experiment.run_bindings;
 for(const r of runs){
  if(!bindings.some(b=>b.label===r.label&&b.run_id===r.metadata.run_id))failures.push('Unbound source run');
  const name=r.label==='rubin'?'Rubin ES':'GB300',map=new Map(r.rows.map(x=>[x.rollout_id,x]));
  const ids=map.size?Array.from({length:Math.max(...map.keys())+1},(_,i)=>i):[];
  for(const [chart,key,timing]of [['reward-chart','training_reward_mean',false],['length-chart','response_length_mean_tokens',false],['truncation-chart','truncated_ratio',false],['time-chart','step_seconds',true],['generation-chart','rollout_seconds',true]]){
   const expected=ids.map(id=>(!timing||p.paired.rollout_ids.includes(id))&&Number.isFinite(map.get(id)?.common?.[key])?map.get(id).common[key]:null);
   const actual=(document.getElementById(chart).data||[]).find(t=>t.name===name);
   if(expected.some(Number.isFinite)&&(JSON.stringify(actual?.x)!==JSON.stringify(ids)||JSON.stringify(actual?.y)!==JSON.stringify(expected)||actual.connectgaps!==false))failures.push(chart+': source values/cohort/gaps changed');
  }
 }
 for(const platform of ['rubin','gb300'])if(!runs.some(r=>r.label===platform)){
  const name=platform==='rubin'?'Rubin ES':'GB300';
  for(const el of document.querySelectorAll('.chart'))if((el.data||[]).some(t=>t.name===name))failures.push('Unmeasured partner trace: '+el.id);
  if(report.status==='PENDING_OR_INTERIM'&&!document.querySelector('#slide-3 .interim-badge')?.textContent.includes(name))failures.push('Missing partner-pending label');
 }
 for(const id of ['reward-chart','length-chart','truncation-chart','eval-chart','logprob-chart','gradient-chart','time-chart','generation-chart']){
  const el=document.getElementById(id);if(!el.data?.length)continue;
  const ticks=[...el.querySelectorAll('.xtick text')].map(t=>Number(t.textContent.replace('−','-')));
  if(ticks.some(t=>!Number.isInteger(t)||t<0))failures.push(id+': fractional or negative ID tick');
 }
 if(runs.length<2&&Object.values(p.ratios).some(v=>v!==null))failures.push('Unmeasured comparison ratio');
 if(!runs.length){
  if(!document.getElementById('slide-1').textContent.includes('PENDING'))failures.push('Missing pending cover');
  for(const id of ['reward-chart','eval-chart','length-chart','truncation-chart','logprob-chart','gradient-chart','time-chart','generation-chart','stage-chart']){
   const el=document.getElementById(id);if(el.data?.length||!el.textContent.includes('PENDING / UNMEASURED'))failures.push(id+': unmeasured chart not pending');
  }
 }
 for(const platform of ['rubin','gb300'])for(const stage of ['prefill','decode']){
  const slot=document.querySelector(`[data-profile-platform="${platform}"][data-profile-stage="${stage}"]`),item=report.derived.profiles.runs[platform]?.profiles[stage],img=slot?.querySelector('img');
  if(!slot)failures.push('Missing profile slot');
  if(item){if(img?.getAttribute('src')!==item.attachments.image.url||!img.complete||!img.naturalWidth)failures.push('Wrong/missing profile image');}
  else if(img||!slot.textContent.includes('PENDING'))failures.push('Unmeasured profile not pending');
 }
 if(report.status==='FINAL'&&(!report.derived.validation.complete||!report.derived.profiles.matched))failures.push('Invalid final completion claim');
 if(/shared ids|eligible observation/i.test(document.getElementById('deck').textContent))failures.push('Unexplained cohort jargon');
 return {status:report.status,run_count:runs.length,profile_platforms:Object.keys(report.derived.profiles.runs),failures};
});
await page.keyboard.press('Home');await page.keyboard.press('Space');
const space=await page.locator('.slide.active').getAttribute('id');
await page.keyboard.press('End');const end=await page.locator('.slide.active').getAttribute('id');
await page.keyboard.press('o');const overview=await page.locator('#overview').isVisible();await page.keyboard.press('Escape');
await page.setViewportSize({width:390,height:844});await page.keyboard.press('Home');
await page.locator('.slide.active').screenshot({path:path.join(output,'mobile-cover.png')});
const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
const result={slides:count,errors,external_requests:network,checks,evidence,keyboard:{space,end,overview},mobile_horizontal_overflow:mobileOverflow};
await writeFile(path.join(output,'browser-checks.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));await browser.close();
if(count!==12||errors.length||network.length||checks.some(c=>c.overflow.length)||evidence.failures.length||mobileOverflow||space!=='slide-2'||end!=='slide-12'||!overview)process.exitCode=1;
