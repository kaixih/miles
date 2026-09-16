// CPU-only rendering checks against real report.js; synthetic fixtures never become a deck.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';
const root=path.dirname(fileURLToPath(import.meta.url));
const source=readFileSync(path.join(root,'assets/report.js'),'utf8');
const experiment=JSON.parse(readFileSync(path.join(root,'experiment.json'),'utf8'));
function render(pairs){
  const report={inputs:{experiment,comparison:null,profiles:null,diagnostics:{pairs}},derived:{paired_timing:{count:0,rollout_ids:[],statistics:{}}},provenance:[]};
  const slides=[],elements=new Map([['report-data',{textContent:JSON.stringify(report)}],['deck',{insertAdjacentHTML:(_,html)=>slides.push(html)}]]);
  const document={getElementById(id){if(!elements.has(id))elements.set(id,{innerHTML:''});return elements.get(id);}};
  vm.runInNewContext(source,{document,console},{timeout:1000});
  assert.equal(slides.length,16);
  return slides[10];
}
const unavailable={platform:'rubin',verified:false,status:'unavailable',status_reason:'ON capture retained; OFF did not run; see receipt.',on:{generation_seconds_mean:123.456}};
const verified={platform:'gb300',verified:true,status:'available',off:{timing_verified:true,generation_seconds_mean:8},on:{timing_verified:true,generation_seconds_mean:4}};
const mixed=render([unavailable,verified]);
assert.match(mixed,/data-diagnostic-platform="rubin" data-diagnostic-status="unavailable"/);
assert.match(mixed,/ON capture retained; OFF did not run; see receipt/);
assert.match(mixed,/data-diagnostic-platform="gb300" data-diagnostic-status="available"/);
assert.match(mixed,/>8 s</);assert.match(mixed,/>4 s</);assert.ok(!mixed.includes('123.456'));
assert.match(mixed,/prefill \+ decode/);assert.ok(!/mean decode/i.test(mixed));
for(const label of experiment.requested.platforms)assert.match(render([]),new RegExp(`data-diagnostic-platform="${label}" data-diagnostic-status="pending"`));
assert.match(render([{...unavailable,status_reason:'<unsafe>'}]),/&lt;unsafe&gt;/);
const onOnly=render([{...unavailable,actual_trace_condition_proof:{on_decode_replay_observed:true},off:null,on:{timing_verified:true,generation_seconds_mean:1.015}},verified]);
assert.match(onOnly,/data-diagnostic-platform="rubin" data-diagnostic-status="partial" data-diagnostic-verified="false"/);
assert.match(onOnly,/<td data-mode="off">Not collected<\/td><td data-mode="on">1.015 s<\/td>/);
assert.match(onOnly,/ON measured; replay verified/);
console.log('PASS: independently measured ON shown without inventing OFF or a matched pair; unverified timing hidden; HTTP scope and escaping.');
