"use strict";
const data=JSON.parse(document.getElementById('trace-data').textContent);
const start=data.window.start_ms_relative_to_origin,end=start+data.window.duration_ms;
document.getElementById('title').textContent=data.title;
document.getElementById('scope').textContent=`Window ${start.toFixed(3)}–${end.toFixed(3)} ms from ${data.origin_used.replaceAll('_',' ')}. ${data.counts.selected_gpu_events.toLocaleString()} GPU events, ${data.tracks.length} streams. ${data.visible_event_cap_applied ? `Timeline shows the first ${data.counts.visible_gpu_events} events.` : 'All selected GPU events shown.'} CPU events remain separate.`;
document.getElementById('provenance').textContent=`Source: ${data.source.path}\nSHA256: ${data.source.sha256}. Categories in the complete trace: ${JSON.stringify(data.counts.complete_events_by_kind)}. Only complete events (ph=X). Timestamp unit: µs. Original events and exact window are in trace-summary.json.`;
const colors={gpu_kernel:'#087f96',gpu_memcpy:'#d4773e',gpu_memset:'#726198'};
const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',size:17,color:'#183043'},margin:{l:180,r:30,t:40,b:65},hovermode:'closest'};
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
const tasks=[];
if(data.timeline.length){
  const traces=Object.entries(colors).map(([kind,color])=>{const rows=data.timeline.filter(e=>e.kind===kind);return {type:'bar',orientation:'h',name:kind.replace('gpu_',''),x:rows.map(e=>e.clipped_duration_ms),base:rows.map(e=>e.relative_start_ms),y:rows.map(e=>e.track),width:.68,marker:{color},customdata:rows.map(e=>[e.name,e.full_duration_ms]),hovertemplate:'%{customdata[0]}<br>%{y}<br>Start %{base:.6f} ms<br>In-window %{x:.6f} ms<br>Full duration %{customdata[1]:.6f} ms<extra>%{fullData.name}</extra>'};});
  tasks.push(Plotly.newPlot('timeline',traces,{...base,barmode:'overlay',xaxis:{title:{text:'Milliseconds from trace origin'},range:[start,end],gridcolor:'#e1e8ec'},yaxis:{categoryorder:'array',categoryarray:[...data.tracks].reverse(),gridcolor:'#e1e8ec'},legend:{orientation:'h',x:0,y:1.15}},config));
}else document.getElementById('timeline').innerHTML='<div class="empty">No classified GPU events intersect this window. Select another window or inspect the trace categories.</div>';
const kernels=data.top_kernels.slice(0,8).reverse();
if(kernels.length){
  const label=name=>name.length>96?name.slice(0,93)+'…':name;
  tasks.push(Plotly.newPlot('kernels',[{type:'bar',orientation:'h',x:kernels.map(k=>k.clipped_duration_ms),y:kernels.map(k=>label(k.name)),marker:{color:'#087f96'},customdata:kernels.map(k=>[k.name,k.calls_intersecting_window]),hovertemplate:'%{customdata[0]}<br>Total %{x:.6f} ms<br>%{customdata[1]} calls<extra></extra>'}],{...base,margin:{l:840,r:50,t:5,b:55},xaxis:{title:{text:'Cumulative kernel milliseconds (clipped to window)'},gridcolor:'#e1e8ec'},yaxis:{tickfont:{size:13},automargin:true},showlegend:false},config));
}else document.getElementById('kernels').innerHTML='<div class="empty">No kernel events in this window.</div>';
Promise.allSettled(tasks).then(()=>{window.TRACE_READY=true;});
