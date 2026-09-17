"use strict";
const data=JSON.parse(document.getElementById('trace-data').textContent);
window.COMPACT_DATA=data;
const begin=data.window.start_ms_relative_to_origin,duration=data.window.duration_ms;
document.getElementById('timeline-scope').textContent=`${duration.toFixed(3)} ms · all ${data.timeline.length} GPU events · ${data.tracks.length} stream${data.tracks.length===1?'':'s'}`;
document.getElementById('foot').textContent=`Kernel sum ${data.kernel_sum_ms.toFixed(3)} ms · kernel union ${data.kernel_union_ms.toFixed(3)} ms. Bars sum durations; overlapping work can count more than once.`;
const colors={gpu_kernel:'#087f96',gpu_memcpy:'#d4773e',gpu_memset:'#726198'};
const font={family:'Arial, sans-serif',size:19,color:'#183043'};
const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font,hovermode:'closest'};
const config={responsive:false,displayModeBar:false,displaylogo:false};
const traces=Object.entries(colors).filter(([kind])=>data.timeline.some(e=>e.kind===kind)).map(([kind,color])=>{
 const rows=data.timeline.filter(e=>e.kind===kind);
 return {type:'bar',orientation:'h',name:kind.replace('gpu_',''),x:rows.map(e=>e.clipped_duration_ms),
  base:rows.map(e=>e.relative_start_ms-begin),y:rows.map(e=>data.tracks.indexOf(e.track)),width:.34,
  marker:{color,line:{width:0}},customdata:rows.map(e=>[e.name,e.track,e.full_duration_ms]),
  hovertemplate:'%{customdata[0]}<br>%{customdata[1]}<br>Start %{base:.6f} ms in forward<br>Duration in window %{x:.6f} ms<extra>%{fullData.name}</extra>'};
});
const timeline=Plotly.newPlot('timeline',traces,{...base,width:574,height:382,barmode:'overlay',
 margin:{l:102,r:16,t:42,b:62},
 xaxis:{title:{text:'Milliseconds within forward',font:{size:19}},range:[0,duration],gridcolor:'#dfe7eb',nticks:6,zeroline:false},
 yaxis:{tickmode:'array',tickvals:data.tracks.map((_,i)=>i),ticktext:data.tracks.map(x=>x.replace(/^GPU \d+ \/ /,'')),
  range:[data.tracks.length-.5,-.5],gridcolor:'#dfe7eb',zeroline:false},
 legend:{orientation:'h',x:0,y:1.18,font:{size:18}}},config);
const kernels=data.top_kernels,max=Math.max(...kernels.map(k=>k.clipped_duration_ms));
const bars=Plotly.newPlot('kernels',[{type:'bar',orientation:'h',x:kernels.map(k=>k.clipped_duration_ms),
 y:kernels.map(k=>k.row_id),width:.60,marker:{color:'#087f96'},customdata:kernels.map(k=>[k.name,k.calls_intersecting_window]),
 text:kernels.map(k=>k.clipped_duration_ms.toFixed(3)),textposition:'outside',textfont:{size:18},cliponaxis:false,
 hovertemplate:'%{customdata[0]}<br>Kernel sum %{x:.6f} ms<br>%{customdata[1]} calls in this window<extra></extra>'}],
 {...base,width:808,height:382,margin:{l:424,r:55,t:20,b:62},showlegend:false,
 xaxis:{title:{text:'Cumulative kernel milliseconds',font:{size:19}},range:[0,max*1.18],gridcolor:'#dfe7eb',nticks:5,zeroline:false},
 yaxis:{tickmode:'array',tickvals:kernels.map(k=>k.row_id),ticktext:kernels.map(k=>`${k.row_id+1}. ${k.prefix}`),
  range:[kernels.length-.5,-.5],tickfont:{size:18},zeroline:false}},config);
Promise.all([timeline,bars]).then(()=>{window.TRACE_READY=true;});
