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
